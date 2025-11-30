from __future__ import annotations

import modal
import os
import sys

app = modal.App("tensornet-softening-app")

CACHE_DIR = "/data"
WEIGHTS_AND_DATA_VOLUME = modal.Volume.from_name(
    "softening-volume", create_if_missing=True
)

# Define the image with necessary dependencies
# MatGL is migrating to PyG, so we install from GitHub to get the latest version.
# We also include torch-geometric and related dependencies.
TENSORNET_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install("uv")
    .pip_install(
        "torch==2.4.0",
        "numpy<2.0.0",
    )
    .pip_install(
        "scikit-learn",
        "pymatgen",
        "ase",
        "huggingface_hub",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    )
    .pip_install("git+https://github.com/materialsvirtuallab/matgl.git")
)

@app.cls(
    image=TENSORNET_IMAGE,
    volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME},
    timeout=3600,
    gpu="T4",
)
class TensorNetSoftening:
    @modal.enter()
    def setup(self):
        import torch
        import matgl
        # from matgl.ext.ase import M3GNetCalculator # Removed as it causes ImportError
        
        # Note: As of late 2024/early 2025, MatGL's ASE calculator might be unified or specific.
        # We will try to load the specific model.
        
        print("Loading TensorNet-MatPES-PBE-v2025.1-PES...", file=sys.stderr)
        
        try:
            # Load the model using matgl.load_model
            # This handles downloading from HF or local cache
            # The model name provided by user: TensorNet-MatPES-PBE-v2025.1-PES
            self.model = matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")
            
            # Create ASE calculator
            from matgl.ext.ase import PESCalculator
            self.calc = PESCalculator(potential=self.model)
            
            print("✓ TensorNet model loaded successfully.", file=sys.stderr)
            
        except Exception as e:
            print(f"FAILED to load TensorNet: {e}", file=sys.stderr)
            # Fallback to listing available models if possible or just fail
            raise e

    @modal.method()
    def calculate_softening_factor(self) -> dict:
        import json
        import numpy as np
        from pymatgen.core import Structure
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error
        
        all_dft_forces = []
        all_model_forces = []
        
        data_path = f"{CACHE_DIR}/WBM_high_energy_states.json"
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}", file=sys.stderr)
            return {}
            
        total_structs = sum(len(v) for v in data.values() if isinstance(v, dict))
        print(f"Found {len(data)} WBM groups ({total_structs} structures). Processing with TensorNet...", file=sys.stderr)
        
        num_structures = 0
        for wbm_id, wbm_data in data.items():
            if not isinstance(wbm_data, dict): continue

            for struct_id, entry in wbm_data.items():
                try:
                    if 'vasp_f' not in entry or 'structure' not in entry: continue

                    dft_forces = np.array(entry['vasp_f'])
                    structure_dict = entry['structure']
                    pmg_struct = Structure.from_dict(structure_dict)
                    atoms = pmg_struct.to_ase_atoms()
                    
                    # Attach Calculator
                    atoms.calc = self.calc
                    
                    # Inference
                    model_forces = atoms.get_forces()
                    
                    all_dft_forces.append(dft_forces.flatten())
                    all_model_forces.append(model_forces.flatten())
                    num_structures += 1
                    
                    if num_structures % 100 == 0:
                        print(f"Processed {num_structures}/{total_structs}...", file=sys.stderr)

                except Exception as e:
                    if num_structures < 5:
                        print(f"Error on {wbm_id}/{struct_id}: {e}", file=sys.stderr)
                    pass
        
        if num_structures == 0:
            print("No structures processed.", file=sys.stderr)
            return {}

        print(f"Processing complete. Analyzing {num_structures} structures...", file=sys.stderr)
        
        y_true = np.concatenate(all_dft_forces)
        y_pred = np.concatenate(all_model_forces)
        
        reg = LinearRegression().fit(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
        slope = reg.coef_[0][0]
        mae = mean_absolute_error(y_true, y_pred)
        cmae = mean_absolute_error(y_true, y_pred / slope)
        
        return {
            "slope": float(slope),
            "intercept": float(reg.intercept_[0]),
            "mae": float(mae),
            "cmae": float(cmae),
            "num_structures": num_structures
        }

@app.local_entrypoint()
def main():
    print("Deploying TensorNet Softening App...")
    try:
        result = TensorNetSoftening().calculate_softening_factor.remote()
        
        if result:
            print("\n--- TensorNet-MatPES-PBE-v2025.1-PES Results ---")
            print(f"Structures: {result['num_structures']}")
            print(f"Slope: {result['slope']:.4f} (Softening Factor)")
            print(f"MAE: {result['mae']:.4f}")
            print(f"cMAE: {result['cmae']:.4f}")
        else:
            print("No results returned.")
    except Exception as e:
        print(f"Execution failed: {e}")
