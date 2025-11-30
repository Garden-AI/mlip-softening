from __future__ import annotations

import modal
import os

app = modal.App("uma-softening-app")

CACHE_DIR = "/data"
WEIGHTS_AND_DATA_VOLUME = modal.Volume.from_name(
    "softening-volume", create_if_missing=True
)

# UMA requires fairchem-core >= 2.0.0
# We pin to the latest release to ensure UMA support
UMA_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential", "cmake")
    .pip_install("uv")
    .uv_pip_install(
        "torch==2.4.0",
        "numpy<2.0.0",
        "scikit-learn",
        "pymatgen",
        "ase",
        "huggingface_hub",
        # fairchem-core 2.0+ drops the complex pyg dependencies!
        # But we still need torch-geometric for some backend operations
        "torch-geometric", 
    )
    .uv_pip_install("fairchem-core>=2.0.0") # <--- CRITICAL: Use v2+ for UMA
)

@app.cls(
    image=UMA_IMAGE,
    volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME},
    timeout=3600,
    gpu="T4",
    # secrets=[modal.Secret.from_name("huggingface-secret")] # Ensure you have this secret set!
)
class UMA_CLASS:
    @modal.enter()
    def setup(self):
        import sys
        import torch
        # NEW IMPORT PATH for V2
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        print("Loading UMA Model...", file=sys.stderr)
        import os
        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")  
        
        # UMA instantiation is different from OCP/eSEN.
        # 1. Load the "predict unit" (the model weights)
        # Valid names: 'uma-s-1p1' (small), 'uma-m-1p1' (medium)
        try:
            model_name = "uma-m-1p1"
            self.predictor = pretrained_mlip.get_predict_unit(
                model_name, 
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # 2. Wrap it in the ASE Calculator
            # task_name='omat' is for inorganic materials (Open Materials)
            self.calc = FAIRChemCalculator(
                self.predictor, 
                task_name="omat"
            )
            print("✓ UMA model loaded successfully.", file=sys.stderr)
            
        except Exception as e:
            print(f"FAILED to load UMA: {e}", file=sys.stderr)
            # Common failure is missing HF_TOKEN
            if "401" in str(e) or "unauthorized" in str(e).lower():
                 print("Hint: Ensure you have access to UMA on Hugging Face and the HF_TOKEN secret is set.", file=sys.stderr)
            raise e

    @modal.method()
    def calculate_softening_factor(self) -> dict:
        import json
        import sys
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
        print(f"Found {len(data)} WBM groups ({total_structs} structures). Processing with UMA...", file=sys.stderr)
        
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
                    
                    # Attach UMA calculator
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
    print("Deploying UMA Softening App...")
    result = UMA_CLASS().calculate_softening_factor.remote()
    
    if result:
        print("\n--- UMA (Small) Results ---")
        print(f"Structures: {result['num_structures']}")
        print(f"Slope: {result['slope']:.4f} (Softening Factor)")
        print(f"MAE: {result['mae']:.4f}")
        print(f"cMAE: {result['cmae']:.4f}")
    else:
        print("No results returned.")