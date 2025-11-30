from __future__ import annotations

import modal
import os
import sys

app = modal.App("mattersim-softening-app")

CACHE_DIR = "/data"
WEIGHTS_AND_DATA_VOLUME = modal.Volume.from_name(
    "softening-volume", create_if_missing=True
)

# MatterSim Image
MATTERSIM_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git", "git-lfs")
    .pip_install("uv")
    .pip_install(
        "torch==2.4.0",
        "numpy<2.0.0",
        "scikit-learn",
        "pymatgen",
        "ase",
        "mattersim",
    )
)

@app.cls(
    image=MATTERSIM_IMAGE,
    volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME},
    timeout=3600,
    gpu="T4",
)
class MatterSimSoftening:
    def _download_model(self, model_path, model_filename):
        import shutil
        import subprocess
        import os
        import sys
        
        print(f"Cloning model to {model_path}...", file=sys.stderr)
        try:
            # Clone repo to temp dir
            temp_dir = "/tmp/mattersim_repo"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            subprocess.run(["git", "lfs", "install"], check=True)
            # Clone only the pretrained_models directory if possible, but sparse checkout is complex.
            # Just clone depth 1.
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/microsoft/mattersim.git", temp_dir], check=True)
            
            # Debug: List files
            print("Listing files in cloned repo:", file=sys.stderr)
            subprocess.run(["find", temp_dir, "-maxdepth", "3"], check=False)
            
            # Move the file
            src_path = f"{temp_dir}/pretrained_models/{model_filename}"
            if os.path.exists(src_path):
                shutil.move(src_path, model_path)
                print(f"Model moved to {model_path}", file=sys.stderr)
            else:
                raise FileNotFoundError(f"Model file not found in cloned repo at {src_path}")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"Failed to clone/copy model: {e}", file=sys.stderr)
            raise

    @modal.enter()
    def setup(self):
        import torch
        from mattersim.forcefield import MatterSimCalculator
        import os
        import sys
        
        print("Loading MatterSim-v1.0.0-5M...", file=sys.stderr)
        
        # Define model path in the volume
        model_filename = "mattersim-v1.0.0-5M.pth"
        model_path = f"{CACHE_DIR}/{model_filename}"
        
        # Check if model exists, if not download it
        if not os.path.exists(model_path):
            self._download_model(model_path, model_filename)
        
        # Load the model
        try:
            self.calc = MatterSimCalculator(load_path=model_path, device="cuda")
            print("✓ MatterSim model loaded successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Error loading model: {e}. Deleting corrupted file and retrying...", file=sys.stderr)
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Retry download
            self._download_model(model_path, model_filename)
            
            # Retry load
            self.calc = MatterSimCalculator(load_path=model_path, device="cuda")
            print("✓ MatterSim model loaded successfully after retry.", file=sys.stderr)

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
        print(f"Found {len(data)} WBM groups ({total_structs} structures). Processing with MatterSim...", file=sys.stderr)
        
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
    print("Deploying MatterSim Softening App...")
    try:
        result = MatterSimSoftening().calculate_softening_factor.remote()
        
        if result:
            print("\n--- MatterSim-v1.0.0-5M Results ---")
            print(f"Structures: {result['num_structures']}")
            print(f"Slope: {result['slope']:.4f} (Softening Factor)")
            print(f"MAE: {result['mae']:.4f}")
            print(f"cMAE: {result['cmae']:.4f}")
        else:
            print("No results returned.")
    except Exception as e:
        print(f"Execution failed: {e}")
