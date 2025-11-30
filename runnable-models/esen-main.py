from __future__ import annotations

import os
import modal

app = modal.App("fairchem-softening-app")

CACHE_DIR = "/data"
WEIGHTS_AND_DATA_VOLUME = modal.Volume.from_name(
    "softening-volume", create_if_missing=True
)

# 1. INSTALLATION: Pin fairchem-core==1.10.0 
# Confirmed by MatBench YAML config (model_version: fairchem_core-1.10.0)
FAIRCHEM_IMAGE = (
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
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        find_links="https://data.pyg.org/whl/torch-2.4.0+cu121.html",
    )
    .uv_pip_install("fairchem-core==1.10.0")
)

@app.cls(
    image=FAIRCHEM_IMAGE,
    volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME},
    timeout=3600, 
    gpu="T4"
)
class FairchemSoftening:
    @modal.enter()
    def load_model(self):
        import sys
        import os
        import warnings
        import importlib
        import pkgutil
        
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        from fairchem.core.common.registry import registry
        from huggingface_hub import hf_hub_download

        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

        # 2. AGGRESSIVE REGISTRY POPULATION
        # The registry is empty by default in scripts. We must force-import 
        # the 'esen' submodule to trigger its @register_model decorators.
        
        # Standard path for 1.10.0 as implied by the YAML
        esen_path = "fairchem.core.models.esen" 
        
        print(f"Attempting to import {esen_path}...", file=sys.stderr)
        try:
            # Import the package itself
            package = importlib.import_module(esen_path)
            print(f"✓ Imported {esen_path}", file=sys.stderr)
            
            # CRITICAL STEP: Walk the package and import all submodules.
            # This ensures 'esen_backbone.py' (or similar) is executed.
            prefix = package.__name__ + "."
            if hasattr(package, "__path__"):
                for _, name, _ in pkgutil.walk_packages(package.__path__, prefix):
                    try:
                        importlib.import_module(name)
                        # print(f"  Imported: {name}", file=sys.stderr) # verbose
                    except Exception:
                        pass
        except ImportError as e:
            print(f"! Critical Import Error: {e}", file=sys.stderr)
            print("Attempting fallback: Import ALL models...", file=sys.stderr)
            import fairchem.core.models
            if hasattr(fairchem.core.models, "__path__"):
                 for _, name, _ in pkgutil.walk_packages(fairchem.core.models.__path__, fairchem.core.models.__name__ + "."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass

        # Check if 'esen_backbone' specifically is registered now
        keys = list(registry.mapping['model_name_mapping'].keys())
        if 'esen_backbone' not in keys:
             print(f"! Warning: 'esen_backbone' NOT found in registry. Keys: {keys}", file=sys.stderr)
        else:
             print(f"✓ 'esen_backbone' verified in registry.", file=sys.stderr)

        filename = "esen_30m_oam.pt"
        checkpoint_path = f"{CACHE_DIR}/{filename}"
        
        if not os.path.exists(checkpoint_path):
            print(f"Downloading {filename}...", file=sys.stderr)
            token = os.environ.get("HF_TOKEN")
            hf_hub_download(
                repo_id="facebook/OMAT24", 
                filename=filename, 
                token=token,
                local_dir=CACHE_DIR, 
                local_dir_use_symlinks=False
            )

        print(f"Loading eSEN-30M-OAM...", file=sys.stderr)
        
        try:
            # 3. TRAINER OVERRIDE
            # Use trainer="ocp" to bypass the internal 'mlip_trainer' name
            self.calc = OCPCalculator(
                checkpoint_path=checkpoint_path, 
                cpu=False,
                trainer="ocp"
            )
            print("✓ Model loaded successfully.", file=sys.stderr)
        except Exception as e:
            print(f"FAILED to load model: {e}", file=sys.stderr)
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
        
        # Count total structures for progress bar
        total_structs = sum(len(v) for v in data.values() if isinstance(v, dict))
        print(f"Found {len(data)} WBM groups ({total_structs} total structures).", file=sys.stderr)
        
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
                    
                    atoms.calc = self.calc
                    model_forces = atoms.get_forces()
                    
                    all_dft_forces.append(dft_forces.flatten())
                    all_model_forces.append(model_forces.flatten())
                    num_structures += 1
                    
                    if num_structures % 100 == 0:
                        print(f"Processed {num_structures}/{total_structs}...", file=sys.stderr)

                except Exception as e:
                    # Only print first few errors to avoid log spam
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
    print("Deploying Fairchem/eSEN Softening App (v1.10.0 + Recursive Import)...")
    result = FairchemSoftening().calculate_softening_factor.remote()
    
    if result:
        print("\n--- eSEN-30M-OAM Results ---")
        print(f"Structures: {result['num_structures']}")
        print(f"Slope: {result['slope']:.4f} (Softening Factor)")
        print(f"MAE: {result['mae']:.4f}")
        print(f"cMAE: {result['cmae']:.4f}")
    else:
        print("No results returned.")