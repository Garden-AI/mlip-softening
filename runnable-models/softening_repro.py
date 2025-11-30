from __future__ import annotations

import modal

app = modal.App("softening-app")

CACHE_DIR = "/data"
WEIGHTS_AND_DATA_VOLUME = modal.Volume.from_name(
    "softening-volume", create_if_missing=True
)

MACE_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "mace-torch>=0.3.10", # Required for newer foundation models
        "torch==2.2.2",       # Per README, >2.1 is needed
        "ase",                # For handling Atoms objects
        "scikit-learn",       # For linear regression
        "numpy<2",            # Pin numpy<2 for compatibility with mace-torch compiled extensions
        "pymatgen",           # To parse structures from the JSON file
    )
)

MODEL_VARIANTS = {
    "MACE-MP-0": "medium",
    "MACE-MPA-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "MACE-OMAT-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "MACE-MatPES-PBE-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
}

@app.cls(
    image=MACE_IMAGE,
    volumes={
        CACHE_DIR: WEIGHTS_AND_DATA_VOLUME,
    },
    timeout=3600, # 30 minute timeout
    gpu="T4"
)
class SOFTENING_CLASS:
    @modal.enter()
    def setup(self):
        """
        Initialize state when the container starts.
        Models are loaded lazily in calculate_softening_factor.
        """
        self.calc = None
        self.current_variant = None

    def _load_model(self, variant: str):
        import sys
        from mace.calculators import mace_mp

        if self.calc is not None and self.current_variant == variant:
            print(f"Model {variant} already loaded.", file=sys.stderr)
            return

        model_arg = MODEL_VARIANTS.get(variant)
        if not model_arg:
            raise ValueError(f"Unknown variant: {variant}. Available: {list(MODEL_VARIANTS.keys())}")

        print(f"Loading MACE model variant '{variant}' ({model_arg}) onto GPU...", file=sys.stderr)
        
        # Load the specified MACE model
        # We specify device='cuda' to ensure it runs on the T4 GPU.
        self.calc = mace_mp(
            model=model_arg, 
            default_dtype="float32", 
            device='cuda'
        )
        self.current_variant = variant
        print(f"MACE model '{variant}' loaded.", file=sys.stderr)

    @modal.method()
    def calculate_softening_factor(self, variant: str = "MACE-MP-0") -> dict:
        """
        Processes all structures in the JSON file to calculate the
        aggregate softening scale (slope) of MACE forces vs. DFT forces.
        """
        self._load_model(variant)
        import json
        import sys

        import numpy as np
        from pymatgen.core import Structure
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error

        all_dft_forces = []
        all_mace_forces = []
        
        data_path = f"{CACHE_DIR}/WBM_high_energy_states.json"
        print(f"Loading data from {data_path}...", file=sys.stderr)
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {data_path}", file=sys.stderr)
            print("Please ensure 'WBM_high_energy_states.json' is in the 'softening-volume' Modal Volume.", file=sys.stderr)
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {data_path}.", file=sys.stderr)
            return {}

        print(f"Found {len(data)} WBM groups. Processing all sub-structures...", file=sys.stderr)
        
        num_structures = 0
        for wbm_id, wbm_data in data.items():
            if not isinstance(wbm_data, dict):
                print(f"Skipping non-dict top-level entry: {wbm_id}", file=sys.stderr)
                continue

            # Iterate through the nested dictionary (e.g., "wbm-5-9051-8")
            for struct_id, entry in wbm_data.items():
                try:
                    # 1. Load ground-truth DFT forces from the correct path
                    # *** THIS IS THE FIX based on the screenshots ***
                    dft_forces = np.array(entry['vasp_f'])
                    
                    # 2. Load structure
                    structure_dict = entry['structure']
                    pmg_struct = Structure.from_dict(structure_dict)
                    
                    # Convert Pymatgen Structure to ASE Atoms object for MACE
                    atoms = pmg_struct.to_ase_atoms()
                    
                    # 3. Run MACE inference
                    atoms.calc = self.calc
                    mace_forces = atoms.get_forces()
                    
                    # 4. Store flattened forces for aggregate regression
                    all_dft_forces.append(dft_forces.flatten())
                    all_mace_forces.append(mace_forces.flatten())
                    
                    num_structures += 1
                    
                    if num_structures % 100 == 0:
                        print(f"Processed {num_structures} structures...", file=sys.stderr)

                except KeyError as e:
                    print(f"KeyError processing {wbm_id}/{struct_id}: {e}. Skipping this sub-structure.", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing {wbm_id}/{struct_id}: {e}. Skipping this sub-structure.", file=sys.stderr)
        
        if num_structures == 0:
            print("Error: No valid structures were processed from the JSON file.", file=sys.stderr)
            return {}

        print(f"Processing complete. {num_structures} total sub-structures processed.", file=sys.stderr)
        print("Performing linear regression...", file=sys.stderr)

        # Concatenate all force components into single 1D arrays
        y_true_all = np.concatenate(all_dft_forces) # DFT forces (X-axis)
        y_pred_all = np.concatenate(all_mace_forces) # MACE forces (Y-axis)
        
        # Reshape for scikit-learn's LinearRegression
        # We fit: MACE_force = slope * DFT_force + intercept
        X = y_true_all.reshape(-1, 1)
        y = y_pred_all.reshape(-1, 1)
        
        reg = LinearRegression().fit(X, y)
        
        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        
        # Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        
        # Calculate cMAE (corrected MAE)
        # Defined as the MAE after correcting the slope to 1
        y_pred_corrected = y_pred_all / slope
        cmae = mean_absolute_error(y_true_all, y_pred_corrected)
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "mae": float(mae),
            "cmae": float(cmae),
            "num_structures": int(num_structures),
            "num_force_components": int(len(y_true_all)),
        }

# Define the local entrypoint
@app.local_entrypoint()
def main(variant: str = "MACE-MP-0"):
    print(f"Running softening analysis for variant: {variant}")
    result = SOFTENING_CLASS().calculate_softening_factor.remote(variant=variant)
    print(f"\n--- {variant} Aggregate Softening Results ---")
    print(f"Processed {result['num_structures']} structures from WBM dataset")
    print(f"({result['num_force_components']:,} total force components)")
    print("\n--- Linear Regression (MACE_forces vs. DFT_forces) ---")
    print(f"Slope (Softening Scale): {result['slope']:.4f}")
    print(f"Intercept: {result['intercept']:.4f} eV/Å")
    print(f"MAE: {result['mae']:.4f} eV/Å")
    print(f"cMAE (Corrected MAE): {result['cmae']:.4f} eV/Å")