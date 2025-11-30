from __future__ import annotations

import modal
import os
import sys

app = modal.App("consolidated-softening-app")

CACHE_DIR = "/data"
WEIGHTS_AND_DATA_VOLUME = modal.Volume.from_name(
    "softening-volume", create_if_missing=True
)

# ==============================================================================
# CONSTANTS & CONFIG
# ==============================================================================

MACE_VARIANTS = {
    "MACE-MP-0": "medium",
    "MACE-MPA-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
    "MACE-OMAT-0": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
    "MACE-MatPES-PBE-0": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
}

# ==============================================================================
# IMAGE DEFINITIONS
# ==============================================================================

# 1. MACE Image
MACE_IMAGE = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "mace-torch>=0.3.10",
        "torch==2.2.2",
        "ase",
        "scikit-learn",
        "numpy<2",
        "pymatgen",
    )
)

# 2. Fairchem (eSEN) Image
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

# 3. UMA Image
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
        "torch-geometric", 
    )
    .uv_pip_install("fairchem-core>=2.0.0")
)

# 4. TensorNet Image
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

# 5. MatterSim Image
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

# ==============================================================================
# SHARED UTILITY FUNCTION
# ==============================================================================

def run_softening_analysis(calculator, data: dict) -> dict:
    """
    Common logic to run softening analysis given a calculator and data dictionary.
    
    Args:
        calculator: An ASE calculator object (or compatible) ready for inference.
        data: Dictionary containing WBM high energy states.
        
    Returns:
        A dictionary containing regression results (slope, intercept, mae, cmae, etc.)
    """
    import sys
    import numpy as np
    from pymatgen.core import Structure
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    all_dft_forces = []
    all_model_forces = []
    
    # Count total structures for progress bar
    total_structs = sum(len(v) for v in data.values() if isinstance(v, dict))
    print(f"Found {len(data)} WBM groups ({total_structs} total structures). Processing...", file=sys.stderr)
    
    num_structures = 0
    for wbm_id, wbm_data in data.items():
        if not isinstance(wbm_data, dict):
            continue

        # Iterate through the nested dictionary (e.g., "wbm-5-9051-8")
        for struct_id, entry in wbm_data.items():
            try:
                if 'vasp_f' not in entry or 'structure' not in entry:
                    continue

                # 1. Load ground-truth DFT forces
                dft_forces = np.array(entry['vasp_f'])
                
                # 2. Load structure
                structure_dict = entry['structure']
                pmg_struct = Structure.from_dict(structure_dict)
                
                # Convert Pymatgen Structure to ASE Atoms object
                atoms = pmg_struct.to_ase_atoms()
                
                # 3. Run inference
                atoms.calc = calculator
                model_forces = atoms.get_forces()
                
                # 4. Store flattened forces for aggregate regression
                all_dft_forces.append(dft_forces.flatten())
                all_model_forces.append(model_forces.flatten())
                
                num_structures += 1
                
                if num_structures % 100 == 0:
                    print(f"Processed {num_structures}/{total_structs}...", file=sys.stderr)

            except Exception as e:
                # Only print first few errors to avoid log spam
                if num_structures < 5:
                    print(f"Error processing {wbm_id}/{struct_id}: {e}. Skipping.", file=sys.stderr)
    
    if num_structures == 0:
        print("Error: No valid structures were processed from the input data.", file=sys.stderr)
        return {}

    print(f"Processing complete. {num_structures} total sub-structures processed.", file=sys.stderr)
    print("Performing linear regression...", file=sys.stderr)

    # Concatenate all force components into single 1D arrays
    y_true_all = np.concatenate(all_dft_forces) # DFT forces (X-axis)
    y_pred_all = np.concatenate(all_model_forces) # Model forces (Y-axis)
    
    # Reshape for scikit-learn's LinearRegression
    # We fit: Model_force = slope * DFT_force + intercept
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

# ==============================================================================
# MODEL CLASSES
# ==============================================================================

@app.cls(image=MACE_IMAGE, volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME}, timeout=3600, gpu="T4")
class MaceModel:
    @modal.enter()
    def setup(self):
        self.calc = None
        self.current_variant = None

    def _load_model(self, variant: str):
        from mace.calculators import mace_mp
        import sys
        
        if self.calc is not None and self.current_variant == variant:
            return
        
        # Resolve variant to URL or keyword
        model_arg = MACE_VARIANTS.get(variant, variant)
        print(f"Loading MACE model: {variant} ({model_arg})", file=sys.stderr)
        
        self.calc = mace_mp(model=model_arg, default_dtype="float32", device='cuda')
        self.current_variant = variant

    @modal.method()
    def calculate_softening_factor(self, data: dict, model_name: str) -> dict:
        self._load_model(model_name)
        return run_softening_analysis(self.calc, data)

@app.cls(image=FAIRCHEM_IMAGE, volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME}, timeout=3600, gpu="T4")
class EsenModel:
    @modal.enter()
    def setup(self):
        self.calc = None
        self.current_model_name = None

    def _load_model(self, model_name: str):
        if self.calc is not None and self.current_model_name == model_name:
            return

        import os
        import importlib
        import pkgutil
        import warnings
        import sys
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator
        from huggingface_hub import hf_hub_download

        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

        # Registry population logic (only needs to run once, but safe to run again)
        esen_path = "fairchem.core.models.esen"
        try:
            package = importlib.import_module(esen_path)
            prefix = package.__name__ + "."
            if hasattr(package, "__path__"):
                for _, name, _ in pkgutil.walk_packages(package.__path__, prefix):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass
        except ImportError:
            import fairchem.core.models
            if hasattr(fairchem.core.models, "__path__"):
                 for _, name, _ in pkgutil.walk_packages(fairchem.core.models.__path__, fairchem.core.models.__name__ + "."):
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass

        # If model_name is a filename in the cache/HF, use it. 
        # Default was "esen_30m_oam.pt"
        filename = model_name
        if not filename.endswith(".pt") and not filename.endswith(".pth"):
             filename = f"{model_name}.pt" # Assumption

        checkpoint_path = f"{CACHE_DIR}/{filename}"
        
        # If not in cache, try to download from OMAT24 repo (assuming it's there)
        if not os.path.exists(checkpoint_path):
            print(f"Downloading {filename} from facebook/OMAT24...", file=sys.stderr)
            try:
                token = os.environ.get("HF_TOKEN")
                hf_hub_download(repo_id="facebook/OMAT24", filename=filename, token=token, local_dir=CACHE_DIR, local_dir_use_symlinks=False)
            except Exception as e:
                print(f"Failed to download {filename}: {e}", file=sys.stderr)
                raise

        print(f"Loading eSEN model from {checkpoint_path}...", file=sys.stderr)
        self.calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False, trainer="ocp")
        self.current_model_name = model_name

    @modal.method()
    def calculate_softening_factor(self, data: dict, model_name: str) -> dict:
        self._load_model(model_name)
        return run_softening_analysis(self.calc, data)

@app.cls(image=UMA_IMAGE, volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME}, timeout=3600, gpu="T4") #, secrets=[modal.Secret.from_name("huggingface-secret")])
class UmaModel:
    @modal.enter()
    def setup(self):
        self.calc = None
        self.current_model_name = None

    def _load_model(self, model_name: str):
        if self.calc is not None and self.current_model_name == model_name:
            return

        import torch
        import os
        import sys
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        
        # os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN") 
        print(f"Loading UMA model: {model_name}...", file=sys.stderr)
        
        self.predictor = pretrained_mlip.get_predict_unit(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.calc = FAIRChemCalculator(self.predictor, task_name="omat")
        self.current_model_name = model_name

    @modal.method()
    def calculate_softening_factor(self, data: dict, model_name: str) -> dict:
        self._load_model(model_name)
        return run_softening_analysis(self.calc, data)

@app.cls(image=TENSORNET_IMAGE, volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME}, timeout=3600, gpu="T4")
class TensorNetModel:
    @modal.enter()
    def setup(self):
        self.calc = None
        self.current_model_name = None

    def _load_model(self, model_name: str):
        if self.calc is not None and self.current_model_name == model_name:
            return
        
        import matgl
        from matgl.ext.ase import PESCalculator
        import sys
        
        print(f"Loading TensorNet/MatGL model: {model_name}...", file=sys.stderr)
        self.model = matgl.load_model(model_name)
        self.calc = PESCalculator(potential=self.model)
        self.current_model_name = model_name

    @modal.method()
    def calculate_softening_factor(self, data: dict, model_name: str) -> dict:
        self._load_model(model_name)
        return run_softening_analysis(self.calc, data)

@app.cls(image=MATTERSIM_IMAGE, volumes={CACHE_DIR: WEIGHTS_AND_DATA_VOLUME}, timeout=3600, gpu="T4")
class MatterSimModel:
    def _download_model(self, model_path, model_filename):
        import shutil
        import subprocess
        import os
        
        temp_dir = "/tmp/mattersim_repo"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/microsoft/mattersim.git", temp_dir], check=True)
        src_path = f"{temp_dir}/pretrained_models/{model_filename}"
        if os.path.exists(src_path):
            shutil.move(src_path, model_path)
        else:
            raise FileNotFoundError(f"Model file not found in cloned repo at {src_path}")
        shutil.rmtree(temp_dir)

    @modal.enter()
    def setup(self):
        self.calc = None
        self.current_model_name = None

    def _load_model(self, model_name: str):
        if self.calc is not None and self.current_model_name == model_name:
            return
            
        from mattersim.forcefield import MatterSimCalculator
        import os
        import sys
        
        # model_name should be the filename, e.g. "mattersim-v1.0.0-5M.pth"
        model_filename = model_name
        if not model_filename.endswith(".pth"):
             model_filename = f"{model_name}.pth"

        model_path = f"{CACHE_DIR}/{model_filename}"
        
        if not os.path.exists(model_path):
            print(f"Model {model_filename} not found in cache. Attempting to download...", file=sys.stderr)
            self._download_model(model_path, model_filename)
        
        print(f"Loading MatterSim model from {model_path}...", file=sys.stderr)
        try:
            self.calc = MatterSimCalculator(load_path=model_path, device="cuda")
        except Exception:
            print("Load failed. Retrying download...", file=sys.stderr)
            if os.path.exists(model_path): os.remove(model_path)
            self._download_model(model_path, model_filename)
            self.calc = MatterSimCalculator(load_path=model_path, device="cuda")
            
        self.current_model_name = model_name

    @modal.method()
    def calculate_softening_factor(self, data: dict, model_name: str) -> dict:
        self._load_model(model_name)
        return run_softening_analysis(self.calc, data)

# ==============================================================================
# ENTRYPOINT
# ==============================================================================

@app.local_entrypoint()
def main(model: str = "mace", file: str = "WBM_high_energy_states.json", variant: str = None):
    """
    Run softening analysis for a specific model.
    
    Args:
        model: Model family (mace, esen, uma, tensornet, mattersim)
        file: Path to local JSON file to load and pass to the remote function (default: WBM_high_energy_states.json)
        variant: Specific model variant/name to load. Defaults vary by model family.
    """
    import json
    import os
    
    model = model.lower()
    
    # Set defaults if variant is not provided
    if variant is None:
        if model == "mace": variant = "MACE-MP-0"
        elif model == "esen": variant = "esen_30m_oam.pt"
        elif model == "uma": variant = "uma-m-1p1"
        elif model == "tensornet": variant = "TensorNet-MatPES-PBE-v2025.1-PES"
        elif model == "mattersim": variant = "mattersim-v1.0.0-5M.pth"
    
    print(f"Loading local data from {file}...")
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file}")
        return

    print(f"Running softening analysis for model: {model}, variant: {variant}")
    
    result = None
    if model == "mace":
        result = MaceModel().calculate_softening_factor.remote(data=data, model_name=variant)
    elif model == "esen":
        result = EsenModel().calculate_softening_factor.remote(data=data, model_name=variant)
    elif model == "uma":
        result = UmaModel().calculate_softening_factor.remote(data=data, model_name=variant)
    elif model == "tensornet":
        result = TensorNetModel().calculate_softening_factor.remote(data=data, model_name=variant)
    elif model == "mattersim":
        result = MatterSimModel().calculate_softening_factor.remote(data=data, model_name=variant)
    else:
        print(f"Unknown model: {model}. Available: mace, esen, uma, tensornet, mattersim")
        return

    if result:
        print(f"\n--- {model.upper()} ({variant}) Results ---")
        print(f"Structures: {result['num_structures']}")
        print(f"Slope: {result['slope']:.4f} (Softening Factor)")
        print(f"MAE: {result['mae']:.4f}")
        print(f"cMAE: {result['cmae']:.4f}")
    else:
        print("No results returned.")
