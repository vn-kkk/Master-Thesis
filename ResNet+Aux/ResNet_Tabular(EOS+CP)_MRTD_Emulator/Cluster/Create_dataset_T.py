################################################################################
# Import all the required modules
################################################################################
import os
import re
import glob

import numpy as np

from tqdm import tqdm


################################################################################
# Create dataset of input and output samples from the files
################################################################################
# ==============================================================================
# 1. Dataset Configuration
# ==============================================================================
NUM_FILES_TO_USE = 100 
DATA_DIR_PATH = "/cluster/users/venkatek/ML/All_MR_Relations/All_MR_Relations/"
save_dir_dataset = f"{NUM_FILES_TO_USE}files"

# Regex to extract parameters
pattern = re.compile(
    r"MREoSm(?P<m>\d+)"
    r"L(?P<L>\d+)"
    r"J(?P<J>\d+)"
    r"T(?P<T>\d+)_"
    r"n(?P<n>\d+)_"
    r"Yp?\+.*?RGgen_"
    r"v(?P<n_v>[-\d\.]+)"
    r"d(?P<d>[-\d\.]+)"
    r"B(?P<Btype>[np])(?P<B>\d+)\.dat"
)

# ==============================================================================
# 2. Processing files
# ==============================================================================
all_files = glob.glob(os.path.join(DATA_DIR_PATH, "MR*.dat"))
if not all_files: raise RuntimeError("No files found.")

# Shufflling files
np.random.shuffle(all_files)
selected_files = all_files[:NUM_FILES_TO_USE]

# Split files into Train and Validation lists (80 / 20)
split_idx = int(0.8 * len(selected_files))
train_files = selected_files[:split_idx]
val_files = selected_files[split_idx:]

def process_files(file_list, label_desc):
    dataset_rows = []
    files_used_for_training = [] # Save files used for training so we can ommit
                                 # these later on when testing the model
    
    for file_path in tqdm(file_list, desc=f"Processing {label_desc}"):
        filename = os.path.basename(file_path)
        match = pattern.match(filename)
        if not match: continue

        files_used_for_training.append(file_path)

        # 1. Extract All input parameters
        m = float(match.group("m")) / 100.0 
        L = float(match.group("L"))
        J = float(match.group("J"))             
        T = float(match.group("T")) # Unused but extracted for completeness             
        n = float(match.group('n')) / 1000.0
        n_v = float(match.group("n_v"))
        d = float(match.group("d"))
        B = float(match.group("B")) / 1000.0
        
        # 2. Load the injection and output parameters from each file
        try:
            data = np.loadtxt(file_path)
        except: continue

        if data.ndim == 1 or data.shape[1] < 3: continue

        # Columns: [Central pressure (P_c), Mass, Radius, Tidal Deformability,
        # R_quarkcore, Baryonic mass (rho)]
        cp = data[:, 0]
        mass = data[:, 1]
        radius = data[:, 2]
        td = data[:, 3]

        # 3. Filter out Unstable Branch
        # We only want data up to the Maximum Mass. 
        # Usually Data is sorted by central pressure.
        # M increases then decreases.
        max_m_index = np.argmax(mass)
        
        # Truncate BEFORE the noisy peak at M_max
        # Drop the last 2 points from the calculated stable branch
        SAFETY_MARGIN = 2 
        
        # Ensure we don't end up with negative index
        end_index = max(1, max_m_index + 1 - SAFETY_MARGIN)
        
        # Slice arrays to only keep stable, safe branch
        cp = cp[:end_index]
        mass = mass[:end_index] 
        radius = radius[:end_index]
        td = td[:end_index]

        # 4. Apply Minimum Mass Cutoff (Filter low-mass instability)
        M_CUTOFF = 0.15 # Filter out anything below 0.15 solar masses
        low_mass_mask = mass >= M_CUTOFF
        cp= cp[low_mass_mask]
        mass = mass[low_mass_mask]
        radius = radius[low_mass_mask]
        td = td[low_mass_mask]

        # 5. Basic filtering to ensure positive values
        valid_mask = (radius > 0) & (mass > 0)
        cp = cp[valid_mask]
        mass = mass[valid_mask]
        radius = radius[valid_mask]
        td = td[valid_mask]

        if len(mass) == 0: continue

        # 6. Create feature vector for every point
        # Features: [m, L, J, n_v, d, B, n, cp] -> Predict: Mass, Radius, TD
        num_points = len(mass)
        eos_params = np.array([m, L, J, n_v, d, B, n])
        
        # 7. Tile EOS params to match number of mass points
        eos_repeated = np.tile(eos_params, (num_points, 1))

        # 8. Stack: [EOS_Params (7), Central Pressure, Mass, Radius, TD]
        rows = np.column_stack([eos_repeated, cp, mass, radius, td])
        dataset_rows.append(rows)

    if not dataset_rows: 
        return None
    
    return np.vstack(dataset_rows), np.array(files_used_for_training)

print("Building Training Set...")
train_data = process_files(train_files, "Train")
print("Building Validation Set...")
val_data = process_files(val_files, "Val")
print("Datasets created!")

print(train_data[1].shape)
print(val_data[1].shape)

# Save files used for training and validation to be used during testing of model
files_used_for_training = np.concatenate([train_data[1], val_data[1]]) if train_data and val_data is not None else []

# Save all datasets
os.makedirs(save_dir_dataset, exist_ok=True)    # Create folder if it doesn't exist

np.save(os.path.join(save_dir_dataset, "train_data_files.npy"), train_data[0])
np.save(os.path.join(save_dir_dataset, "val_data_files.npy"), val_data[0])
np.save(os.path.join(save_dir_dataset, "files_used_for_training.npy"), files_used_for_training)

print("Datasets saved!")

