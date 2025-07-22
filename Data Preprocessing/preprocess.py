import open3d as o3d
import numpy as np
import pandas as pd
import os
import sys
from plyfile import PlyData, PlyElement

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- Configuration ---
# IMPORTANT: Adjust these paths to your downloaded EDF dataset location
dataset_root_path = 'C:/RUTUL/GBC/WIP/Dataset/EDF Industrial Facility' # <<< VERIFY THIS PATH CAREFULLY

global_labels_file = 'ytrain_i9bpfD4.csv'
master_map_file = 'ytrain_map_ind_station.csv'
ply_folder_name_fixed = 'xtrain_kW4SLO1' # Or 'xtrain_kW4SLO1' if that's the literal folder name

# Define class colors (as before, for visualization purposes in the demo)
CLASS_COLORS = {
    0: [0.5, 0.5, 0.5],  # Background (grey)
    1: [1.0, 0.0, 0.0],  # Beams (red)
    2: [0.0, 1.0, 0.0],  # Cabletrays (green)
    3: [0.0, 0.0, 1.0],  # Civils (blue - for walls/floors)
    4: [1.0, 1.0, 0.0],  # Gratings (yellow)
    5: [1.0, 0.5, 0.0],  # Guardrails (orange)
    6: [0.0, 1.0, 1.0],  # Hvac (cyan)
    7: [0.5, 0.0, 0.5],  # Ladders (purple)
    8: [1.0, 0.0, 1.0],  # Pipping (magenta - for pipes!)
    9: [0.5, 0.5, 0.0],  # Supports (olive)
}

# --- Data Loading Function (Adapted for fixed ply_folder_name) ---
def _load_station_data_internal(station_id, ply_folder_path_full, all_labels_global, map_df_global):
    """
    Internal helper to load point cloud and corresponding labels for a given station ID.
    Assumes all_labels_global and map_df_global are already loaded.
    Returns NumPy arrays for points, colors, and labels.
    Raises ValueError if data cannot be loaded or is inconsistent.
    """
    ply_file_name = f'SCAN_{station_id}.ply'
    ply_file_path = os.path.join(ply_folder_path_full, ply_file_name)

    station_info = map_df_global[map_df_global['Station_index'] == station_id]
    if station_info.empty:
        raise ValueError(f"Station ID {station_id} not found in map file.")

    index_start = station_info['index_start'].iloc[0]
    index_end = station_info['index_end'].iloc[0]

    total_labels_count = len(all_labels_global)
    if index_end >= total_labels_count or index_start < 0 or index_end < index_start:
        raise ValueError(f"Label index out of bounds or invalid for Station {station_id}: [{index_start}:{index_end+1}] "
                         f"vs total labels {total_labels_count}.")
    labels_for_station = all_labels_global[index_start : index_end + 1]

    try:
        plydata = PlyData.read(ply_file_path)
        points_element = plydata['points']
        points = np.vstack([points_element['x'], points_element['y'], points_element['z']]).T

        colors = None
        if 'rgb' in plydata:
            rgb_element = plydata['rgb']
            if 'r' in rgb_element.properties and 'g' in rgb_element.properties and 'b' in rgb_element.properties:
                colors = np.vstack([rgb_element['r'], rgb_element['g'], rgb_element['b']]).T
                if colors.max() > 1.0: # Normalize to [0, 1] if 0-255
                    colors = colors / 255.0

        if len(points) == 0:
            raise ValueError(f"PLY file '{ply_file_name}' loaded but contains no points.")

    except Exception as e:
        raise ValueError(f"Failed to load PLY file '{ply_file_name}': {e}")

    if len(points) != len(labels_for_station):
        raise ValueError(f"Mismatch: Points ({len(points)}) != Labels ({len(labels_for_station)}) for Station {station_id}.")

    return points, colors, labels_for_station

# --- Preprocessing Functions (No Changes, but included for completeness) ---
def remove_outliers_np(points, labels, nb_neighbors=20, std_ratio=2.0):
    if len(points) < nb_neighbors:
        return np.array([]), np.array([]) # Return empty arrays if not enough points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    return points[ind], labels[ind]

def normalize_and_center_np(points):
    if len(points) == 0:
        return points

    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_distance = np.max(np.linalg.norm(points_centered, axis=1))
    
    if max_distance > 0:
        points_normalized = points_centered / max_distance
    else:
        points_normalized = points_centered

    return points_normalized

def downsample_voxel_grid_np(points, labels, voxel_size=0.05):
    if len(points) == 0:
        return points, labels

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = downsampled_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    downsampled_points = np.asarray(downsampled_pcd.points)
    
    if len(downsampled_points) == 0:
        return np.array([]), np.array([])

    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    original_pcd_tree = o3d.geometry.KDTreeFlann(original_pcd)
    
    downsampled_labels = np.zeros(len(downsampled_points), dtype=labels.dtype)
    for i, p in enumerate(downsampled_points):
        [k, idx, _] = original_pcd_tree.search_knn_vector_3d(p, 1)
        if k > 0:
            downsampled_labels[i] = labels[idx[0]]
        else:
            downsampled_labels[i] = 0

    return downsampled_points, downsampled_labels

def compute_normals_np(points, radius=0.1, max_nn=30):
    if len(points) == 0:
        return points, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
    
    return np.asarray(pcd.points), np.asarray(pcd.normals)

def augment_data_np(points, labels, rotation_range=(-np.pi/12, np.pi/12), scale_range=(0.8, 1.2), jitter_std=0.01):
    if len(points) == 0:
        return points, labels

    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    points = np.dot(points.reshape(-1, 3), rotation_matrix)

    scale = np.random.uniform(scale_range[0], scale_range[1])
    points = points * scale

    jitter = np.random.normal(0, jitter_std, points.shape)
    points = points + jitter

    return points, labels

# --- Custom Collate Function to Handle None Values ---
def custom_collate_fn(batch):
    """
    Custom collate function that filters out None values and handles empty batches.
    Also properly handles None values within valid batch items (e.g., colors, normals).
    """
    # Filter out None values (completely failed samples)
    valid_batch = [item for item in batch if item is not None]
    
    if len(valid_batch) == 0:
        # Return a dummy batch with empty tensors
        return {
            'points': torch.empty(0, 0, 3),
            'labels': torch.empty(0, 0, dtype=torch.long),
            'colors': None,
            'normals': None,
            'station_id': torch.tensor([-1])
        }
    
    # Extract each field separately with proper handling
    points_list = []
    labels_list = []
    colors_list = []
    normals_list = []
    station_ids = []
    
    for item in valid_batch:
        # Check if item is a valid dictionary
        if not isinstance(item, dict):
            print(f"Warning: Invalid batch item type: {type(item)}")
            continue
            
        # Extract points (required field)
        if 'points' in item and item['points'] is not None:
            points_list.append(item['points'])
        else:
            print(f"Warning: Missing or None points in batch item")
            continue
            
        # Extract labels (required field)
        if 'labels' in item and item['labels'] is not None:
            labels_list.append(item['labels'])
        else:
            print(f"Warning: Missing or None labels in batch item")
            continue
            
        # Extract station_id (required field)
        if 'station_id' in item and item['station_id'] is not None:
            station_ids.append(item['station_id'])
        else:
            print(f"Warning: Missing or None station_id in batch item")
            continue
            
        # Extract colors (optional field)
        if 'colors' in item and item['colors'] is not None:
            colors_list.append(item['colors'])
        else:
            colors_list.append(None)
            
        # Extract normals (optional field)
        if 'normals' in item and item['normals'] is not None:
            normals_list.append(item['normals'])
        else:
            normals_list.append(None)
    
    # If no valid items remain after filtering
    if len(points_list) == 0:
        return {
            'points': torch.empty(0, 0, 3),
            'labels': torch.empty(0, 0, dtype=torch.long),
            'colors': None,
            'normals': None,
            'station_id': torch.tensor([-1])
        }
    
    # Stack the required tensors
    try:
        points_batch = torch.stack(points_list)
        labels_batch = torch.stack(labels_list)
        station_ids_batch = torch.tensor(station_ids)
    except Exception as e:
        print(f"Error stacking required tensors: {e}")
        return {
            'points': torch.empty(0, 0, 3),
            'labels': torch.empty(0, 0, dtype=torch.long),
            'colors': None,
            'normals': None,
            'station_id': torch.tensor([-1])
        }
    
    # Handle colors
    colors_batch = None
    if any(color is not None for color in colors_list):
        try:
            # Fill None values with zero tensors of appropriate shape
            processed_colors = []
            for i, color in enumerate(colors_list):
                if color is not None:
                    processed_colors.append(color)
                else:
                    # Create zero tensor with same shape as corresponding points
                    zero_colors = torch.zeros_like(points_list[i])
                    processed_colors.append(zero_colors)
            colors_batch = torch.stack(processed_colors)
        except Exception as e:
            print(f"Warning: Failed to process colors: {e}")
            colors_batch = None
    
    # Handle normals
    normals_batch = None
    if any(normal is not None for normal in normals_list):
        try:
            # Fill None values with zero tensors of appropriate shape
            processed_normals = []
            for i, normal in enumerate(normals_list):
                if normal is not None:
                    processed_normals.append(normal)
                else:
                    # Create zero tensor with same shape as corresponding points
                    zero_normals = torch.zeros_like(points_list[i])
                    processed_normals.append(zero_normals)
            normals_batch = torch.stack(processed_normals)
        except Exception as e:
            print(f"Warning: Failed to process normals: {e}")
            normals_batch = None
    
    return {
        'points': points_batch,
        'labels': labels_batch,
        'colors': colors_batch,
        'normals': normals_batch,
        'station_id': station_ids_batch
    }

# --- Custom PyTorch Dataset Class ---
class EDFPointCloudDataset(Dataset):
    def __init__(self, dataset_root_path, global_labels_file, master_map_file,
                 station_ids_for_this_split, ply_folder_name,
                 preprocess_params=None, augment=False):
        """
        Initializes the EDF Point Cloud Dataset for a specific split (train/val/test).
        :param dataset_root_path: Root directory of the EDF dataset.
        :param global_labels_file: Name of the CSV file with all global labels (e.g., ytrain_i9bpfD4.csv).
        :param master_map_file: Name of the master map file (e.g., ytrain_map_ind_station.csv).
        :param station_ids_for_this_split: List of station IDs specific to this dataset instance (train/val/test).
        :param ply_folder_name: The name of the subfolder containing the PLY files (e.g., 'train' or 'xtrain_kW4SLO1').
        :param preprocess_params: Dictionary of parameters for preprocessing steps.
        :param augment: Boolean, whether to apply data augmentation (True for training, False for validation/test).
        """
        self.dataset_root_path = dataset_root_path
        self.ply_folder_path_full = os.path.join(dataset_root_path, ply_folder_name)
        self.augment = augment
        self.preprocess_params = preprocess_params if preprocess_params is not None else {}
        
        # --- Load global labels once ---
        print(f"Dataset Init: Loading global labels from: {os.path.join(dataset_root_path, global_labels_file)}")
        try:
            self.all_labels_global = pd.read_csv(os.path.join(dataset_root_path, global_labels_file), dtype={'class': np.int64})['class'].values
            print(f"Dataset Init: Loaded {len(self.all_labels_global)} global labels.")
        except Exception as e:
            print(f"Dataset Init ERROR: Could not load global labels: {e}")
            sys.exit(1)

        # --- Load master map file once ---
        print(f"Dataset Init: Loading master map file from: {os.path.join(dataset_root_path, master_map_file)}")
        try:
            self.master_map_df = pd.read_csv(os.path.join(dataset_root_path, master_map_file), header=None,
                                             names=['Station_index', 'index_start', 'index_end'],
                                             dtype={'Station_index': np.int64, 'index_start': np.int64, 'index_end': np.int64})
            print(f"Dataset Init: Master map loaded with {len(self.master_map_df)} entries.")
        except Exception as e:
            print(f"Dataset Init ERROR: Could not load master map file: {e}")
            sys.exit(1)

        # --- Filter and Validate Station IDs for this specific split ---
        self.station_ids = []
        self.failed_stations = []  # Track failed stations for debugging
        print(f"Dataset Init: Validating {len(station_ids_for_this_split)} stations for this split...")
        for station_id in station_ids_for_this_split:
            try:
                # Attempt to load data for validation (without full preprocessing)
                # This will call _load_station_data_internal and catch its ValueErrors
                points, _, _ = _load_station_data_internal(
                    station_id, self.ply_folder_path_full,
                    self.all_labels_global, self.master_map_df
                )
                if points is not None and len(points) > 0:
                    self.station_ids.append(station_id)
                else:
                    print(f"Dataset Init WARNING: Station {station_id} is empty or failed basic load. Skipping.")
                    self.failed_stations.append(station_id)
            except (ValueError, FileNotFoundError) as e:
                print(f"Dataset Init WARNING: Skipping Station {station_id} due to loading error: {e}")
                self.failed_stations.append(station_id)
            except Exception as e:
                print(f"Dataset Init WARNING: Unexpected error validating Station {station_id}: {e}. Skipping.")
                self.failed_stations.append(station_id)

        if not self.station_ids:
            print(f"Dataset Init ERROR: No valid stations found for this split. Check data paths and integrity.")
            print(f"Failed stations: {self.failed_stations}")
            # sys.exit(1) # Consider exiting here if no valid data at all, or let it pass if it's a small split
        
        print(f"Dataset Init: This instance will use {len(self.station_ids)} valid stations from '{ply_folder_name}' folder.")
        if self.failed_stations:
            print(f"Dataset Init: Failed to load {len(self.failed_stations)} stations: {self.failed_stations}")

    def __len__(self):
        """Returns the number of valid stations in this dataset split."""
        return len(self.station_ids)

    def _create_dummy_sample(self, station_id):
        """
        Creates a minimal valid sample when all else fails.
        """
        print(f"Creating dummy sample for station {station_id}")
        # Create a minimal point cloud with 100 points
        points = np.random.randn(100, 3).astype(np.float32)
        labels = np.zeros(100, dtype=np.int64)  # All background class
        
        return {
            'points': torch.from_numpy(points).float(),
            'labels': torch.from_numpy(labels).long(),
            'colors': None,
            'normals': None,
            'station_id': station_id
        }

    def __getitem__(self, idx):
        """
        Debug version of __getitem__ with extensive logging
        """
        print(f"DEBUG: __getitem__ called with idx={idx}")
        
        if idx >= len(self.station_ids):
            print(f"ERROR: Index {idx} out of range for {len(self.station_ids)} valid stations")
            return None
            
        station_id = self.station_ids[idx]
        print(f"DEBUG: Processing station_id={station_id}")
        
        try:
            # 1. Load raw data
            print(f"DEBUG: Loading raw data for station {station_id}")
            points, colors, labels = _load_station_data_internal(
                station_id, self.ply_folder_path_full,
                self.all_labels_global, self.master_map_df
            )
            
            if points is None or len(points) == 0:
                print(f"ERROR: Station {station_id} has no points after loading")
                return None

            print(f"DEBUG: Loaded {len(points)} points, colors: {colors is not None}, labels: {len(labels)}")

            # 2. Apply Preprocessing Steps
            original_point_count = len(points)
            
            # Outlier Removal
            if self.preprocess_params.get('remove_outliers', False):
                print(f"DEBUG: Removing outliers...")
                points, labels = remove_outliers_np(
                    points, labels,
                    nb_neighbors=self.preprocess_params.get('outlier_nb_neighbors', 20),
                    std_ratio=self.preprocess_params.get('outlier_std_ratio', 2.0)
                )
                if len(points) == 0:
                    print(f"ERROR: Station {station_id} became empty after outlier removal")
                    return None
                print(f"DEBUG: After outlier removal: {len(points)} points")

            # Normalization and Centering
            if self.preprocess_params.get('normalize_center', True):
                print(f"DEBUG: Normalizing and centering...")
                points = normalize_and_center_np(points)
                print(f"DEBUG: After normalization: {points.shape}")

            # Downsampling
            if self.preprocess_params.get('downsample', False):
                print(f"DEBUG: Downsampling...")
                pre_downsample_count = len(points)
                points, labels = downsample_voxel_grid_np(
                    points, labels,
                    voxel_size=self.preprocess_params.get('voxel_size', 0.05)
                )
                if len(points) == 0:
                    print(f"ERROR: Station {station_id} became empty after downsampling")
                    return None
                print(f"DEBUG: After downsampling: {pre_downsample_count} -> {len(points)} points")

            # Compute Normals
            normals = None
            if self.preprocess_params.get('compute_normals', False):
                print(f"DEBUG: Computing normals...")
                try:
                    points, normals = compute_normals_np(
                        points,
                        radius=self.preprocess_params.get('normal_radius', 0.1),
                        max_nn=self.preprocess_params.get('normal_max_nn', 30)
                    )
                    print(f"DEBUG: Computed normals: {normals is not None and len(normals)}")
                except Exception as e:
                    print(f"WARNING: Failed to compute normals for station {station_id}: {e}")
                    normals = None

            # 3. Apply Data Augmentation
            if self.augment:
                print(f"DEBUG: Applying augmentation...")
                points, labels = augment_data_np(
                    points, labels,
                    rotation_range=self.preprocess_params.get('augment_rotation_range', (-np.pi/12, np.pi/12)),
                    scale_range=self.preprocess_params.get('augment_scale_range', (0.9, 1.1)),
                    jitter_std=self.preprocess_params.get('augment_jitter_std', 0.005)
                )
            
            # Final check for empty data
            if len(points) == 0:
                print(f"ERROR: Station {station_id} is empty after all preprocessing")
                return None
            
            print(f"DEBUG: Final data shapes - Points: {points.shape}, Labels: {labels.shape}")
            print(f"DEBUG: Colors available: {colors is not None}, Normals available: {normals is not None}")
            
            # --- Convert to PyTorch Tensors ---
            try:
                points_tensor = torch.from_numpy(points.astype(np.float32))
                labels_tensor = torch.from_numpy(labels.astype(np.int64))
                print(f"DEBUG: Converted points and labels to tensors")
            except Exception as e:
                print(f"ERROR: Failed to convert points/labels to tensors: {e}")
                return None
            
            # Handle colors
            colors_tensor = None
            if colors is not None and len(colors) > 0:
                try:
                    colors_tensor = torch.from_numpy(colors.astype(np.float32))
                    print(f"DEBUG: Converted colors to tensor: {colors_tensor.shape}")
                except Exception as e:
                    print(f"WARNING: Failed to convert colors to tensor: {e}")
                    colors_tensor = None
            
            # Handle normals
            normals_tensor = None
            if normals is not None and len(normals) > 0:
                try:
                    normals_tensor = torch.from_numpy(normals.astype(np.float32))
                    print(f"DEBUG: Converted normals to tensor: {normals_tensor.shape}")
                except Exception as e:
                    print(f"WARNING: Failed to convert normals to tensor: {e}")
                    normals_tensor = None

            result = {
                'points': points_tensor,
                'labels': labels_tensor,
                'colors': colors_tensor,
                'normals': normals_tensor,
                'station_id': station_id
            }
            
            print(f"DEBUG: Returning result for station {station_id}")
            print(f"DEBUG: Result keys: {list(result.keys())}")
            for key, value in result.items():
                if value is None:
                    print(f"DEBUG: {key} is None")
                elif hasattr(value, 'shape'):
                    print(f"DEBUG: {key} shape: {value.shape}")
                else:
                    print(f"DEBUG: {key} value: {value}")
            
            return result
            
        except Exception as e:
            print(f"ERROR: Failed to process station {station_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- Main Script for Data Splitting and DataLoader Creation ---
if __name__ == "__main__":
    # --- Step 1: Load Master Map and Filter Stations (0-30) ---
    print("\n--- Preparing Dataset Splits (Stations 0-30) ---")
    master_map_path = os.path.join(dataset_root_path, master_map_file)
    
    try:
        full_map_df = pd.read_csv(master_map_path, header=None,
                                  names=['Station_index', 'index_start', 'index_end'],
                                  dtype={'Station_index': np.int64, 'index_start': np.int64, 'index_end': np.int64})
    except Exception as e:
        print(f"ERROR: Could not load master map file '{master_map_file}': {e}")
        sys.exit(1)

    # Filter for stations 0 to 30 (inclusive)
    filtered_stations_df = full_map_df[
        (full_map_df['Station_index'] >= 0) & 
        (full_map_df['Station_index'] <= 30)
    ]
    all_station_ids_raw = filtered_stations_df['Station_index'].tolist()

    if not all_station_ids_raw:
        print("ERROR: No stations found in the range 0-30 in the master map file. Please check your data.")
        sys.exit(1)

    print(f"Found {len(all_station_ids_raw)} stations in the range 0-30 before validation.")

    # --- Step 2: Define Preprocessing Parameters for initial validation ---
    init_validation_preprocess_params = {
        'remove_outliers': False,
        'normalize_center': False,
        'downsample': False,
        'compute_normals': False,
        'augment': False
    }

    # Temporarily create a dummy dataset instance to get the list of truly valid station IDs
    print("\n--- Performing initial validation of all stations (0-30) to filter invalid ones ---")
    temp_dataset = EDFPointCloudDataset(
        dataset_root_path=dataset_root_path,
        global_labels_file=global_labels_file,
        master_map_file=master_map_file,
        station_ids_for_this_split=all_station_ids_raw,
        ply_folder_name=ply_folder_name_fixed,
        preprocess_params=init_validation_preprocess_params,
        augment=False
    )
    all_valid_station_ids = temp_dataset.station_ids
    del temp_dataset

    if not all_valid_station_ids:
        print("ERROR: After initial validation, no valid stations remain. Cannot proceed with splitting.")
        sys.exit(1)
    
    print(f"Found {len(all_valid_station_ids)} valid stations after initial loadability check.")

    # --- Step 3: Split Valid Station IDs into Train, Validation, and Test Sets ---
    train_ids, val_test_ids = train_test_split(
        all_valid_station_ids, test_size=0.2, random_state=42
    )
    val_ids, test_ids = train_test_split(
        val_test_ids, test_size=0.5, random_state=42
    )

    print(f"\nTrain stations: {len(train_ids)} IDs: {sorted(train_ids)}")
    print(f"Validation stations: {len(val_ids)} IDs: {sorted(val_ids)}")
    print(f"Test stations: {len(test_ids)} IDs: {sorted(test_ids)}")

    # --- Step 4: Define Preprocessing Parameters for actual training/val/test ---
    train_preprocess_params = {
        'remove_outliers': True,
        'outlier_nb_neighbors': 20,
        'outlier_std_ratio': 2.0,
        'normalize_center': True,
        'downsample': True,
        'voxel_size': 0.05,
        'compute_normals': True,
        'normal_radius': 0.1,
        'normal_max_nn': 30,
        'augment_rotation_range': (-np.pi/12, np.pi/12),
        'augment_scale_range': (0.9, 1.1),
        'augment_jitter_std': 0.005
    }

    val_test_preprocess_params = {
        'remove_outliers': True,
        'outlier_nb_neighbors': 20,
        'outlier_std_ratio': 2.0,
        'normalize_center': True,
        'downsample': True,
        'voxel_size': 0.05,
        'compute_normals': True,
        'normal_radius': 0.1,
        'normal_max_nn': 30,
    }

    # --- Step 5: Create Dataset and DataLoader Instances for Each Split ---
    print("\n--- Initializing Training Dataset and DataLoader ---")
    train_dataset = EDFPointCloudDataset(
        dataset_root_path=dataset_root_path,
        global_labels_file=global_labels_file,
        master_map_file=master_map_file,
        station_ids_for_this_split=train_ids,
        ply_folder_name=ply_folder_name_fixed,
        preprocess_params=train_preprocess_params,
        augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # Fixed pin_memory warning
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    print(f"Training DataLoader created with {len(train_loader)} batches.")

    print("\n--- Initializing Validation Dataset and DataLoader ---")
    val_dataset = EDFPointCloudDataset(
        dataset_root_path=dataset_root_path,
        global_labels_file=global_labels_file,
        master_map_file=master_map_file,
        station_ids_for_this_split=val_ids,
        ply_folder_name=ply_folder_name_fixed,
        preprocess_params=val_test_preprocess_params,
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,  # Fixed pin_memory warning
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    print(f"Validation DataLoader created with {len(val_loader)} batches.")

    print("\n--- Initializing Test Dataset and DataLoader ---")
    test_dataset = EDFPointCloudDataset(
        dataset_root_path=dataset_root_path,
        global_labels_file=global_labels_file,
        master_map_file=master_map_file,
        station_ids_for_this_split=test_ids,
        ply_folder_name=ply_folder_name_fixed,
        preprocess_params=val_test_preprocess_params,
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,  # Fixed pin_memory warning
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    print(f"Test DataLoader created with {len(test_loader)} batches.")

    # --- Test Data Loading and Preprocessing from DataLoaders ---
    print("\n--- Testing Data Loading from Training DataLoader (first few batches) ---")
    print(train_loader)
    print(test_loader)
    print(val_loader)
    successful_batches = 0
    failed_batches = 0
    
    for i, batch in enumerate(train_loader):
        if batch is None or batch['station_id'].item() == -1:
            print(f"Batch {i+1}: Failed/Empty batch")
            failed_batches += 1
            continue
            
        points = batch['points']
        labels = batch['labels']
        colors = batch['colors']
        normals = batch['normals']
        station_id = batch['station_id'].item()

        print(f"\nBatch {i+1} (Train Set):")
        print(f"  Station ID: {station_id}")
        print(f"  Points shape: {points.shape}")
        print(f"  Labels shape: {labels.shape}")
        if colors is not None and colors[0] is not None: 
            print(f"  Colors shape: {colors.shape}")
        if normals is not None and normals[0] is not None: 
            print(f"  Normals shape: {normals.shape}")
        
        successful_batches += 1

        if successful_batches == 1:
            print("\nVisualizing the first successfully processed point cloud from Training DataLoader. Close window to continue.")
            pcd_display = o3d.geometry.PointCloud()
            pcd_display.points = o3d.utility.Vector3dVector(points[0].numpy())
            
            display_labels = labels[0].numpy()
            display_colors = np.zeros((len(display_labels), 3))
            for j, label_id in enumerate(display_labels):
                display_colors[j] = CLASS_COLORS.get(int(label_id), [0.8, 0.8, 0.8])
            pcd_display.colors = o3d.utility.Vector3dVector(display_colors)

            if normals is not None and normals[0] is not None:
                pcd_display.normals = o3d.utility.Vector3dVector(normals[0].numpy())
                o3d.visualization.draw_geometries([pcd_display], point_show_normal=True)
            else:
                o3d.visualization.draw_geometries([pcd_display])
        
        # Test first 5 successful batches
        if successful_batches >= 5:
            break

    print(f"\n--- Data Loading Test Complete ---")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed/Empty batches: {failed_batches}")
    print("DataLoaders are ready for model training.")