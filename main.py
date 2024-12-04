import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import yaml
import os

# Load parameters from YAML
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Step 1: Preprocess PCD to generate BEV
def preprocess_pcd_to_bev(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds):
    # Load the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    
    # Flip horizontally
    points[:, 0] = -points[:, 0]
    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Ground removal using RANSAC
    plane_model, inliers = flipped_pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=5000)
    non_ground = flipped_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)
    
    # Filter points within ROI
    x_min, x_max, y_min, y_max, z_min, z_max = roi_bounds
    roi_points = non_ground_points[
        (non_ground_points[:, 0] >= x_min) & (non_ground_points[:, 0] <= x_max) &
        (non_ground_points[:, 1] >= y_min) & (non_ground_points[:, 1] <= y_max) &
        (non_ground_points[:, 2] >= z_min) & (non_ground_points[:, 2] <= z_max)
    ]
    
    # Generate BEV grid
    w, h = grid_resolution
    x_bins = np.arange(x_range[0], x_range[1], w)
    y_bins = np.arange(y_range[0], y_range[1], h)
    bev_grid = np.zeros((len(x_bins), len(y_bins)))

    for x, y, z in roi_points:
        x_idx = int((x - x_range[0]) / w)
        y_idx = int((y - y_range[0]) / h)

        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            bev_grid[x_idx, y_idx] = max(bev_grid[x_idx, y_idx], z / z_max)  # Normalize height

    return bev_grid

# Step 2: Compute optical flow between two BEV grids
def compute_velocity_vectors(bev1, bev2, x_range, y_range):
    farneback_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow = cv2.calcOpticalFlowFarneback(bev1.astype(np.float32), bev2.astype(np.float32), None, **farneback_params)
    vx, vy = flow[..., 0], flow[..., 1]

    # Convert flow to real-world velocities
    pixel_size_x = (x_range[1] - x_range[0]) / bev1.shape[1]
    pixel_size_y = (y_range[1] - y_range[0]) / bev1.shape[0]
    velocity_x = vx * pixel_size_x
    velocity_y = vy * pixel_size_y

    return velocity_x, velocity_y

# Step 3: Apply propagation mask
def propagation_mask(vx, vy, dt, grid_shape, grid_size):
    rows, cols = grid_shape
    cell_height, cell_width = grid_size
    vx_prop = np.zeros_like(vx)
    vy_prop = np.zeros_like(vy)

    for i in range(rows):
        for j in range(cols):
            x_new = i + int(vx[i, j] * dt / cell_height + 0.5)
            y_new = j + int(vy[i, j] * dt / cell_width + 0.5)

            if 0 <= x_new < rows and 0 <= y_new < cols:
                vx_prop[x_new, y_new] = vx[i, j]
                vy_prop[x_new, y_new] = vy[i, j]

    return vx_prop, vy_prop


# Step 4: Apply rigid-body continuity mask
def rigid_body_continuity_mask(vx, vy, alpha_cont):
    div_v = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    curl_v = np.gradient(vy, axis=1) - np.gradient(vx, axis=0)
    mask = (np.abs(div_v) <= alpha_cont) & (np.abs(curl_v) <= alpha_cont)
    return mask.astype(int)

# Step 5: Combine everything into a pipeline
def process_and_compare_pcds(pcd_file1, pcd_file2, config):
    # Extract parameters
    grid_resolutions = config['grid_resolution']
    x_range = config['x_range']
    y_range = config['y_range']
    z_max = config['z_max']
    roi_bounds = config['roi_bounds']
    dt = config['dt']
    alpha_p_list = config['masks']['alpha_p']
    alpha_cont_list = config['masks']['alpha_cont']

    # Iterate over all combinations of grid resolutions, alpha_p, and alpha_cont
    for grid_resolution in grid_resolutions:
        for alpha_p in alpha_p_list:
            for alpha_cont in alpha_cont_list:
                print(f"Processing with grid_resolution={grid_resolution}, alpha_p={alpha_p}, alpha_cont={alpha_cont}")

                # Call your preprocessing and processing pipeline
                bev1 = preprocess_pcd_to_bev(pcd_file1, grid_resolution, x_range, y_range, z_max, roi_bounds)
                bev2 = preprocess_pcd_to_bev(pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds)
                velocity_x, velocity_y = compute_velocity_vectors(bev1, bev2, x_range, y_range)

                # Calculate grid shape
                grid_shape = (
                    int((x_range[1] - x_range[0]) / grid_resolution[0]),
                    int((y_range[1] - y_range[0]) / grid_resolution[1])
                )
                grid_size = (grid_resolution[0], grid_resolution[1])

                vx_prop, vy_prop = propagation_mask(velocity_x, velocity_y, dt, grid_shape, grid_size)
                mask = rigid_body_continuity_mask(velocity_x, velocity_y, alpha_cont)

                # Visualization or further processing as needed
                plt.figure(figsize=(10, 10))
                plt.quiver(
                    np.linspace(x_range[0], x_range[1], bev1.shape[1]),
                    np.linspace(y_range[0], y_range[1], bev1.shape[0]),
                    vx_prop * mask, vy_prop * mask,
                    angles='xy', scale_units='xy', scale=1, color='blue'
                )
                plt.title(f"Velocity Vectors with Masks (grid_resolution={grid_resolution}, alpha_p={alpha_p}, alpha_cont={alpha_cont})")
                plt.xlabel("X (meters)")
                plt.ylabel("Y (meters)")
                plt.grid(color='black')
                plt.show()


# Main script
if __name__ == "__main__":
    # Load YAML configuration
    yaml_file = r"config/config.yaml"  
    config = load_config(yaml_file)

    # File paths for PCD files
    pcd_file1 = config['pcd_file1']
    pcd_file2 = config['pcd_file2']

    # Run the pipeline
    process_and_compare_pcds(pcd_file1, pcd_file2, config)
