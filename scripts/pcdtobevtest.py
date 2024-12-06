import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

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
def propagation_mask(vx, vy, dt, grid_resolution, grid_size):
    rows, cols = grid_resolution
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
def process_and_compare_pcds(pcd_file1, pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds, dt, alpha_p, alpha_cont):
    # Generate BEV for both PCD files
    bev1 = preprocess_pcd_to_bev(pcd_file1, grid_resolution, x_range, y_range, z_max, roi_bounds)
    bev2 = preprocess_pcd_to_bev(pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds)

    # Compute velocity vectors
    velocity_x, velocity_y = compute_velocity_vectors(bev1, bev2, x_range, y_range)

    # Apply propagation mask
    grid_size = (grid_resolution[0], grid_resolution[1])
    grid_shape = (len(np.arange(x_range[0], x_range[1], grid_resolution[0])),
                  len(np.arange(y_range[0], y_range[1], grid_resolution[1])))
    vx_prop, vy_prop = propagation_mask(velocity_x, velocity_y, dt, grid_shape, grid_size)

    # Apply rigid-body continuity mask
    mask = rigid_body_continuity_mask(velocity_x, velocity_y, alpha_cont)

    # Visualization
    X, Y = np.meshgrid(
        np.arange(x_range[0], x_range[1], grid_resolution[0]),
        np.arange(y_range[0], y_range[1], grid_resolution[1])
    )
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, vx_prop * mask, vy_prop * mask, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.title("Velocity Vectors with Masks on BEV Grid")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(color='black')
    plt.show()

# Parameters
grid_resolution = (1.0, 1.0)
x_range = (-50, 50)
y_range = (-50, 50)
z_max = 10.0
roi_bounds = (-20, 20, -20, 22.09, -2.4, 1)
dt = 1.0
alpha_p = 0.5
alpha_cont = 0.5

# File paths for PCD files
pcd_file1 = r"C:/Users/anvit/Desktop/Desktop/Fall_24/MR/Project/DATMO/GT2/test_2_GT_PCD/test_2_GT_PCD/lidar_frame_0030.pcd"
pcd_file2 = r"C:/Users/anvit/Desktop/Desktop/Fall_24/MR/Project/DATMO/GT2/test_2_GT_PCD/test_2_GT_PCD/lidar_frame_0090.pcd"

# Run the pipeline
process_and_compare_pcds(pcd_file1, pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds, dt, alpha_p, alpha_cont)
