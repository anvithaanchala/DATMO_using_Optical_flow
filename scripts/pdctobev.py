import numpy as np
import open3d as o3d
import os
import re
import matplotlib.pyplot as plt

# Step 1: Filter points within the ROI box
def filter_points_in_roi(points, roi_bounds):
    x_min, x_max, y_min, y_max, z_min, z_max = roi_bounds
    return points[
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &  # X bounds
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &  # Y bounds
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)    # Z bounds
    ]

# Step 2: Generate elevation map
def generate_elevation_map(points, cell_size, x_range, y_range):
    x_bins = np.arange(x_range[0], x_range[1], cell_size)
    y_bins = np.arange(y_range[0], y_range[1], cell_size)
    elevation_map = np.full((len(x_bins), len(y_bins)), np.nan)

    for x, y, z in points:
        
        x_idx = int((x - x_range[0]) / cell_size)
        y_idx = int((y - y_range[0]) / cell_size)

        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            if np.isnan(elevation_map[x_idx, y_idx]) or z > elevation_map[x_idx, y_idx]:
                elevation_map[x_idx, y_idx] = z

    return elevation_map

# Step 3: Compute BEV Grid
def compute_bev_grid(points, grid_resolution, x_range, y_range, a=0.5, b=0.5, h_max=5.0):
    w, h = grid_resolution
    x_bins = np.arange(x_range[0], x_range[1], w)
    y_bins = np.arange(y_range[0], y_range[1], h)

    bev_grid = [[[] for _ in range(len(y_bins))] for _ in range(len(x_bins))]
    for x, y, z in points:
        x_idx = int((x - x_range[0]) / w)
        y_idx = int((y - y_range[0]) / h)

        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            bev_grid[x_idx][y_idx].append(z)

    bev_values = np.zeros((len(x_bins), len(y_bins)))
    for i in range(len(x_bins)):
        for j in range(len(y_bins)):
            heights = np.array(bev_grid[i][j])
            if len(heights) > 0:
                mean_height = np.mean(heights)
                std_height = np.std(heights)
                bev_values[i, j] = (a * mean_height + b * std_height) / h_max
            else:
                bev_values[i, j] = 0

    return bev_values

# Step 4: Process a single PCD file
def process_pcd_to_2_5d(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds, output_folder):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    points[:, 0] = -points[:, 0]  # Flip along the X-axis
    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = flipped_pcd.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=5000)
    non_ground = flipped_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)

    roi_points = filter_points_in_roi(non_ground_points, roi_bounds)
    if roi_points.size == 0:
        print(f"No ROI points for {pcd_file}. Adjust ROI bounds.")
        return

    bev_grid = compute_bev_grid(roi_points, grid_resolution, x_range, y_range, h_max=z_max)


    bev_image_path = os.path.join(output_folder, os.path.basename(pcd_file).replace('.pcd', '_bev.png'))
    plt.figure(figsize=(8, 8))
    plt.imshow(bev_grid, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='G_ij Value')
    plt.title('Bird’s Eye View (BEV)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.savefig(bev_image_path)
    plt.close()
    print(f"Saved BEV Grayscale Image: {bev_image_path}")

# Step 5: Process all PCD files in a folder
def process_folder_to_2_5d(input_folder, output_folder, grid_resolution, x_range, y_range, z_max, roi_bounds):
    os.makedirs(output_folder, exist_ok=True)
    pcd_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.pcd')]

    filtered_pcd_files = [
        f for f in pcd_files if int(re.search(r'(\d+)', os.path.basename(f)).group(1)) >= 300
    ]

    for pcd_file in sorted(filtered_pcd_files):
        print(f"Processing: {pcd_file}")
        try:
            process_pcd_to_2_5d(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds, output_folder)
        except Exception as e:
            print(f"Error processing {pcd_file}: {e}")

# Parameters
input_folder = r'C:\Users\anvit\OneDrive\Desktop\Fall_24\MR\Project\DATMO\GT2\test_2_GT_PCD\test_2_GT_PCD'
output_folder = r'C:\Users\anvit\OneDrive\Desktop\Fall_24\MR\Project\DATMO\GT2\bev_gt2'
grid_resolution = (1.0, 1.0)
x_range = (-50, 50)
y_range = (-50, 50)
z_max = 10.0
roi_bounds = (-20, 20, -20, 22.09, -2.4, 1)

# Run the pipeline
process_folder_to_2_5d(input_folder, output_folder, grid_resolution, x_range, y_range, z_max, roi_bounds)