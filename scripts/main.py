import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import yaml

# Load parameters from YAML
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def filter_points_in_roi(points, roi_bounds):
    x_min, x_max, y_min, y_max, z_min, z_max = roi_bounds
    return points[
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    ]

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

def preprocess_pcd(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds):
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    points = np.asarray(pcd.points)

    # Flip the point cloud horizontally
    points[:, 0] = -points[:, 0]
    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample and remove statistical outliers
    downpcd = flipped_pcd.voxel_down_sample(voxel_size=0.05)
    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    clean_pcd = downpcd.select_by_index(ind)
    o3d.visualization.draw_geometries([clean_pcd])

    # Ground removal using RANSAC
    plane_model, inliers = clean_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000)
    non_ground = clean_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)

    # Filter points within ROI
    roi_points = filter_points_in_roi(non_ground_points, roi_bounds)
    if roi_points.size == 0:
        print(f"No ROI points for {pcd_file}. Adjust ROI bounds.")
        return None

    # Generate BEV grid
    bev_grid = compute_bev_grid(roi_points, grid_resolution, x_range, y_range, h_max=z_max)

    # Visualize BEV
    plt.figure(figsize=(8, 8))
    plt.imshow(bev_grid, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='G_ij Value')
    plt.title('Bird’s Eye View (BEV)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()

    return bev_grid

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
    frame1 = cv2.imread(bev1, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(bev2, cv2.IMREAD_GRAYSCALE)

    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, **farneback_params)
    vx, vy = flow[..., 0], flow[..., 1]

    # Convert flow to real-world velocities
    pixel_size_x = (x_range[1] - x_range[0]) / bev1.shape[1]
    pixel_size_y = (y_range[1] - y_range[0]) / bev1.shape[0]
    velocity_x = vx * pixel_size_x
    velocity_y = vy * pixel_size_y

    return velocity_x, velocity_y


def apply_propagation_mask(vx, vy, dt, grid_shape, grid_size):
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

def apply_continuity_mask(vx, vy, alpha_cont):
    div_v = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    curl_v = np.gradient(vy, axis=1) - np.gradient(vx, axis=0)
    mask = (np.abs(div_v) <= alpha_cont) & (np.abs(curl_v) <= alpha_cont)
    return mask.astype(int)

def process_and_compare_pcds(pcd_file1, pcd_file2, config):
    grid_resolution = config['grid_resolution']
    x_range = config['x_range']
    y_range = config['y_range']
    z_max = config['z_max']   
    dt = config['dt']
    alpha_cont = config['masks']['alpha_cont'][0]

    bev1 = preprocess_pcd(pcd_file1, grid_resolution, x_range, y_range, z_max, roi_bounds)
    bev2 = preprocess_pcd(pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds)

    if bev1 is not None and bev2 is not None:
        velocity_x, velocity_y = compute_velocity_vectors(bev1, bev2, x_range, y_range)

        grid_shape = bev1.shape
        grid_size = grid_resolution
        vx_prop, vy_prop = apply_propagation_mask(velocity_x, velocity_y, dt, grid_shape, grid_size)
        continuity_mask = apply_continuity_mask(velocity_x, velocity_y, alpha_cont)

        plt.figure(figsize=(10, 10))
        plt.quiver(
            np.linspace(x_range[0], x_range[1], grid_shape[1]),
            np.linspace(y_range[0], y_range[1], grid_shape[0]),
            vx_prop * continuity_mask,
            vy_prop * continuity_mask,
            angles='xy', scale_units='xy', scale=1, color='red'
        )
        plt.title("Velocity Vectors with Continuity Mask")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(color='black')
        plt.show()

if __name__ == "__main__":
    yaml_file = "config/config.yaml"
    config = load_config(yaml_file)

    pcd_file1 = config['pcd_file1']
    pcd_file2 = config['pcd_file2']

    process_and_compare_pcds(pcd_file1, pcd_file2, config)

