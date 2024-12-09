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


# Functions from pdctobev.py
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
    print(h_max)

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
    
    bev_values = bev_values/bev_values.max()
    bev_values = (bev_values * h_max).astype(np.uint8)


    return bev_values


def preprocess_pcd(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds):
    pcd = o3d.io.read_point_cloud(pcd_file)

    points = np.asarray(pcd.points)

    # Flip the point cloud horizontally
    points[:, 0] = -points[:, 0]
    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)

    # # Downsample and remove statistical outliers
    # downpcd = flipped_pcd.voxel_down_sample(voxel_size=0.05)
    # cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # clean_pcd = downpcd.select_by_index(ind)

    # Ground removal using RANSAC
    plane_model, inliers = flipped_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000)
    non_ground = flipped_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)

    # Filter points within ROI
    roi_points = filter_points_in_roi(non_ground_points, roi_bounds)
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    o3d.visualization.draw_geometries([roi_pcd])
    if roi_points.size == 0:
        print(f"No ROI points for {pcd_file}. Adjust ROI bounds.")
        return None

    # Generate BEV grid
    bev_grid = compute_bev_grid(roi_points, grid_resolution, x_range, y_range, h_max=z_max)

    return bev_grid


def compute_velocity_vectors(bev1, bev2, x_range, y_range,dt):
    farneback_params = dict(
        pyr_scale=0.3,
        levels=5,
        winsize=15,
        iterations=5,
        poly_n=5,
        poly_sigma=5,
        flags=0,
    )
    flow = cv2.calcOpticalFlowFarneback(bev1.astype(np.float32), bev2.astype(np.float32), None, **farneback_params)
    vx, vy = flow[..., 0], flow[..., 1]

    # Convert flow to real-world velocities
    pixel_size_x = (x_range[1] - x_range[0]) / bev1.shape[1]
    pixel_size_y = (y_range[1] - y_range[0]) / bev1.shape[0]
    velocity_x = vx * pixel_size_x
    velocity_y = vy * pixel_size_y
    # Compute derivatives
    dvx_dy, dvx_dx = np.gradient(velocity_x)
    dvy_dy, dvy_dx = np.gradient(velocity_y)

# Compute angular velocity (z-axis rotation)
    angular_velocity = dvy_dx - dvx_dy  # Curl formula
    print(velocity_x)
    print(velocity_y)# Visualize Linear Velocities
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(velocity_x, cmap='coolwarm', origin='lower')
    plt.colorbar(label="v_x (m/s)")
    plt.title("Linear Velocity (X)")

    plt.subplot(1, 2, 2)
    plt.imshow(velocity_y, cmap='coolwarm', origin='lower')
    plt.colorbar(label="v_y (m/s)")
    plt.title("Linear Velocity (Y)")
    plt.show()

    # Visualize Angular Velocity
    plt.figure(figsize=(6, 6))
    plt.imshow(angular_velocity, cmap='coolwarm', origin='lower')
    plt.colorbar(label="Angular Velocity ω_z (rad/s)")
    plt.title("Angular Velocity (Z)")
    plt.show()
    X, Y = np.meshgrid(
    np.linspace(x_range[0], x_range[1], velocity_x.shape[1]),
    np.linspace(y_range[0], y_range[1], velocity_x.shape[0])
)
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, velocity_x, velocity_y, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.title("Velocity Vector Grid Before Continuity Mask")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(color='black')  # Add gridlines for clarity
    plt.show()

        
    return velocity_x, velocity_y,angular_velocity

def propagation_mask(vx, vy, dt, grid_resolution, alpha_p):
    h, w = vx.shape
    propagated_vx = np.zeros_like(vx)
    propagated_vy = np.zeros_like(vy)
    
    # Iterate over grid cells
    for i in range(h):
        for j in range(w):
            i_prime = int(i + np.floor(vx[i, j] * dt / grid_resolution[0]))
            j_prime = int(j + np.floor(vy[i, j] * dt / grid_resolution[1]))
            if 0 <= i_prime < h and 0 <= j_prime < w:
                propagated_vx[i_prime, j_prime] = vx[i, j]
                propagated_vy[i_prime, j_prime] = vy[i, j]

    # Compare propagated velocities with actual velocities to generate mask
    mask = (np.abs(propagated_vx - vx) <= alpha_p) & (np.abs(propagated_vy - vy) <= alpha_p)
    return mask.astype(int)

def continuity_mask(vx, vy, alpha_cont):
    div_v = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    curl_v = np.gradient(vy, axis=1) - np.gradient(vx, axis=0)
    mask = (np.abs(div_v) <= alpha_cont) & (np.abs(curl_v) <= alpha_cont)
    return mask.astype(int)

#main pipeline
def process_and_compare_pcds(pcd_file1, pcd_file2, pcd_file3, config):
    grid_resolution = config['grid_resolution']
    x_range = config['x_range']
    y_range = config['y_range']
    z_max = config['z_max']
    roi_bounds = config['roi_bounds']
    alpha_p = config['masks']['alpha_p'][0]
    alpha_cont =config['masks']['alpha_cont'][0]
    dt = config['dt']

    # Preprocess PCDs and compute BEVs
    bev1 = preprocess_pcd(pcd_file1, grid_resolution, x_range, y_range, z_max, roi_bounds)
    bev2 = preprocess_pcd(pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds)
    bev3 = preprocess_pcd(pcd_file3, grid_resolution, x_range, y_range, z_max, roi_bounds)
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(bev1, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='G_ij Value')
    plt.title('Bird’s Eye View (BEV)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.subplot(2, 2, 2)
    plt.imshow(bev2, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='G_ij Value')
    plt.title('Bird’s Eye View (BEV)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()
    plt.subplot(2, 2, 3)
    plt.imshow(bev3, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='G_ij Value')
    plt.title('Bird’s Eye View (BEV)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()

    #checking difference in bevs as velocity vectors are not being generated 
    diff1 = np.abs(bev1 - bev2)
    diff2 = np.abs(bev2 - bev3)
    print("Difference between BEV grids 1 and 2 :", np.sum(diff1))
    print("Difference between BEV grids:", np.sum(diff2))
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(diff1, cmap='gray')
    plt.title("Difference between BEV1 and BEV2")
    plt.subplot(1,2,2)
    plt.imshow(diff2, cmap='gray')
    plt.title("Difference between BEV2 and BEV3")
    plt.show()


    if bev1 is not None and bev2 is not None and bev3 is not None:
    # Compute velocities for frames 1->2 and 2->3
        velocity_x1, velocity_y1, _ = compute_velocity_vectors(bev1, bev2, x_range, y_range, dt)
        velocity_x2, velocity_y2, _ = compute_velocity_vectors(bev2, bev3, x_range, y_range, dt)

        # Apply continuity mask
        cont_mask = continuity_mask(velocity_x2, velocity_y2, alpha_cont)

        # Apply propagation mask
        prop_mask = propagation_mask(velocity_x1, velocity_y1, dt, grid_resolution, alpha_p)

        # Combine masks
        combined_mask = cont_mask * prop_mask
        #combined_mask =  prop_mask

        # Filter velocity vectors
        vx_filtered = velocity_x2 * combined_mask
        vy_filtered = velocity_y2 * combined_mask

        # Visualization
        X, Y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], bev1.shape[1]),
            np.linspace(y_range[0], y_range[1], bev1.shape[0])
        )
        plt.figure(figsize=(10, 10))
        plt.quiver(
            X, Y, vx_filtered, vy_filtered,
            angles='xy', scale_units='xy', scale=1, color='blue'
        )
        plt.title("Filtered Velocity Vectors with Combined Mask")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(color='black')
        plt.show()


 

    # if bev1 is not None and bev2 is not None and bev3 is not None:
    #     velocity_x, velocity_y, angular_velocity = compute_velocity_vectors(bev1, bev2, x_range, y_range)
    #     cont_mask = continuity_mask(velocity_x, velocity_y, alpha_cont)
    #     div_v = np.gradient(velocity_x, axis=1) + np.gradient(velocity_y, axis=0)
    #     curl_v = np.gradient(velocity_y, axis=1) - np.gradient(velocity_x, axis=0)

    # #     plt.imshow(div_v, cmap='gray')
    # #     plt.title("Divergence")
    # #     plt.colorbar()
    # #     plt.show()

    # #     plt.imshow(curl_v, cmap='gray')
    # #     plt.title("Curl")
    # #     plt.colorbar()
    # #     plt.show()




if __name__ == "__main__":
    yaml_file = "config/config.yaml"
    config = load_config(yaml_file)

    pcd_file1 = config['pcd_file1']
    pcd_file2 = config['pcd_file2']
    pcd_file3 = config['pcd_file3']

    process_and_compare_pcds(pcd_file1, pcd_file2,pcd_file3, config)