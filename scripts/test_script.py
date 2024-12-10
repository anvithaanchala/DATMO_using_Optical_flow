import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import yaml
import os
from sklearn.cluster import DBSCAN
from matplotlib import cm


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
def increase_point_density(points, expansion_factor=2, noise_std=0.01):
    """
    Increases the density of a point cloud by replicating points with small perturbations.

    Args:
        points (np.ndarray): Original point cloud (N x 3).
        expansion_factor (int): Number of times each point is replicated.
        noise_std (float): Standard deviation of random noise added to replicated points.

    Returns:
        np.ndarray: Expanded point cloud with increased density.
    """
    # Replicate each point 'expansion_factor' times
    replicated_points = np.repeat(points, expansion_factor, axis=0)

    # Add small random noise for perturbation
    noise = np.random.normal(scale=noise_std, size=replicated_points.shape)
    expanded_points = replicated_points + noise

    return expanded_points

# function to compute bev grid
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
    bev_values = (bev_values * 255).astype(np.uint8)


    return bev_values
#function to display bev
def display_bev_grid(bev, x_range, y_range, n):
    """
    Displays a Bird's Eye View (BEV) grid in a subplot.

    Parameters:
    bev (2D array): The grid to be visualized.
    x_range (tuple): Range of x-axis values (min, max).
    y_range (tuple): Range of y-axis values (min, max).
    n (int): Position in the subplot grid (e.g., 1 for top-left in a 2x2 grid).
    """
    plt.subplot(2, 2, n)  # Define the subplot layout and position
    plt.imshow(bev, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.colorbar(label='G_ij Value')
    plt.title('Bird’s Eye View (BEV)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
#selecting road end
def select_road_endpoints(bev, x_range, y_range):
    """
    Manually select two endpoints defining the road boundaries on the BEV.

    Args:
        bev (np.ndarray): BEV grid.
        x_range (tuple): X-axis range in meters.
        y_range (tuple): Y-axis range in meters.

    Returns:
        tuple: Two endpoints as (x1, y1) and (x2, y2) in real-world coordinates.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(bev, cmap='gray', origin='lower', extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    plt.title("Select Two Endpoints Defining the Road")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")

    points = plt.ginput(2)  # Allow user to click two points
    plt.close()

    print(f"Selected Points: {points}")
    return points

#function that converts pcd into bev
'''
Flip the pcd 
Ground plane segmentation 
Implementing ROI 
Computing BEV
'''
def preprocess_pcd(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    #Flip the point cloud horizontally
    # Flip the point cloud both vertically and horizontally
    points[:, 0] = -points[:, 0]  # Flip horizontally (X-axis)
    #points[:, 1] = -points[:, 1]  # Flip vertically (Y-axis)

    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([flipped_pcd])

    # Ground removal using RANSAC
    plane_model, inliers = flipped_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=5000)
    non_ground = flipped_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)

    # Filter points within ROI
    roi_points = filter_points_in_roi(non_ground_points, roi_bounds)
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    #o3d.visualization.draw_geometries([roi_pcd])
    if roi_points.size == 0:
        print(f"No ROI points for {pcd_file}. Adjust ROI bounds.")
        return None
    expanded_points = increase_point_density(roi_points, expansion_factor=10, noise_std=0.01)
    expanded_pcd = o3d.geometry.PointCloud()
    expanded_pcd.points = o3d.utility.Vector3dVector(expanded_points)
    #o3d.visualization.draw_geometries([expanded_pcd])
    # Generate BEV grid
    bev_grid = compute_bev_grid(expanded_points, grid_resolution, x_range, y_range, h_max=z_max)

    return bev_grid
#function to compute velocity vectors from bev
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
    X, Y = np.meshgrid(
    np.linspace(x_range[0], x_range[1], velocity_x.shape[1]),
    np.linspace(y_range[0], y_range[1], velocity_x.shape[0])
)
    plt.figure(figsize=(10, 10))
    plt.quiver(X, Y, velocity_x, velocity_y, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.title("Velocity Vector Grid Before Filtering")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(color='black')  # Add gridlines for clarity
    plt.show()

        
    return velocity_x, velocity_y,angular_velocity

def map_points_to_velocity_grid(points, x_range, y_range, grid_resolution_x, grid_resolution_y):
    """
    Map real-world points from BEV to their corresponding indices in the velocity grid.

    Args:
        points (list): List of (x, y) points in real-world coordinates.
        x_range (tuple): X-axis range in meters.
        y_range (tuple): Y-axis range in meters.
        grid_resolution_x (float): Grid resolution in X.
        grid_resolution_y (float): Grid resolution in Y.

    Returns:
        list: List of mapped indices as (row, col) in the grid.
    """
    mapped_indices = []
    for x, y in points:
        col = int((x - x_range[0]) / grid_resolution_x)
        row = int((y - y_range[0]) / grid_resolution_y)
        mapped_indices.append((row, col))
    return mapped_indices

#propagation mask
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
#continuity mask
def continuity_mask(vx, vy, alpha_cont):
    div_v = np.gradient(vx, axis=1) + np.gradient(vy, axis=0)
    curl_v = np.gradient(vy, axis=1) - np.gradient(vx, axis=0)
    mask = (np.abs(div_v) <= alpha_cont) & (np.abs(curl_v) <= alpha_cont)
    return mask.astype(int)
    """
    Visualize clusters obtained from Connected Component Analysis with different colors.

    Args:
        labels (np.ndarray): Label matrix from connected components.
        vx_filtered (np.ndarray): X-components of velocity vectors.
        vy_filtered (np.ndarray): Y-components of velocity vectors.
    """
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 10))

    # Generate a colormap with unique colors for each cluster
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for idx, cluster_id in enumerate(unique_labels):
        if cluster_id == 0:  # Background (label 0)
            continue

        # Extract cluster points
        cluster_mask = (labels == cluster_id)
        cluster_y, cluster_x = np.nonzero(cluster_mask)  # Get grid coordinates
        cluster_vx = vx_filtered[cluster_mask]
        cluster_vy = vy_filtered[cluster_mask]

        if cluster_vx.size == 0 or cluster_vy.size == 0:
            continue  # Skip empty clusters

        # Visualize the velocity vectors in this cluster
        plt.quiver(
            cluster_x, cluster_y,  # Grid coordinates
            cluster_vx, cluster_vy,  # Velocity vectors
            angles="xy", scale_units="xy", scale=1, color=colors[idx % len(colors)],
            label=f"Cluster {cluster_id}"
        )

    plt.title("Clusters from Connected Component Analysis with Colors")
    plt.xlabel("X (grid cells)")
    plt.ylabel("Y (grid cells)")
    plt.legend()
    plt.grid()
    plt.show()
#dbscan for clustering 
def dbscan_clustering(vx_filtered, vy_filtered, valid_mask, eps=1.0, min_samples=5):
    """
    Perform DBSCAN clustering on velocity vectors.

    Args:
        vx_filtered (np.ndarray): X-components of velocity vectors.
        vy_filtered (np.ndarray): Y-components of velocity vectors.
        valid_mask (np.ndarray): Binary mask of valid velocity vectors.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.

    Returns:
        np.ndarray: Cluster labels for each valid point (-1 for noise).
        np.ndarray: Indices of valid points in the original grid.
    """
    # Get the indices of valid points
    valid_indices = np.array(np.nonzero(valid_mask)).T

    # Get the corresponding velocity components
    valid_vx = vx_filtered[valid_mask]
    valid_vy = vy_filtered[valid_mask]

    # Combine spatial and velocity features
    features = np.column_stack((valid_indices, valid_vx, valid_vy))

    # Apply DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

    return clustering.labels_, valid_indices
def calculate_dbscan_cluster_velocities(labels, valid_indices, vx_filtered, vy_filtered):
    """
    Calculate velocities for each DBSCAN cluster.

    Args:
        labels (np.ndarray): Cluster labels for each valid point (-1 for noise).
        valid_indices (np.ndarray): Indices of valid points in the original grid.
        vx_filtered (np.ndarray): X-components of velocity vectors.
        vy_filtered (np.ndarray): Y-components of velocity vectors.

    Returns:
        dict: A dictionary with cluster IDs as keys and average velocities as values.
    """
    cluster_velocities = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise points
            continue

        # Get the indices of points belonging to this cluster
        cluster_points = valid_indices[labels == cluster_id]

        # Extract the velocity components for the cluster
        vx_cluster = vx_filtered[cluster_points[:, 0], cluster_points[:, 1]]
        vy_cluster = vy_filtered[cluster_points[:, 0], cluster_points[:, 1]]

        # Calculate the average velocity magnitude for the cluster
        velocity_magnitude = np.sqrt(vx_cluster**2 + vy_cluster**2)
        avg_velocity = np.mean(velocity_magnitude)

        cluster_velocities[cluster_id] = avg_velocity

    return cluster_velocities
def filter_clusters_by_roi(db_labels, valid_indices, velocity_grid, valid_mask, road_indices):
    """
    Filter DBSCAN clusters based on the road-defined ROI.

    Args:
        db_labels (np.ndarray): Cluster labels from DBSCAN.
        valid_indices (np.ndarray): Spatial indices of valid grid cells.
        velocity_grid (tuple): Velocity grid components (vx, vy).
        valid_mask (np.ndarray): Binary mask of valid velocity vectors.
        road_indices (list): Two indices defining the road region [(row1, col1), (row2, col2)].

    Returns:
        tuple: Filtered cluster labels, filtered indices, and velocity components (vx, vy).
    """
    (row1, col1), (row2, col2) = road_indices
    min_row, max_row = min(row1, row2), max(row1, row2)
    min_col, max_col = min(col1, col2), max(col1, col2)

    # Filter valid indices and corresponding labels
    roi_mask = (
        (valid_indices[:, 0] >= min_row) & (valid_indices[:, 0] <= max_row) &
        (valid_indices[:, 1] >= min_col) & (valid_indices[:, 1] <= max_col)
    )

    # Apply mask to valid_indices and labels
    filtered_indices = valid_indices[roi_mask]
    filtered_labels = db_labels[roi_mask]

    # Apply mask to velocity components
    valid_vx = velocity_grid[0][valid_mask]  # Extract valid vx
    valid_vy = velocity_grid[1][valid_mask]  # Extract valid vy
    vx_filtered = valid_vx[roi_mask]  # Filtered vx
    vy_filtered = valid_vy[roi_mask]  # Filtered vy

    return filtered_labels, filtered_indices, vx_filtered, vy_filtered


def visualize_filtered_clusters(labels, indices, vx, vy, x_range, y_range, grid_resolution_x, grid_resolution_y):
    """
    Visualize the filtered DBSCAN clusters on the velocity vector grid and overlay average velocities.

    Args:
        labels (np.ndarray): Filtered cluster labels.
        indices (np.ndarray): Spatial indices of filtered points.
        vx (np.ndarray): X-components of velocity vectors.
        vy (np.ndarray): Y-components of velocity vectors.
        x_range (tuple): X-axis range in meters.
        y_range (tuple): Y-axis range in meters.
        grid_resolution_x (float): Grid resolution in X.
        grid_resolution_y (float): Grid resolution in Y.
    """
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    colormap = plt.cm.get_cmap("tab10", len(unique_labels))

    # Dictionary to store average velocities for each cluster
    cluster_velocities = {}

    for idx, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            color = "gray"
            label = "Noise"
        else:
            color = colormap(idx % 10)
            label = f"Cluster {cluster_id}"

        cluster_mask = labels == cluster_id
        cluster_points = indices[cluster_mask]
        cluster_vx = vx[cluster_mask]
        cluster_vy = vy[cluster_mask]

        # Calculate average velocity magnitude for the cluster
        velocity_magnitude = np.sqrt(cluster_vx**2 + cluster_vy**2)
        avg_velocity = np.mean(velocity_magnitude) if len(velocity_magnitude) > 0 else 0
        cluster_velocities[cluster_id] = avg_velocity

        # Plot velocity vectors for the cluster
        plt.quiver(
            cluster_points[:, 1] * grid_resolution_x + x_range[0],  # Convert grid index to meters
            cluster_points[:, 0] * grid_resolution_y + y_range[0],
            cluster_vx, cluster_vy,
            angles="xy", scale_units="xy", scale=1, color=color, label=label
        )

        # Annotate the cluster with average velocity
        if cluster_id != -1 and len(cluster_points) > 0:
            cluster_centroid_x = np.mean(cluster_points[:, 1]) * grid_resolution_x + x_range[0]
            cluster_centroid_y = np.mean(cluster_points[:, 0]) * grid_resolution_y + y_range[0]
            plt.text(
                cluster_centroid_x,
                cluster_centroid_y,
                f"Vel: {avg_velocity:.2f} m/s",
                color="black",
                fontsize=8,
                ha="center"
            )

    # Add legend dynamically
    max_legend_entries = 10
    if len(unique_labels) <= max_legend_entries:
        plt.legend(loc="upper right")
    else:
        print(f"Too many clusters ({len(unique_labels)}) for legend display. Showing only visualization.")

    plt.title("Filtered DBSCAN Clusters with Velocities")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(color='black', linestyle='--', linewidth=0.5)
    plt.show()



#---------------------------------------main pipeline------------------------------------------------#
def process_and_compare_pcds(pcd_file1, pcd_file2, pcd_file3, config):
    # Get values from the config YAML
    grid_resolution = config['grid_resolution']
    x_range = config['x_range']
    y_range = config['y_range']
    z_max = config['z_max']
    roi_bounds = config['roi_bounds']
    alpha_p = config['masks']['alpha_p'][0]
    alpha_cont = config['masks']['alpha_cont'][0]
    dt = config['dt']

    # Preprocess PCDs and compute BEVs
    bev1 = preprocess_pcd(pcd_file1, grid_resolution, x_range, y_range, z_max, roi_bounds)
    bev2 = preprocess_pcd(pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds)
    bev3 = preprocess_pcd(pcd_file3, grid_resolution, x_range, y_range, z_max, roi_bounds)

    plt.figure(figsize=(10, 10))
    display_bev_grid(bev1, x_range, y_range, 1)
    display_bev_grid(bev2, x_range, y_range, 2)
    display_bev_grid(bev3, x_range, y_range, 3)
    plt.tight_layout()
    plt.show()

    # Select road endpoints on the BEV
    print("Select the two road endpoints in the BEV:")
    road_endpoints = select_road_endpoints(bev1, x_range, y_range)
    print(f"Selected road endpoints: {road_endpoints}")

    # Map the selected road endpoints to velocity grid indices
    grid_res_x, grid_res_y = grid_resolution
    road_indices = map_points_to_velocity_grid(road_endpoints, x_range, y_range, grid_res_x, grid_res_y)
    print(f"Mapped road indices to velocity grid: {road_indices}")

    # Compute velocity vectors for frames 1->2 and 2->3
    velocity_x1, velocity_y1, _ = compute_velocity_vectors(bev1, bev2, x_range, y_range, dt)
    velocity_x2, velocity_y2, _ = compute_velocity_vectors(bev2, bev3, x_range, y_range, dt)

    # Apply masks and filter velocity vectors
    cont_mask = continuity_mask(velocity_x2, velocity_y2, alpha_cont)
    prop_mask = propagation_mask(velocity_x1, velocity_y1, dt, grid_resolution, alpha_p)
    combined_mask = cont_mask * prop_mask
    vx_filtered = velocity_x2 * combined_mask
    vy_filtered = velocity_y2 * combined_mask
    velocity_magnitude = np.sqrt(vx_filtered**2 + vy_filtered**2)
    valid_mask = velocity_magnitude > 0.1  # Threshold for significant motion

    vx_filtered = vx_filtered * valid_mask
    vy_filtered = vy_filtered * valid_mask

    # Perform DBSCAN clustering
    db_labels, valid_indices = dbscan_clustering(vx_filtered, vy_filtered, valid_mask, eps=3.0, min_samples=3)

    # Filter DBSCAN clusters based on the road-defined ROI
    filtered_labels, filtered_indices, vx_roi, vy_roi = filter_clusters_by_roi(
    db_labels, valid_indices, (vx_filtered, vy_filtered), valid_mask, road_indices
)

    print(f"valid_mask shape: {valid_mask.shape}")
    print(f"valid_indices shape: {valid_indices.shape}")
    print(f"Filtered indices shape: {filtered_indices.shape}")
    print(f"Filtered vx shape: {vx_filtered.shape}")
    print(f"Filtered vy shape: {vy_filtered.shape}")
    # Visualize filtered DBSCAN clusters
    visualize_filtered_clusters(filtered_labels, filtered_indices, vx_roi, vy_roi, x_range, y_range, grid_res_x, grid_res_y)

    print("Processing and visualization complete.")

if __name__ == "__main__":
    yaml_file = "config/config.yaml"
    config = load_config(yaml_file)

    pcd_file1 = config['pcd_file1']
    pcd_file2 = config['pcd_file2']
    pcd_file3 = config['pcd_file3']

    process_and_compare_pcds(pcd_file1, pcd_file2,pcd_file3, config)