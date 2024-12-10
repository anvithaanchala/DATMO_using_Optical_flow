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
#function to preprocess bev before generating velocity vector 





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
#cluster velocity calculation
def calculate_cluster_velocities(labels, vx_filtered, vy_filtered):
    """
    Calculate average velocity magnitude for each cluster.

    Args:
        labels (np.ndarray): Cluster labels for each grid cell.
        vx_filtered (np.ndarray): X-components of velocity vectors.
        vy_filtered (np.ndarray): Y-components of velocity vectors.

    Returns:
        dict: A dictionary with cluster IDs as keys and average velocities as values.
    """
    cluster_velocities = {}
    unique_labels = np.unique(labels)

    for cluster_id in unique_labels:
        if cluster_id == 0:  # Ignore background (label 0 typically represents background)
            continue

        # Extract points in the current cluster
        cluster_mask = (labels == cluster_id)
        vx_cluster = vx_filtered[cluster_mask]
        vy_cluster = vy_filtered[cluster_mask]

        if vx_cluster.size == 0 or vy_cluster.size == 0:
            continue  # Skip empty clusters

        # Compute velocity magnitude
        velocities = np.sqrt(vx_cluster**2 + vy_cluster**2)
        avg_velocity = np.mean(velocities)

        cluster_velocities[cluster_id] = avg_velocity

    return cluster_velocities
#connected component analysis for clustering 
def connected_components_clustering(valid_mask):
    """
    Perform Connected Component Analysis (CCA) to cluster valid regions.

    Args:
        valid_mask (np.ndarray): Binary mask of valid velocity regions.

    Returns:
        num_labels (int): Number of connected components found.
        labels (np.ndarray): Label matrix where each unique value corresponds to a connected component.
    """
    # Apply connected component analysis
    num_labels, labels = cv2.connectedComponents(valid_mask.astype(np.uint8), connectivity=8)

    return num_labels, labels
#function to visualize connected components 
def visualize_connected_components(labels, vx_filtered, vy_filtered):
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


#---------------------------------------main pipeline------------------------------------------------#
def process_and_compare_pcds(pcd_file1, pcd_file2, pcd_file3, config):
    #getting values from config yaml
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

    plt.figure(figsize=(10, 10))
    display_bev_grid(bev1, x_range, y_range, 1)
    display_bev_grid(bev2, x_range, y_range, 2)
    display_bev_grid(bev3, x_range, y_range, 3)
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

    #checking difference in bevs as velocity vectors are not being generated 
    diff1 = np.abs(bev1 - bev2)
    diff2 = np.abs(bev2 - bev3)
    print("Difference between BEV grids 1 and 2 :", np.sum(diff1))
    print("Difference between BEV grids:", np.sum(diff2))


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
        vx_filtered = velocity_x2 * combined_mask
        vy_filtered = velocity_y2 * combined_mask
        velocity_magnitude = np.sqrt(vx_filtered**2 + vy_filtered**2)
        threshold = 0.1  # Define a velocity threshold (tune as needed)
        valid_mask = velocity_magnitude > threshold  # Keep only vectors with significant motion

        vx_filtered = vx_filtered * valid_mask
        vy_filtered = vy_filtered * valid_mask
        X, Y = np.meshgrid(
        np.linspace(x_range[0], x_range[1], vx_filtered.shape[1]),
        np.linspace(y_range[0], y_range[1], vy_filtered.shape[0])
    )
        plt.figure(figsize=(10, 10))
        plt.quiver(X, Y, vy_filtered, vy_filtered, angles='xy', scale_units='xy', scale=1, color='blue')
        plt.title("Velocity Vector Grid After Filtering")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.grid(color='black')  # Add gridlines for clarity
        plt.show()

        # # Perform Connected Component Analysis
        # num_labels, labels = connected_components_clustering(combined_mask)

        # # Calculate cluster velocities
        # cluster_velocities = calculate_cluster_velocities(labels, vx_filtered, vy_filtered)

        # # Find the cluster with the highest average velocity
        # if cluster_velocities:
        #     mean_velocity_cluster = np.mean(list(cluster_velocities.values()))
        #     print(f"Mean velocity: {mean_velocity_cluster:.2f}")
        #     clusters_above_mean = {cluster_id: velocity for cluster_id, velocity in cluster_velocities.items() if velocity > mean_velocity_cluster}

        #     if clusters_above_mean:
        #         print("Clusters with velocities greater than the mean:")
        #         for cluster_id, velocity in clusters_above_mean.items():
        #             print(f"Cluster {cluster_id}: Velocity = {velocity:.2f}")
        #     else:
        #         print("No clusters with velocities greater than the mean velocity.")

        # # Mask only the selected cluster
        # selected_cluster_mask = (labels >= mean_velocity_cluster)
        # vx_selected = vx_filtered * selected_cluster_mask
        # vy_selected = vx_filtered * selected_cluster_mask

        # # Visualize selected clusters with  velocity greater than the average
        # visualize_connected_components(labels, vx_filtered, vy_filtered)
        '''--------------------------------------------------#Perform DBSCAN clustering----------------------------------------------'''
        # Perform DBSCAN clustering
        db_labels, valid_indices = dbscan_clustering(vx_filtered, vy_filtered, valid_mask, eps=3.0, min_samples=3)
        unique_labels = np.unique(db_labels)

        # Print the number of clusters
        num_clusters = len(unique_labels) - (1 if -1 in db_labels else 0)
        print(f"Number of clusters obtained by DBSCAN: {num_clusters}")

        # Visualize the DBSCAN clusters
        plt.figure(figsize=(12, 12))

        # Generate a colormap using `matplotlib.cm`
        from matplotlib import cm
        colormap = cm.get_cmap("tab10", len(unique_labels))  # Use `tab10` colormap for clusters

        # Loop through each cluster
        for idx, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:  # Noise points
                color = "gray"
                label = "Noise"
            else:
                color = colormap(idx % 10)[:3]  # Get RGB values for color
                label = f"Cluster {cluster_id}"

            # Extract cluster points
            cluster_mask = db_labels == cluster_id
            cluster_points = valid_indices[cluster_mask]  # Spatial indices for the cluster
            cluster_vx = vx_filtered[valid_mask][cluster_mask]
            cluster_vy = vy_filtered[valid_mask][cluster_mask]

            # Plot the velocity vectors for this cluster
            plt.quiver(
                cluster_points[:, 1], cluster_points[:, 0],  # Grid indices
                cluster_vx, cluster_vy,
                angles="xy", scale_units="xy", scale=1, color=color, label=label
            )

            # Calculate and annotate the average velocity
            if cluster_id != -1 and len(cluster_vx) > 0:
                avg_velocity = np.sqrt(np.mean(cluster_vx**2 + cluster_vy**2))
                cluster_centroid_x = np.mean(cluster_points[:, 1])
                cluster_centroid_y = np.mean(cluster_points[:, 0])
                plt.text(
                    cluster_centroid_x,
                    cluster_centroid_y,
                    f"ID: {cluster_id}\nVel: {avg_velocity:.2f}",
                    color="black",
                    fontsize=8,
                    ha="center"
                )

        # Add legend dynamically if clusters are not too many
        max_legend_entries = 10
        if len(unique_labels) <= max_legend_entries:  # Display legend only if number of clusters is manageable
            plt.legend(loc="upper right")
        else:
            print(f"Too many clusters ({len(unique_labels)}) for legend display. Showing only visualization.")

        plt.title("DBSCAN Clusters with Velocities")
        plt.xlabel("X (grid cells)")
        plt.ylabel("Y (grid cells)")
        plt.grid(color='black', linestyle='--', linewidth=0.5)
        plt.show()



if __name__ == "__main__":
    yaml_file = "config/config.yaml"
    config = load_config(yaml_file)

    pcd_file1 = config['pcd_file1']
    pcd_file2 = config['pcd_file2']
    pcd_file3 = config['pcd_file3']

    process_and_compare_pcds(pcd_file1, pcd_file2,pcd_file3, config)