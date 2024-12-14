import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from natsort import natsorted
import os

def load_config(yaml_file):
    import yaml
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Preprocessing Functions
def filter_points_in_roi(points, roi_bounds):
    x_min, x_max, y_min, y_max, z_min, z_max = roi_bounds
    return points[
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    ]

def increase_point_density(points, expansion_factor=2, noise_std=0.01):
    replicated_points = np.repeat(points, expansion_factor, axis=0)
    noise = np.random.normal(scale=noise_std, size=replicated_points.shape)
    return replicated_points + noise

def preprocess_pcd(pcd_file, roi_bounds):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # Flip the point cloud horizontally
    points[:, 0] = -points[:, 0]
    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)

    # Ground removal using RANSAC
    plane_model, inliers = flipped_pcd.segment_plane(distance_threshold=0.5, ransac_n=5, num_iterations=5000)
    non_ground = flipped_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)

    # Filter points within ROI
    roi_points = filter_points_in_roi(non_ground_points, roi_bounds)
    if roi_points.size == 0:
        print(f"No ROI points for {pcd_file}. Skipping.")
        return None, None

    expanded_points = increase_point_density(roi_points, expansion_factor=10, noise_std=0.01)
    expanded_pcd = o3d.geometry.PointCloud()
    expanded_pcd.points = o3d.utility.Vector3dVector(expanded_points)

    return roi_points, expanded_pcd

def dbscan_clustering(points, eps=0.5, min_samples=10):
    """
    Perform DBSCAN clustering with specified parameters.
    """
    print(f"Running DBSCAN with eps={eps} and min_samples={min_samples}")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return clustering.labels_
# Utility Functions
def point_to_grid_index(point, grid_size, cell_resolution):
    x, y = point[:2]
    cell_resolution_x, cell_resolution_y = cell_resolution  # Unpack cell_resolution list
    row = int((x + grid_size * cell_resolution_x / 2) // cell_resolution_x)
    col = int((y + grid_size * cell_resolution_y / 2) // cell_resolution_y)
    return row, col

def compute_motion_residuals(points, previous_points, transformation):
    """
    Compute motion residuals after compensating for ego-motion.
    """
    # Transform previous points
    previous_points_transformed = (transformation[:3, :3] @ previous_points.T).T + transformation[:3, 3]

    # Handle mismatched point cloud sizes using nearest neighbors
    if len(points) != len(previous_points_transformed):
        # Create Open3D PointCloud for KDTree
        o3d_previous = o3d.geometry.PointCloud()
        o3d_previous.points = o3d.utility.Vector3dVector(previous_points_transformed)
        kdtree = o3d.geometry.KDTreeFlann(o3d_previous)

        # Align points using nearest neighbors
        aligned_points = []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            aligned_points.append(previous_points_transformed[idx[0]])
        previous_points_transformed = np.array(aligned_points)

    # Compute residuals
    residuals = np.linalg.norm(points - previous_points_transformed, axis=1)
    return residuals

def classify_points_with_gmfa(points, previous_points, transformation, static_threshold, moving_threshold):
    """
    Classify points into static, moving, or uncertain using GMFA and residual motion.
    """
    if len(points) == 0 or len(previous_points) == 0:
        return []  # Return an empty classification if either input is empty

    # Transform previous points
    previous_points_transformed = (transformation[:3, :3] @ previous_points.T).T + transformation[:3, 3]

    # Handle mismatched point cloud sizes using nearest neighbors
    if len(points) != len(previous_points_transformed):
        # Create Open3D PointCloud for KDTree
        o3d_previous = o3d.geometry.PointCloud()
        o3d_previous.points = o3d.utility.Vector3dVector(previous_points_transformed)
        kdtree = o3d.geometry.KDTreeFlann(o3d_previous)

        # Align points using nearest neighbors
        aligned_points = []
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            aligned_points.append(previous_points_transformed[idx[0]])
        previous_points_transformed = np.array(aligned_points)

    # Check again if arrays are empty after alignment
    if len(points) == 0 or len(previous_points_transformed) == 0:
        return []  # Return an empty classification if alignment results in empty arrays

    # Compute residuals and classify points
    residuals = np.linalg.norm(points - previous_points_transformed, axis=1)
    classifications = [
        3 if residual < static_threshold else 2 if residual > moving_threshold else 1
        for residual in residuals
    ]
    return classifications


def update_som_with_gmfa(som_grid, points, residuals, static_threshold, moving_threshold, grid_size, cell_resolution):
    for point, residual in zip(points, residuals):
        row, col = point_to_grid_index(point, grid_size, cell_resolution)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            if residual < static_threshold:
                som_grid[row, col] = min(som_grid[row, col] + 0.1, 0.95)
            elif residual > moving_threshold:
                som_grid[row, col] = max(som_grid[row, col] - 0.1, 0.05)
    return som_grid

# EKF Functions
def ekf_predict(x, P, dt):
    F = np.array([
        [1, 0, dt,  0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1]
    ])
    Q = np.diag([0.1, 0.1, 0.01, 0.01])
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, H, R):
    y = z - (H @ x_pred)
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_updated = x_pred + K @ y
    P_updated = (np.eye(len(P_pred)) - K @ H) @ P_pred
    return x_updated, P_updated
def calculate_feature_vector(cluster_points):
    centroid = np.mean(cluster_points, axis=0)
    covariance_matrix = np.cov(cluster_points, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    lambda_max, lambda_min = np.max(eigenvalues), np.min(eigenvalues)
    return np.array([centroid[0], centroid[1], lambda_max, lambda_min])
def group_points_by_cluster(points, labels, max_samples=None):
    """
    Groups points by their DBSCAN cluster labels and filters clusters by max_samples.
    """
    clusters = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    return clusters
def assign_clusters_to_tracks(previous_tracks, current_clusters, cost_threshold=1.0):
    """
    Assign clusters to tracks using a cost matrix and the Hungarian algorithm.
    Handles cases where tracks or clusters are empty.
    """
    if not previous_tracks or not current_clusters:
        # If no previous tracks or current clusters, return no assignments
        return {}, set(range(len(current_clusters)))

    previous_features = np.array([track['features'] for track in previous_tracks])
    current_features = np.array([calculate_feature_vector(cluster) for cluster in current_clusters])

    # Ensure the features are 2-dimensional
    if previous_features.ndim == 1:
        previous_features = previous_features.reshape(1, -1)
    if current_features.ndim == 1:
        current_features = current_features.reshape(1, -1)

    # Compute cost matrix
    cost_matrix = cdist(previous_features, current_features, metric='euclidean')

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = {}
    unassigned_clusters = set(range(len(current_clusters)))
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < cost_threshold:
            assignments[i] = j
            unassigned_clusters.discard(j)

    return assignments, unassigned_clusters


def update_tracks(previous_tracks, assignments, current_clusters, dt=0.1):
    updated_tracks = []
    for track_idx, cluster_idx in assignments.items():
        cluster_features = calculate_feature_vector(current_clusters[cluster_idx])
        position = cluster_features[:2]
        track = previous_tracks[track_idx]

        # Compute velocity based on position change
        previous_position = track['state'][:2].flatten()
        velocity = (position - previous_position) / dt

        track['features'] = cluster_features
        track['state'][:2] = position.reshape(-1, 1)  # Update position
        track['state'][2:] = velocity.reshape(-1, 1)  # Update velocity
        track['age'] += 1
        updated_tracks.append(track)
    return updated_tracks


def initialize_new_tracks(unassigned_clusters, current_clusters, previous_positions=None, dt=0.1):
    new_tracks = []
    for cluster_idx in unassigned_clusters:
        cluster_points = current_clusters[cluster_idx]
        if cluster_points.size == 0:  # Skip empty clusters
            continue

        feature_vector = calculate_feature_vector(cluster_points)
        position = feature_vector[:2]

        # Compute velocity based on previous positions if available
        if previous_positions is not None and cluster_idx in previous_positions:
            velocity = (position - previous_positions[cluster_idx]) / dt
        else:
            velocity = np.zeros(2)  # Initialize velocity to zero if no previous position is available

        new_tracks.append({
            'id': np.random.randint(1e5),
            'features': feature_vector,
            'state': np.hstack((position, velocity)).reshape(-1, 1),
            'covariance': np.eye(4) * 0.1,
            'age': 1
        })
    return new_tracks
import matplotlib.pyplot as plt

def visualize_positions_and_velocities(points, tracks, classifications, title="Frame Visualization"):
    """
    Visualize positions and velocity vectors of tracks, distinguishing between moving and static objects.
    """
    plt.figure(figsize=(30, 8))

    # Separate moving and static objects
    moving_points = points[np.array(classifications) == 2]  # Moving
    static_points = points[np.array(classifications) == 3]  # Static

    # Plot positions of points
    if len(static_points) > 0:
        plt.scatter(static_points[:, 0], static_points[:, 1], color='blue', label='Static Objects', alpha=0.5)
    if len(moving_points) > 0:
        plt.scatter(moving_points[:, 0], moving_points[:, 1], color='red', label='Moving Objects', alpha=0.5)

    # Plot velocity vectors
    for track in tracks:
        position = track['state'][:2].flatten()
        velocity = track['state'][2:].flatten()

        # Velocity vector
        plt.quiver(
            position[0], position[1],
            velocity[0], velocity[1],
            angles='xy', scale_units='xy', scale=1, color='green', width=0.002,
            label='Velocity Vector' if 'Velocity Vector' not in plt.gca().get_legend_handles_labels()[1] else None
        )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def estimate_transformation(source_pcd, target_pcd):
    """
    Estimate transformation between two point clouds using ICP.
    """
    threshold = 0.02  # Set a distance threshold for ICP
    initial_guess = np.eye(4)  # Use identity matrix as initial guess

    # Perform ICP registration
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, initial_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return result_icp.transformation

def plot_moving_vs_static(points,tracks, classifications, title="Moving vs Static Objects"):
    """
    Plot moving vs static objects based on classifications.
    """
    plt.figure(figsize=(10, 10))

    # Separate moving and static points
    moving_points = points[np.array(classifications) == 2]  # Moving
    static_points = points[np.array(classifications) == 3]  # Static

    # Plot static points
    if len(static_points) > 0:
        plt.scatter(static_points[:, 0], static_points[:, 1], c='blue', label='Static Objects', alpha=0.5)

    # Plot moving points
    if len(moving_points) > 0:
        plt.scatter(moving_points[:, 0], moving_points[:, 1], c='red', label='Moving Objects', alpha=0.5)
    positions = np.array([track['state'][:2].flatten() for track in tracks])
    if len(positions) > 0:
        plt.scatter(positions[:, 0], positions[:, 1], c='purple', label='Final Positions', alpha=0.7, s=50)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()
def visualize_final_positions_and_velocities(points, tracks, title="Moving Object Detection"):
    """
    Visualize the original point cloud with final positions and velocity vectors overlaid.
    """
    plt.figure(figsize=(10, 10))
    
    # Plot the original point cloud
    plt.scatter(points[:, 0], points[:, 1], c='gray', s=1, label="Point Cloud")
    
    # To avoid duplicate legends, use a flag
    final_position_plotted = False
    velocity_vector_plotted = False

    # Plot final positions and velocity vectors
    for track in tracks:
        position = track['state'][:2].flatten()
        velocity = track['state'][2:].flatten()

        # Plot final position
        if not final_position_plotted:
            plt.scatter(position[0], position[1], color='blue', label='Target Vehicle', alpha=0.8, s=50)
            final_position_plotted = True
        else:
            plt.scatter(position[0], position[1], color='blue', alpha=0.8, s=50)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()



def filter_moving_points_in_roi(points, classifications, roi_bounds):
    """
    Filters moving points within a specified ROI.
    """
    x_min, x_max, y_min, y_max = roi_bounds
    moving_points = points[np.array(classifications) == 2]  # Moving
    roi_moving_points = moving_points[
        (moving_points[:, 0] >= x_min) & (moving_points[:, 0] <= x_max) &
        (moving_points[:, 1] >= y_min) & (moving_points[:, 1] <= y_max)
    ]
    return roi_moving_points

import pandas as pd

# Function to plot the filtered point cloud
def plot_filtered_cloud(points, classifications, title="Filtered Point Cloud"):
    plt.figure(figsize=(10, 10))

    # Separate moving and static points
    moving_points = points[np.array(classifications) == 2]  # Moving
    static_points = points[np.array(classifications) == 3]  # Static

    # Plot static points
    if len(static_points) > 0:
        plt.scatter(static_points[:, 0], static_points[:, 1], c='blue', label='Static Objects', alpha=0.5)

    # Plot moving points
    if len(moving_points) > 0:
        plt.scatter(moving_points[:, 0], moving_points[:, 1], c='red', label='Moving Objects', alpha=0.5)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot SOM heat map
def plot_som_heat_map(som_grid, title="SOM Heat Map"):
    plt.figure(figsize=(10, 10))
    plt.imshow(som_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Occupancy Probability")
    plt.title(title)
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.show()

# Save track data to Excel
def save_tracks_to_excel(track_data_list, output_file="track_data.xlsx"):
    track_df = pd.DataFrame(track_data_list)
    track_df.to_excel(output_file, index=False)
    print(f"Track data saved to {output_file}")

if __name__ == "__main__":
    # Load configuration
    config_file = "config.yaml"  # Replace with your YAML config path
    config = load_config(config_file)

    pcd_folder = "/content/data"
    roi_bounds = config['roi_bounds']
    static_threshold = 0.2
    moving_threshold =0.6

    # SOM parameters
    grid_size = 200
    cell_resolution = [0.2, 0.2]
    som_grid = np.full((grid_size, grid_size), 0.05)

    # Initialize tracking state
    previous_pcd = None
    tracks = []
    previous_positions = {}
    track_data_list = []

    # Process each PCD file sequentially
    pcd_files = natsorted([os.path.join(pcd_folder, file) for file in os.listdir(pcd_folder) if file.endswith(".pcd")])
    if not pcd_files:
        print("No PCD files found in the folder.")
        exit()

    for i, pcd_file in enumerate(pcd_files):
        print(f"Processing Frame {i + 1}/{len(pcd_files)}: {pcd_file}")

        # Preprocess the current PCD
        roi_points, expanded_pcd = preprocess_pcd(pcd_file, roi_bounds)
        if roi_points is None:
            continue

        if previous_pcd is not None:
            # Current and previous points
            points = np.asarray(expanded_pcd.points)
            previous_points = np.asarray(previous_pcd.points)

            # Estimate transformation using ICP
            transformation = estimate_transformation(previous_pcd, expanded_pcd)

            # Classify points and compute residuals
            classifications = classify_points_with_gmfa(points, previous_points, transformation, static_threshold, moving_threshold)
            residuals = compute_motion_residuals(points, previous_points, transformation)

            # Filter moving points within the defined ROI
            moving_roi_bounds = [-20, 20, -20, 5]  # x: [-10, 10], y: [-5, 5]
            roi_moving_points = filter_moving_points_in_roi(points, classifications, moving_roi_bounds)

            if roi_moving_points.size == 0:
                print(f"No moving objects in ROI for Frame {i + 1}. Skipping.")
                continue
                
            # Perform DBSCAN clustering
            labels = dbscan_clustering(roi_moving_points, eps=config['dbscan_params']['eps'], min_samples=1000)
            clusters = group_points_by_cluster(roi_moving_points, labels)
            print(f"Frame {i}: Found {len(clusters)} clusters.")
            for cluster_idx, cluster in enumerate(clusters):
              print(f"Cluster {cluster_idx + 1}: Size {len(cluster)} points (Num samples: {len(cluster)})")
            # Update tracks and SOM
            assignments, unassigned_clusters = assign_clusters_to_tracks(tracks, clusters)
            tracks = update_tracks(tracks, assignments, clusters)
            new_tracks = initialize_new_tracks(unassigned_clusters, clusters, previous_positions, dt=0.1)
            tracks.extend(new_tracks)

            som_grid = update_som_with_gmfa(som_grid, roi_moving_points, residuals, static_threshold, moving_threshold, grid_size, cell_resolution)

            # EKF Updates for each track
            for track in tracks:
                z = np.array(track['features'][:2]).reshape(-1, 1)  # Measurement from cluster features
                track['state'], track['covariance'] = ekf_predict(track['state'], track['covariance'], dt=0.1)
                track['state'], track['covariance'] = ekf_update(track['state'], track['covariance'], z, np.array([[1, 0, 0, 0], [0, 1, 0, 0]]), np.eye(2) * 0.05)

            # Save track data
            for track in tracks:
                position = track['state'][:2].flatten()
                velocity = track['state'][2:].flatten()
                track_data_list.append({
                    'Frame': i,
                    'Track ID': track['id'],
                    'X': position[0],
                    'Y': position[1],
                    'VX': velocity[0],
                    'VY': velocity[1]
                })
                print(f"Frame {i + 1} Track States:")
                print(f"Track ID: {track['id']}, Position: {track['state'][:2].flatten()}, Velocity: {track['state'][2:].flatten()}")
                # Debugging: Print cluster sizes and assignments
                print(f"Frame {i + 1}: Found {len(clusters)} clusters.")
                for cluster_idx, cluster in enumerate(clusters):
                    print(f"Cluster {cluster_idx + 1}: Size {len(cluster)} points")

                # Debugging: Print track states
                for track in tracks:
                    print(f"Track ID: {track['id']}, Position: {track['state'][:2].flatten()}, Velocity: {track['state'][2:].flatten()}")

                # Update previous positions with current cluster centroids
                previous_positions = {idx: np.mean(cluster, axis=0)[:2] for idx, cluster in enumerate(clusters)}
                
                # Visualize point cloud with velocity vectors
            plot_som_heat_map(som_grid, title=f"Frame {i + 1}: SOM Heat Map")
            plot_moving_vs_static(points,tracks, classifications, title=f"Frame {i + 1}: Moving vs Static Objects")
            visualize_final_positions_and_velocities(points, tracks, title=f"Frame {i + 1}: Moving Object Detection")

        # Update previous frame
        previous_pcd = expanded_pcd
        # Visualization of Moving and Static Objects
       

# Save track data to Excel
    save_tracks_to_excel(track_data_list)


  