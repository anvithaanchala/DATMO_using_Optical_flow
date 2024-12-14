import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import yaml
import os
from sklearn.cluster import DBSCAN
from matplotlib import cm
from matplotlib.path import Path
from shapely.geometry import Polygon, Point
import csv
from saving_utils import (
    save_all_filtered_velocities_to_csv,
    save_bev,
    save_velocity_grid,
    save_all_velocities_to_csv,
    save_dbscan_results,
    save_ekf_tracks,
    print_final_track_velocities,
)
output_dir = "Varying_velocity_single_target_car_moving_stop_sign"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Load parameters from YAML
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
#-------------------------------------------------------------------------------------Preprocessing functions ----------------------------------------------------------------------------------
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
#function that converts pcd into bev
def preprocess_pcd(pcd_file, grid_resolution, x_range, y_range, z_max, roi_bounds):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    #Flip the point cloud horizontally
    # Flip the point cloud both vertically and horizontally
    points[:, 0] = -points[:, 0]  # Flip horizontally (X-axis)
   

    flipped_pcd = o3d.geometry.PointCloud()
    flipped_pcd.points = o3d.utility.Vector3dVector(points)
    

    # Ground removal using RANSAC
    plane_model, inliers = flipped_pcd.segment_plane(distance_threshold=0.5, ransac_n=5, num_iterations=5000)
    non_ground = flipped_pcd.select_by_index(inliers, invert=True)
    non_ground_points = np.asarray(non_ground.points)
    #print(f"Points after ground removal: {non_ground_points.shape[0]}")

    # Filter points within ROI
    roi_points = filter_points_in_roi(non_ground_points, roi_bounds)
    
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    
    if roi_points.size == 0:
        print(f"No ROI points for {pcd_file}. Adjust ROI bounds.")
        return None
    expanded_points = increase_point_density(roi_points, expansion_factor=10, noise_std=0.01)
    expanded_pcd = o3d.geometry.PointCloud()
    expanded_pcd.points = o3d.utility.Vector3dVector(expanded_points)
    
    # Generate BEV grid
    bev_grid = compute_bev_grid(expanded_points, grid_resolution, x_range, y_range, h_max=z_max)
    print(f"BEV grid computed for file: {pcd_file}")

    return bev_grid
#----------------------------------------------------------------------------------------------------BEV utils ---------------------------------------------------------------------------
# function to compute bev grid
def compute_bev_grid(points, grid_resolution, x_range, y_range, a=0.5, b=0.5, h_max=5.0):
    w, h = grid_resolution
    x_bins = np.arange(x_range[0], x_range[1], w)
    y_bins = np.arange(y_range[0], y_range[1], h)
    #print(h_max)

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

#---------------------------------------------------------------------------------Velocity from bev------------------------------------------------------------------------------------------

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
    print("Bev difference is",bev1-bev2)
    flow = cv2.calcOpticalFlowFarneback(bev1.astype(np.float32), bev2.astype(np.float32), None, **farneback_params)
    vx, vy = flow[..., 0], flow[..., 1]
    print(f"Optical flow raw X velocities: {np.mean(vx)}")
    print(f"Optical flow raw Y velocities: {np.mean(vy)}")
    # Convert flow to real-world velocities
    pixel_size_x = (x_range[1] - x_range[0]) / bev1.shape[1]
    pixel_size_y = (y_range[1] - y_range[0]) / bev1.shape[0]
    velocity_x = vx * pixel_size_x
    velocity_y = vy * pixel_size_y
    print(f"Converted X velocities: {np.mean(velocity_x)}")
    print(f"Converted Y velocities: {np.mean(velocity_y)}")
    # Compute derivatives
    dvx_dy, dvx_dx = np.gradient(velocity_x)
    dvy_dy, dvy_dx = np.gradient(velocity_y)
    print(f"Velocity X gradient dx: {dvx_dx}, dy: {dvx_dy}")
    print(f"Velocity Y gradient dx: {dvy_dx}, dy: {dvy_dy}")

# Compute angular velocity (z-axis rotation) 
    angular_velocity = dvy_dx - dvx_dy  
    print(f"Angular velocity: {np.mean(angular_velocity)}")

        
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

def propagation_mask_with_acceleration(vx, vy, ax, ay, dt, grid_resolution, alpha_p):
    """
    Create a propagation mask considering both velocity and acceleration.

    Args:
        vx (np.ndarray): X-component of velocity.
        vy (np.ndarray): Y-component of velocity.
        ax (np.ndarray): X-component of acceleration.
        ay (np.ndarray): Y-component of acceleration.
        dt (float): Time step.
        grid_resolution (tuple): Grid resolution (dx, dy).
        alpha_p (float): Threshold for propagation difference.

    Returns:
        np.ndarray: Propagation mask (binary).
    """
    h, w = vx.shape
    propagated_vx = np.zeros_like(vx)
    propagated_vy = np.zeros_like(vy)
    
    dx, dy = grid_resolution

    # Iterate over grid cells
    for i in range(h):
        for j in range(w):
            # Predicted positions with acceleration
            i_prime = int(i + np.floor((vx[i, j] * dt + 0.5 * ax[i, j] * dt**2) / dx))
            j_prime = int(j + np.floor((vy[i, j] * dt + 0.5 * ay[i, j] * dt**2) / dy))
            
            # Ensure indices are within bounds
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
#---------------------------------------------------------------------------------Clustering ------------------------------------------------------------------------------------------------
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


def filter_clusters_by_roi(db_labels, valid_indices, velocity_grid, valid_mask, road_polygon):
    """
    Filter DBSCAN clusters based on the road-defined ROI.

    Args:
        db_labels (np.ndarray): Cluster labels from DBSCAN.
        valid_indices (np.ndarray): Spatial indices of valid grid cells.
        velocity_grid (tuple): Velocity grid components (vx, vy).
        valid_mask (np.ndarray): Valid motion mask.
        road_polygon (Polygon): Shapely Polygon defining the road.

    Returns:
        tuple: Filtered labels, indices, and velocity components (vx, vy).
    """
    from shapely.geometry import Point

    # Initialize filtered results
    filtered_indices = []
    filtered_labels = []
    filtered_vx = []
    filtered_vy = []

    # Iterate over valid_indices and check if points are within the road_polygon
    for idx, (row, col) in enumerate(valid_indices):
        point = Point(col, row)
        if road_polygon.contains(point):
            filtered_indices.append(valid_indices[idx])
            filtered_labels.append(db_labels[idx])
            filtered_vx.append(velocity_grid[0][row, col])
            filtered_vy.append(velocity_grid[1][row, col])

    # Convert results to numpy arrays
    filtered_indices = np.array(filtered_indices)
    filtered_labels = np.array(filtered_labels)
    filtered_vx = np.array(filtered_vx)
    filtered_vy = np.array(filtered_vy)

    return filtered_labels, filtered_indices, filtered_vx, filtered_vy

def visualize_filtered_clusters(labels, indices, vx, vy, x_range, y_range, grid_resolution_x, grid_resolution_y):
    """
    Visualize the filtered DBSCAN clusters on the velocity vector grid.

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

    # Use a predefined colormap with sufficient unique colors
    colormap = cm.get_cmap("tab10")  # Use a colormap with 10 colors

    for idx, cluster_id in enumerate(unique_labels):
        if cluster_id == -1:
            color = "gray"
            label = "Noise"
        else:
            color = colormap(idx % 10)  # Cycle through the colormap
            label = f"Cluster {cluster_id}"

        cluster_mask = labels == cluster_id
        cluster_points = indices[cluster_mask]
        cluster_vx = vx[cluster_mask]
        cluster_vy = vy[cluster_mask]

        plt.quiver(
            cluster_points[:, 1] * grid_resolution_x + x_range[0],  # Convert grid index to meters
            cluster_points[:, 0] * grid_resolution_y + y_range[0],
            cluster_vx, cluster_vy,
            angles="xy", scale_units="xy", scale=1, color=color, label=label
        )

        # Annotate the velocity for each cluster
        if cluster_id != -1 and len(cluster_vx) > 0:
            avg_velocity = np.sqrt(np.mean(cluster_vx**2 + cluster_vy**2))
            cluster_centroid_x = np.mean(cluster_points[:, 1] * grid_resolution_x + x_range[0])
            cluster_centroid_y = np.mean(cluster_points[:, 0] * grid_resolution_y + y_range[0])
            plt.text(
                cluster_centroid_x,
                cluster_centroid_y,
                f"ID: {cluster_id}\nVel: {avg_velocity:.2f}",
                color="black",
                fontsize=8,
                ha="center"
            )

    max_legend_entries = 10
    if len(unique_labels) <= max_legend_entries:
        plt.legend(loc="upper right")
    else:
        print(f"Too many clusters ({len(unique_labels)}) for legend display. Showing visualization only.")

    plt.title("Filtered DBSCAN Clusters with Velocities")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid()
    plt.show()

def extract_cluster_data(labels, indices, vx, vy):
    clusters = {}
    unique_labels = np.unique(labels)
    
    if len(labels) != len(indices):
        raise ValueError("Mismatch between labels and valid_indices dimensions.")
    
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
        cluster_mask = labels == label
        if len(cluster_mask) != len(indices):
            raise ValueError("Cluster mask size does not match valid_indices.")
        
        cluster_points = indices[cluster_mask]
        
        if np.any(cluster_points[:, 0] >= vx.shape[0]) or np.any(cluster_points[:, 1] >= vy.shape[1]):
            raise IndexError("Cluster points are out of bounds for velocity grid.")
        
        cluster_vx = vx[cluster_points[:, 0], cluster_points[:, 1]]
        cluster_vy = vy[cluster_points[:, 0], cluster_points[:, 1]]
        
        centroid = np.mean(cluster_points, axis=0)
        velocity = [np.mean(cluster_vx), np.mean(cluster_vy)]
        cov_matrix = np.cov(cluster_points.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        
        clusters[label] = {
            'centroid': centroid,
            'measurement': [centroid[0], centroid[1], velocity[0], velocity[1]],
            'eigenvalues': eigenvalues
        }
    return clusters

#---------------------------------------------------------------------------------------------------------EKF-----------------------------------------------------------------------------
class EKF:   
    def __init__(self, state, process_noise, measurement_noise):
            self.state = np.array(state)
            self.P = np.eye(4)  # State covariance matrix
            self.Q = process_noise  # Process noise covariance
            self.R = measurement_noise  # Measurement noise covariance
            self.F = np.eye(4)  # State transition model
            self.H = np.eye(4)  # Measurement model

    def predict(self, dt, u):

            v, omega = u
            theta = self.state[2]
            self.F[0, 2] = dt
            self.F[1, 3] = dt

            # Update state using dynamic model (Equation 7)
            self.state[0] += self.state[3] * np.cos(theta) * dt
            self.state[1] += self.state[3] * np.sin(theta) * dt
            self.state[2] += omega * dt
            self.state[3] += v * dt

            # Update covariance
            self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, measurement):
                y = measurement - np.dot(self.H, self.state)  # Measurement residual
                S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
                K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
                self.state += np.dot(K, y)
                self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)
def track_clusters(tracks, clusters, dt, process_noise, measurement_noise, gamma):
            """
            Track clusters using EKF and assign them via GNN.
            Args:
                tracks (dict): Existing tracks.
                clusters (dict): Detected clusters with centroids and velocities.
                dt (float): Time step.
                process_noise (np.ndarray): Process noise covariance matrix.
                measurement_noise (np.ndarray): Measurement noise covariance matrix.
                gamma (float): Threshold for assigning clusters to tracks.
            Returns:
                dict: Updated tracks.
            """
            new_tracks = {}
            for cluster_id, cluster in clusters.items():
                matched_track = None
                min_distance = float('inf')
                cluster_feature = np.array([*cluster['centroid'], *cluster['eigenvalues']])

                for track_id, ekf in tracks.items():
                    predicted_position = ekf.state[:2]
                    track_feature = np.array([predicted_position[0], predicted_position[1], 0, 0])  # Add eigenvalues if stored in tracks
                    distance = np.linalg.norm(cluster_feature - track_feature)

                    if distance < min_distance and distance < gamma:
                        matched_track = track_id
                        min_distance = distance

                if matched_track is not None:
                    ekf = tracks[matched_track]
                    ekf.predict(dt, cluster['measurement'][2:])  # Pass velocities as control inputs
                    ekf.update(cluster['measurement'])
                    new_tracks[matched_track] = ekf
                else:
                    new_track_id = max(tracks.keys(), default=0) + 1
                    ekf = EKF(cluster['measurement'], process_noise, measurement_noise)
                    new_tracks[new_track_id] = ekf

            return new_tracks
def manage_tracks(tracks, track_lifetimes, confirmed_tracks, M1, N1, M2, N2):

            for track_id in list(tracks.keys()):
                if track_id in confirmed_tracks:
                    if track_lifetimes[track_id] > N2 and track_lifetimes[track_id] - M2 <= N2:
                        del tracks[track_id]
                else:
                    if track_lifetimes[track_id] >= N1 and track_lifetimes[track_id] - M1 <= N1:
                        confirmed_tracks.add(track_id)

def visualize_tracks(tracks):
                """
                Visualize EKF tracks.

                Args:
                    tracks (dict): EKF tracks.
                """
                plt.figure(figsize=(10, 10))
                for track_id, ekf in tracks.items():
                    state = ekf.state
                    #plt.plot(state[0], state[1], 'o', label=f'Track {track_id=}')
                    plt.plot(state[0], state[1], 'o', label=f'Track 1')
                    plt.quiver(
                        state[0], state[1], state[2], state[3],
                        #angles="xy", scale_units="xy", scale=1, label=f'Velocity {track_id=}'
                        angles="xy", scale_units="xy", scale=2, label=f'Velocity 1'
                    )
                plt.title("Tracked Objects")
                plt.xlabel("X (meters)")
                plt.ylabel("Y (meters)")
                plt.legend()
                plt.grid()
                plt.show()
#------------------------------------------------------------------------------------------------------main pipeline---------------------------------------------------------------------------
def process_multiple_frames(pcd_files, config):
    grid_resolution = config['grid_resolution']
    x_range = config['x_range']
    y_range = config['y_range']
    z_max = config['z_max']
    roi_bounds = config['roi_bounds']
    alpha_p = config['masks']['alpha_p'][0]
    alpha_cont = config['masks']['alpha_cont'][0]
    dt = config['dt']
    dbscan_params = config['dbscan_params']

    # Initialize tracking variables
    tracks = {}
    track_lifetimes = {}
    confirmed_tracks = set()
    csv_file = os.path.join(output_dir, "curved_Vehicle_Intersection.csv")
    if os.path.exists(csv_file):
        os.remove(csv_file)  # Remove the file if it already exists to start fresh
    prev_velocity_x = None
    prev_velocity_y = None
    for i in range(len(pcd_files) - 1):
        pcd_file1 = pcd_files[i]
        pcd_file2 = pcd_files[i + 1]

        try:
            print(f"Processing frames {i} and {i + 1}: {pcd_file1}, {pcd_file2}")
            # Preprocess PCDs and compute BEVs
            bev1 = preprocess_pcd(pcd_file1, grid_resolution, x_range, y_range, z_max, roi_bounds)
            bev2 = preprocess_pcd(pcd_file2, grid_resolution, x_range, y_range, z_max, roi_bounds)
            save_bev(bev1, i)
            save_bev(bev2, i + 1)
            if bev1 is None or bev2 is None:
                print(f"Invalid BEV grid for frames {i} and {i + 1}. Skipping.")
                continue

            # Compute velocity vectors
            velocity_x, velocity_y, _ = compute_velocity_vectors(bev1, bev2, x_range, y_range, dt)
            

            save_velocity_grid(velocity_x, velocity_y, i)
            if prev_velocity_x is not None and prev_velocity_y is not None:
                ax = (velocity_x - prev_velocity_x) / dt
                ay = (velocity_y - prev_velocity_y) / dt
            else:
                ax = np.zeros_like(velocity_x)
                ay = np.zeros_like(velocity_y)

            ax = (velocity_x - prev_velocity_x) / dt
            ay = (velocity_y - prev_velocity_y) / dt

            prev_velocity_x = velocity_x
            prev_velocity_y = velocity_y


            # Apply masks and filter valid motion
            cont_mask = continuity_mask(velocity_x, velocity_y, alpha_cont)
            combined_mask = cont_mask
            

            vx_filtered = velocity_x * combined_mask
            vy_filtered = velocity_y * combined_mask
            velocity_magnitude = np.sqrt(vx_filtered**2 + vy_filtered**2)
            # Compute angular velocity
            dvx_dy, dvx_dx = np.gradient(vx_filtered)
            dvy_dy, dvy_dx = np.gradient(vy_filtered)
            angular_velocity = dvy_dx - dvx_dy  # Curl formula
            save_all_filtered_velocities_to_csv(vx_filtered, vy_filtered, velocity_magnitude, angular_velocity, i)

            valid_mask = velocity_magnitude > 0.1
            save_velocity_grid(vx_filtered, vy_filtered, i)
            # Perform DBSCAN clustering
            db_labels, valid_indices = dbscan_clustering(vx_filtered, vy_filtered, valid_mask, **dbscan_params)
            save_dbscan_results(db_labels, valid_indices, i)
            # Extract clusters
            clusters = extract_cluster_data(db_labels, valid_indices, vx_filtered, vy_filtered)

            # Update tracks with EKF
            tracks = track_clusters(tracks, clusters, dt, np.eye(4) * 0.1, np.eye(4) * 0.05, gamma=0.5)
            save_ekf_tracks(tracks, i)
            save_all_velocities_to_csv(tracks, i, csv_file)
            # Update track lifetimes
            for track_id in list(track_lifetimes.keys()):
                if track_id in tracks:
                    track_lifetimes[track_id] += 1
                else:
                    del track_lifetimes[track_id]

            # Add new tracks with initial lifetime
            for track_id in tracks:
                if track_id not in track_lifetimes:
                    track_lifetimes[track_id] = 1

            # Manage tracks
            manage_tracks(tracks, track_lifetimes, confirmed_tracks, M1=1, N1=4, M2=10, N2=15)
        except Exception as e:
            print(f"Error processing frames {i} and {i + 1}: {e}")
            continue

    # Visualize final tracks
    print_final_track_velocities(tracks)
    visualize_tracks(tracks)


if __name__ == "__main__":
    yaml_file = "config/config.yaml"
    config = load_config(yaml_file)

    pcd_files = config['pcd_files']
    pcd_files = sorted(pcd_files)
    process_multiple_frames(pcd_files, config)
