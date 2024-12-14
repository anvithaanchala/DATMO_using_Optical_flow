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

output_dir = "Varying_velocity_single_target_car_moving_stop_sign"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_all_filtered_velocities_to_csv(vx_filtered, vy_filtered, magnitude, angular_velocity, frame_index, csv_file="Varying_velocity_single_target_car_moving_stop_sign_filtered_velocities.csv"):
    """
    Save filtered linear velocities, magnitude, and angular velocities to a CSV file.

    Args:
        vx_filtered (np.ndarray): Filtered X velocities.
        vy_filtered (np.ndarray): Filtered Y velocities.
        magnitude (np.ndarray): Magnitudes of velocities.
        angular_velocity (np.ndarray): Angular velocities.
        frame_index (int): Frame index for the CSV file.
        csv_file (str): Path to the output CSV file.
    """
    # Ensure the file exists only for the first call
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if the file does not exist
        if not file_exists:
            writer.writerow(["Frame Index", "Point Index", "Filtered X Velocity", "Filtered Y Velocity", "Magnitude", "Angular Velocity"])

        # Write data for each valid point
        valid_indices = np.nonzero((vx_filtered != 0) | (vy_filtered != 0))  # Only include non-zero velocities
        for idx, (i, j) in enumerate(zip(valid_indices[0], valid_indices[1])):
            writer.writerow([
                frame_index, idx, 
                vx_filtered[i, j], vy_filtered[i, j], 
                magnitude[i, j], angular_velocity[i, j]
            ])

def print_final_track_velocities(tracks):
    """
    Print the final velocities of EKF tracks.

    Args:
        tracks (dict): Final EKF tracks.
    """
    print("Final Track Velocities:")
    for track_id, ekf in tracks.items():
        final_state = ekf.state
        final_vx, final_vy = final_state[2], final_state[3]
        final_magnitude = np.sqrt(final_vx**2 +final_vy**2)
        print(f"Track : 1 ")
        print(f"  Final Velocity: vx = {final_vx:.2f}, vy = {final_vy:.2f}")
        print(f"  Magnitude: {final_magnitude:.2f}\n")


def save_bev(bev, frame_index):
    np.save(os.path.join(output_dir, f"bev_frame_{frame_index}.npy"), bev)
    plt.imsave(os.path.join(output_dir, f"bev_frame_{frame_index}.png"), bev, cmap='gray')

def save_velocity_grid(vx, vy, frame_index):
    np.save(os.path.join(output_dir, f"velocity_x_frame_{frame_index}.npy"), vx)
    np.save(os.path.join(output_dir, f"velocity_y_frame_{frame_index}.npy"), vy)
    plt.figure(figsize=(10, 10))
    plt.quiver(vx, vy, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.title(f"Velocity Vectors for Frame {frame_index}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(os.path.join(output_dir, f"velocity_grid_frame_{frame_index}.png"))
    plt.close()

def save_all_velocities_to_csv(tracks, frame_index, csv_file="Varying_velocity_single_car_straight.csv"):
    """
    Save the filtered linear velocities, x and y velocities, and angular velocities to a single CSV file.

    Args:
        tracks (dict): EKF tracks containing state information.
        frame_index (int): Frame index for the CSV file.
        csv_file (str): Path to the output CSV file.
    """
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        
        # Write header only if the file does not exist
        if not file_exists:
            writer.writerow(["Frame Index", "Track ID", "Linear Velocity", "X Velocity", "Y Velocity", "Angular Velocity"])
        
        # Write data for each track
        for track_id, ekf in tracks.items():
            state = ekf.state
            linear_velocity = np.linalg.norm(state[2:4])  # √(vx^2 + vy^2)
            x_velocity = state[2]
            y_velocity = state[3]
            angular_velocity = state[1]  # Assuming the angular velocity is stored in `state[1]`
            writer.writerow([frame_index, track_id, linear_velocity, x_velocity, y_velocity, angular_velocity])

def save_dbscan_results(labels, valid_indices, frame_index):
    np.save(os.path.join(output_dir, f"dbscan_labels_frame_{frame_index}.npy"), labels)
    np.save(os.path.join(output_dir, f"dbscan_indices_frame_{frame_index}.npy"), valid_indices)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.scatter(valid_indices[:, 1], valid_indices[:, 0], c=labels, cmap='tab20', s=5)
    plt.title(f"DBSCAN Clustering for Frame {frame_index}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(os.path.join(output_dir, f"dbscan_frame_{frame_index}.png"))
    plt.close()

def save_ekf_tracks(tracks, frame_index):
    track_data = {}
    for track_id, ekf in tracks.items():
        track_data[track_id] = ekf.state.tolist()

    with open(os.path.join(output_dir, f"ekf_tracks_frame_{frame_index}.yaml"), 'w') as file:
        yaml.dump(track_data, file)

    # Visualization
    plt.figure(figsize=(10, 10))
    for track_id, ekf in tracks.items():
        state = ekf.state
        plt.plot(state[0], state[1], 'o', label=f"Track {track_id}")
        plt.quiver(
            state[0], state[1], state[2], state[3],
            angles='xy', scale_units='xy', scale=1, label=f"Velocity {track_id}"
        )
    plt.title(f"EKF Tracks for Frame {frame_index}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"ekf_tracks_frame_{frame_index}.png"))
    plt.close()