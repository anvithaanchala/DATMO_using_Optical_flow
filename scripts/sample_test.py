import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import os
import yaml
from utils import *

# Load parameters from YAML
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Multi-plane detection
def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    plane_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        w, index = PlaneRegression(
            target, threshold=threshold, init_n=3, iter=iterations)
        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)
    return plane_list

# BEV grid computation
def compute_bev_grid(points, grid_resolution, x_range, y_range, a=0.5, b=0.5, h_max=5.0):
    w, h = grid_resolution
    x_bins = np.arange(x_range[0], x_range[1], w)
    y_bins = np.arange(y_range[0], y_range[1], h)
    bev_values = np.zeros((len(x_bins), len(y_bins)))

    for x, y, z in points:
        x_idx = int((x - x_range[0]) / w)
        y_idx = int((y - y_range[0]) / h)
        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            bev_values[x_idx, y_idx] += (a * z + b * np.std(z)) / h_max
    return bev_values

# Compute velocity using optical flow
def compute_velocity_vectors(bev1, bev2, x_range, y_range):
    farneback_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0,
    }
    flow = cv2.calcOpticalFlowFarneback(bev1.astype(np.float32), bev2.astype(np.float32), None, **farneback_params)
    vx, vy = flow[..., 0], flow[..., 1]

    pixel_size_x = (x_range[1] - x_range[0]) / bev1.shape[1]
    pixel_size_y = (y_range[1] - y_range[0]) / bev1.shape[0]
    velocity_x = vx * pixel_size_x
    velocity_y = vy * pixel_size_y

    return velocity_x, velocity_y

# Apply propagation and continuity masks
def apply_masks(vx, vy, dt, alpha_cont):
    grid_shape = vx.shape
    vx_prop, vy_prop = np.zeros_like(vx), np.zeros_like(vy)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            x_new = i + int(vx[i, j] * dt)
            y_new = j + int(vy[i, j] * dt)
            if 0 <= x_new < grid_shape[0] and 0 <= y_new < grid_shape[1]:
                vx_prop[x_new, y_new] = vx[i, j]
                vy_prop[x_new, y_new] = vy[i, j]

    div_v = np.gradient(vx_prop, axis=1) + np.gradient(vy_prop, axis=0)
    curl_v = np.gradient(vy_prop, axis=1) - np.gradient(vx_prop, axis=0)
    mask = (np.abs(div_v) <= alpha_cont) & (np.abs(curl_v) <= alpha_cont)
    return vx_prop * mask, vy_prop * mask, mask

# Main pipeline
def process_and_visualize(pcd_file1, pcd_file2, config):
    grid_resolution = config['grid_resolution']
    x_range = config['x_range']
    y_range = config['y_range']
    z_max = config['z_max']
    roi_bounds = config['roi_bounds']
    dt = config['dt']
    alpha_cont = config['masks']['alpha_cont'][0]

    # Process PCD files
    points1 = np.asarray(o3d.io.read_point_cloud(pcd_file1).points)
    points2 = np.asarray(o3d.io.read_point_cloud(pcd_file2).points)

    # Detect planes and generate BEV
    planes1 = DetectMultiPlanes(points1)
    planes2 = DetectMultiPlanes(points2)

    bev1 = compute_bev_grid(points1, grid_resolution, x_range, y_range, h_max=z_max)
    bev2 = compute_bev_grid(points2, grid_resolution, x_range, y_range, h_max=z_max)

    velocity_x, velocity_y = compute_velocity_vectors(bev1, bev2, x_range, y_range)
    vx_prop, vy_prop, mask = apply_masks(velocity_x, velocity_y, dt, alpha_cont)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.quiver(np.arange(x_range[0], x_range[1], grid_resolution[0]),
               np.arange(y_range[0], y_range[1], grid_resolution[1]),
               vx_prop, vy_prop, angles='xy', scale_units='xy', scale=1, color='blue')
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

    process_and_visualize(pcd_file1, pcd_file2, config)
