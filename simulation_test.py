import glob
import os
import sys
import random
import time
import cv2
import numpy as np
import open3d as o3d
from datetime import datetime
import argparse

# Handle the CARLA Python API .egg file
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("CARLA .egg file not found!")
    sys.exit(1)

import carla


def save_point_cloud(frame, point_list, save_path):
    """Save point cloud to the specified folder."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, f"lidar_frame_{frame}.pcd")
    o3d.io.write_point_cloud(filename, point_list)
    print(f"Saved LiDAR frame {frame} to {filename}")


def lidar_callback(point_cloud, point_list):
    """Process LiDAR data."""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    points = data[:, :-1]
    points[:, :1] = -points[:, :1]  # Adjust Y-coordinate for Open3D compatibility

    point_list.points = o3d.utility.Vector3dVector(points)
    
def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '0.2')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp

def main(arg):
    """Main function of the script."""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        # Set simulation settings
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = arg.no_rendering 
        world.apply_settings(settings)

        # Set up traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # Spawn the ego vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter(arg.filter)[0]
        spawn_points = world.get_map().get_spawn_points()

        # Select a spawn point away from traffic lights
        ego_spawn_point = spawn_points[5]  # Adjust spawn index if needed
        ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn_point)

        # Set constant velocity for the ego vehicle
        #ego_velocity = carla.Vector3D(10.0, 0.0, 0.0)  # Move forward at 10 m/s
        #ego_vehicle.enable_constant_velocity(ego_velocity)

        # Enable autopilot for the ego vehicle
        ego_vehicle.set_autopilot(True)

        # Configure traffic manager to handle ego vehicle behavior
        traffic_manager.ignore_lights_percentage(ego_vehicle, 0)  # Follow traffic lights
        traffic_manager.ignore_walkers_percentage(ego_vehicle, 0)  # Avoid pedestrians
        traffic_manager.ignore_vehicles_percentage(ego_vehicle, 0)  # Avoid other vehicles
        traffic_manager.auto_lane_change(ego_vehicle, True)  # Allow lane changes


        # Attach LiDAR sensor to the ego vehicle
        lidar_bp = generate_lidar_bp(arg, world, blueprint_library, settings.fixed_delta_seconds)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)

        point_list = o3d.geometry.PointCloud()
        lidar_save_path = r"C:\Users\kodur\CARLA_0.9.12\WindowsNoEditor\PythonAPI\examples\PCD"
        lidar.listen(lambda data: lidar_callback(data, point_list))

        # Attach a camera to the ego vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

        # Video writer setup
        video_save_path = r"C:\Users\kodur\CARLA_0.9.12\WindowsNoEditor\PythonAPI\examples\ego_vehicle_perspective.avi"
        video_writer = cv2.VideoWriter(
            video_save_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1920, 1080)
        )

        def camera_callback(image):
            """Process camera frames and save to video."""
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))  # BGRA format
            array = array[:, :, :3]  # Drop alpha channel
            video_writer.write(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

        camera_sensor.listen(camera_callback)

        # Spawn additional vehicles
        other_vehicles = []
        selected_spawn_points = [spawn_points[10], spawn_points[15], spawn_points[20]]
        for spawn_point in selected_spawn_points:
            other_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            other_vehicle = world.spawn_actor(other_vehicle_bp, spawn_point)
            other_vehicle.set_autopilot(True)
            other_vehicles.append(other_vehicle)

        # Simulation loop
        frame = 0
        dt0 = datetime.now()
        while frame < 100000:  # Run for 100000 frames
            world.tick()

            # Save LiDAR data every 50 frames
            if frame % 50 == 0:
                if not os.path.exists(lidar_save_path):
                    os.makedirs(lidar_save_path)
                filename = os.path.join(lidar_save_path, f"lidar_frame_{frame}.pcd")
                o3d.io.write_point_cloud(filename, point_list)
                print(f"Saved frame {frame} to {filename}")

            # Update frame count and processing time
            process_time = datetime.now() - dt0
            sys.stdout.write(f'\rFPS: {1.0 / process_time.total_seconds():.2f}')
            sys.stdout.flush()
            dt0 = datetime.now()
            frame += 1

    finally:
        # Restore settings and clean up
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        lidar.destroy()
        camera_sensor.destroy()
        ego_vehicle.destroy()
        for vehicle in other_vehicles:
            vehicle.destroy()

        video_writer.release()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__
    )
    argparser.add_argument('--host', default='localhost', help='Host IP')
    argparser.add_argument('-p', '--port', type=int, default=2000, help='Port')
    argparser.add_argument('--no-rendering', action='store_true', help='Disable rendering')
    argparser.add_argument('--filter', default='model3', help='Vehicle filter')
    argparser.add_argument('--no-autopilot', action='store_false', help='Disable autopilot')
    argparser.add_argument('--semantic', action='store_true', help='Use semantic LiDAR')
    argparser.add_argument('--no-noise', action='store_true', help='Disable LiDAR noise')
    argparser.add_argument('--upper-fov', type=float, default=15.0, help='Upper field of view (degrees)')
    argparser.add_argument('--lower-fov', type=float, default=-25.0, help='Lower field of view (degrees)')
    argparser.add_argument('--channels', type=int, default=64, help='Number of LiDAR channels')
    argparser.add_argument('--range', type=float, default=100.0, help='LiDAR range (meters)')
    argparser.add_argument('--points-per-second', type=int, default=500000, help='LiDAR points per second')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print('Simulation interrupted by user.')

