"""
This script sets up a CARLA simulation where an ego vehicle follows a target motorcycle while maintaining a safe distance. The ego vehicle is equipped with LiDAR, RGB Camera,
and Collision Sensor to collect data and respond to the environment. It logs sensor outputs, velocities, and angular velocities, while dynamically adjusting throttle and steering
to ensure smooth control. The simulation supports bird's-eye visualization and saves outputs for post-processing. Key features include configurable sensor parameters, real-time collision handling,
and modular control logic. The script is ideal for autonomous driving experiments with focus on perception, control, and data logging.
"""


import glob
import os
import sys
import random
import numpy as np
from datetime import datetime
import argparse
import open3d as o3d
import cv2
import math

# Handle the CARLA Python API .egg file
# Attempt to dynamically append the CARLA .egg file path based on system architecture and Python version.
# This ensures the CARLA library can be imported without manually specifying the path.
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("CARLA .egg file not found!")
    sys.exit(1)

import carla

"""
The `attach_lidar_to_vehicle` function initializes and attaches a LiDAR sensor to a specified vehicle in the CARLA simulation world.

1. **Blueprint Retrieval**:
   - Retrieves the `ray_cast` LiDAR sensor blueprint from the world’s blueprint library.

2. **LiDAR Configuration**:
   - Configures essential LiDAR parameters:
     - **range**: Specifies the maximum distance (100 meters) for point detection.
     - **channels**: Sets the number of vertical beams to 32, ensuring adequate vertical coverage.
     - **points_per_second**: Increased to 1,000,000 points per second for higher resolution scans.
     - **rotation_frequency**: Adjusted to 30 Hz for smoother and more frequent scans.
     - **upper_fov** and **lower_fov**: Define the vertical field of view between 15° (up) and -30° (down).

3. **Attachment**:
   - The LiDAR sensor is attached to the vehicle using a fixed transformation:
     - **Position**: `x=0.0`, `z=2.5` meters (directly above the vehicle center).
   - The `world.spawn_actor` function creates the LiDAR actor and attaches it to the specified vehicle.

4. **Return**:
   - Returns the spawned LiDAR actor for further use, such as data collection and callbacks.
"""

def attach_lidar_to_vehicle(world, vehicle):
    """Attach a LiDAR sensor to the vehicle."""
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    # Configure LiDAR attributes
    lidar_bp.set_attribute('range', '100.0')
    lidar_bp.set_attribute('channels', '32')
    #changed from 50000 to 1000000
    lidar_bp.set_attribute('points_per_second', '1000000')
    #changed from 10 to 30 
    lidar_bp.set_attribute('rotation_frequency', '30.0')
    lidar_bp.set_attribute('upper_fov', '15.0')
    lidar_bp.set_attribute('lower_fov', '-30.0')

    # Attach LiDAR to the vehicle
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    return lidar


def attach_camera_to_vehicle(world, vehicle):
    """Attach a camera sensor to the vehicle."""
    
    # Retrieve the blueprint library from the CARLA simulation world
    blueprint_library = world.get_blueprint_library()
    
    # Find the RGB camera sensor blueprint
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    
    # Configure camera attributes
    camera_bp.set_attribute('image_size_x', '1920')  # Set horizontal resolution to 1920 pixels
    camera_bp.set_attribute('image_size_y', '1080')  # Set vertical resolution to 1080 pixels
    camera_bp.set_attribute('fov', '110')  # Set field of view (FOV) to 110 degrees for wide-angle capture

    # Define the transformation for attaching the camera to the vehicle
    # The camera is positioned 1.5 meters forward and 2.4 meters above the vehicle's center
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    
    # Spawn the camera actor and attach it to the specified vehicle
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    
    # Return the spawned camera actor for further operations (e.g., data collection, visualization)
    return camera

def set_birds_eye_view(world, ego_vehicle):
    """Set the spectator view for a bird's-eye view above the ego vehicle."""
    
    # Get the spectator object in the simulation, which represents the camera used for rendering
    spectator = world.get_spectator()
    # Get the current transform (position and orientation) of the ego vehicle
    vehicle_transform = ego_vehicle.get_transform()

    # Define the spectator's position directly above the ego vehicle
    # The spectator's height is set 50 meters above the vehicle
    spectator_location = carla.Location(
        x=vehicle_transform.location.x,  # Match the vehicle's x position
        y=vehicle_transform.location.y,  # Match the vehicle's y position
        z=vehicle_transform.location.z + 50.0  # Add 50 meters to the z position
    )

    # Define the spectator's orientation
    # The pitch is set to -90 degrees to look straight down, 
    # and yaw is aligned with the vehicle's heading
    spectator_rotation = carla.Rotation(
        pitch=-90, 
        yaw=vehicle_transform.rotation.yaw, 
        roll=0
    )

    # Set the spectator's transform to the newly defined location and rotation
    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))


def compute_control(ego_location, target_location, ego_rotation, safe_distance=5.0, stop_distance=2.0):
    """Compute throttle and steering to follow the target while maintaining safe distance."""
    dx = target_location.x - ego_location.x
    dy = target_location.y - ego_location.y
    distance = math.sqrt(dx**2 + dy**2)
    
    # Desired yaw to face the target
    desired_yaw = math.degrees(math.atan2(dy, dx))
    
    # Compute the yaw error
    yaw_error = desired_yaw - ego_rotation.yaw
    yaw_error = (yaw_error + 180) % 360 - 180  # Normalize to [-180, 180]
    
    # Throttle logic with safe and stop distance
    if distance < stop_distance:
        throttle = 0.0  # Stop completely
    elif distance < safe_distance:
        throttle = min(0.5, (distance - stop_distance) * 0.1)  # Slow down as it gets closer
    else:
        throttle = min(1.0, distance * 0.1)  # Proportional to distance
    
    # Steering logic
    steering = max(-1.0, min(1.0, yaw_error * 0.05))  # Proportional to yaw error
    
    return throttle, steering


def compute_control(ego_location, target_location, ego_rotation, safe_distance=5.0, stop_distance=2.0):
    """Compute throttle and steering to follow the target while maintaining a safe distance."""

    # Calculate the difference in x and y coordinates between the ego and target locations
    dx = target_location.x - ego_location.x
    dy = target_location.y - ego_location.y

    # Compute the Euclidean distance between the ego vehicle and the target
    distance = math.sqrt(dx**2 + dy**2)

    # Calculate the desired yaw angle (in degrees) for the ego vehicle to face the target
    desired_yaw = math.degrees(math.atan2(dy, dx))

    # Compute the yaw error between the desired yaw and the ego vehicle's current yaw
    yaw_error = desired_yaw - ego_rotation.yaw
    # Normalize the yaw error to the range [-180, 180] to handle angle wrapping
    yaw_error = (yaw_error + 180) % 360 - 180

    # Determine throttle based on the distance to the target
    if distance < stop_distance:
        throttle = 0.0  # Stop the vehicle if it is within the stopping distance
    elif distance < safe_distance:
        # Gradually reduce speed as the ego vehicle approaches the safe distance
        throttle = min(0.5, (distance - stop_distance) * 0.1)
    else:
        # Maintain proportional throttle based on distance when beyond the safe distance
        throttle = min(1.0, distance * 0.1)

    # Determine steering angle based on the yaw error
    # Clamp the steering value between -1.0 (left) and 1.0 (right)
    steering = max(-1.0, min(1.0, yaw_error * 0.05))  # Proportional to the yaw error

    # Return the computed throttle and steering values
    return throttle, steering




def main(arg):
    """Main function of the script."""
    client = carla.Client(arg.host, arg.port)
    client.set_timeout(10.0)
    client.load_world('Town02')
    world = client.get_world()

    try:
        # Set simulation settings
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        #changed from 0.05 to 0.033
        settings.fixed_delta_seconds = 0.033
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        # Set up traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # Get blueprint library and spawn points
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()

        # Spawn the ego vehicle
        vehicle_bp = blueprint_library.filter(arg.filter)[0]
        ego_spawn_point = spawn_points[1]  # Select a spawn point
        ego_vehicle = world.spawn_actor(vehicle_bp, ego_spawn_point)

                
        print("Ego vehicle spawned")

        # Attach LiDAR and Camera to the ego vehicle
        lidar = attach_lidar_to_vehicle(world, ego_vehicle)
        camera = attach_camera_to_vehicle(world, ego_vehicle)

        # Attach Collision Sensor to the ego vehicle
        collision_sensor = attach_collision_sensor(world, ego_vehicle)
        print("collison sensor detected")

        # Create folders for saving sensor outputs
        lidar_save_path = os.path.join(os.getcwd(), 'pcd')
        camera_save_path = os.path.join(os.getcwd(), 'images')
        velocity_save_path = r'C:\Users\kodur\CARLA_0.9.12\WindowsNoEditor\PythonAPI\examples\Velocity'
        os.makedirs(lidar_save_path, exist_ok=True)
        os.makedirs(camera_save_path, exist_ok=True)
        os.makedirs(velocity_save_path, exist_ok=True)

        # Open files for saving velocities
        target_velocity_file = open(os.path.join(velocity_save_path, 'target_velocity.txt'), 'w')
        ego_velocity_file = open(os.path.join(velocity_save_path, 'ego_velocity.txt'), 'w')
        # Create paths for saving angular velocities
        ego_angular_velocity_file = open(os.path.join(velocity_save_path, 'ego_angular_velocity.txt'), 'w')
        target_angular_velocity_file = open(os.path.join(velocity_save_path, 'target_angular_velocity.txt'), 'w')


        # Define callbacks for LiDAR and Camera
        point_list = o3d.geometry.PointCloud()
        camera_frames = []

        def lidar_callback(lidar_data):
            """Handle LiDAR data."""
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
            point_list.points = o3d.utility.Vector3dVector(points)

        def camera_callback(image):
            """Handle Camera data."""
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            array = array[:, :, :3]#[:, :, ::-1]  # Convert BGRA to BGR
            camera_frames.append(array)

        lidar.listen(lidar_callback)
        camera.listen(camera_callback)

        # Spawn the target vehicle 10 meters ahead of the ego vehicle
        target_spawn_point = carla.Transform(
            carla.Location(
                x=ego_spawn_point.location.x,
                y=ego_spawn_point.location.y + 10,
                z=ego_spawn_point.location.z
            ),
            ego_spawn_point.rotation
        )
        target_vehicle_bp = blueprint_library.find('vehicle.bh.crossbike')
        target_vehicle = world.spawn_actor(target_vehicle_bp, target_spawn_point)
        print('Target vehicle spawned')

        # Configure the vehicles
        #target_velocity = carla.Vector3D(4.0, 0.0, 0.0)  # Target moves at 4 m/s
        #target_vehicle.enable_constant_velocity(target_velocity)
        

        ego_vehicle.set_autopilot(True, traffic_manager.get_port())
        target_vehicle.set_autopilot(True, traffic_manager.get_port())
        velocity = target_vehicle.get_velocity()
        print(f"Target vehicle velocity: x={velocity.x:.2f}, y={velocity.y:.2f}, z={velocity.z:.2f}")


        # Simulation loop for 5000 frames
        total_frames = 5000
        frame = 0
        dt0 = datetime.now()
        while frame < total_frames:
            world.tick()

            # Update bird's-eye view periodically to follow the ego vehicle
            if frame % 10 == 0:  # Update every 10 frames
                set_birds_eye_view(world, ego_vehicle)

            # Save outputs every 30 frames
            if frame % 30 == 0:
                # Save LiDAR PCD file
                lidar_filename = os.path.join(lidar_save_path, f"lidar_frame_{frame}.pcd")
                o3d.io.write_point_cloud(lidar_filename, point_list)
                print(f"Saved LiDAR frame {frame} to {lidar_filename}")

                # Save Camera JPEG file
                if camera_frames:
                    image_array = camera_frames.pop(0)
                    camera_filename = os.path.join(camera_save_path, f"image_frame_{frame}.jpeg")
                    cv2.imwrite(camera_filename, image_array)
                    print(f"Saved Camera frame {frame} to {camera_filename}")

                # Log velocities
                target_velocity = target_vehicle.get_velocity()
                ego_velocity = ego_vehicle.get_velocity()
                target_velocity_file.write(
                    f"Frame {frame}: x={target_velocity.x:.2f}, y={target_velocity.y:.2f}, z={target_velocity.z:.2f}\n"
                )
                ego_velocity_file.write(
                    f"Frame {frame}: x={ego_velocity.x:.2f}, y={ego_velocity.y:.2f}, z={ego_velocity.z:.2f}\n"
                )
                print(f"Logged velocities at frame {frame}")


            # Compute control inputs for the ego vehicle
                ego_location = ego_vehicle.get_location()
                target_location = target_vehicle.get_location()
                ego_rotation = ego_vehicle.get_transform().rotation
                throttle, steering = compute_control(ego_location, target_location, ego_rotation, safe_distance=7.0, stop_distance=3.0)

                # Gradual deceleration logic
                current_throttle = control.throttle if 'control' in locals() else 0.0  # Use existing throttle
                desired_throttle = throttle
                smoothed_throttle = current_throttle + (desired_throttle - current_throttle) * 0.1  # Smooth change

                # Apply control to the ego vehicle
                control = carla.VehicleControl()
                control.throttle = smoothed_throttle
                control.steer = steering
                ego_vehicle.apply_control(control)


            # Apply control to the ego vehicle
            ego_vehicle.apply_control(control)


            # Calculate and display FPS
            process_time = datetime.now() - dt0
            sys.stdout.write(f'\rFrame {frame}/{total_frames}, FPS: {1.0 / process_time.total_seconds():.2f}')
            sys.stdout.flush()
            dt0 = datetime.now()

            frame += 1


    finally:
        # Restore settings and clean up
        world.apply_settings(original_settings)
        lidar.destroy()
        camera.destroy()
        ego_vehicle.destroy()
        target_vehicle.destroy()
        traffic_manager.set_synchronous_mode(False)
        target_velocity_file.close()
        ego_velocity_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__
    )
    argparser.add_argument('--host', default='localhost', help='Host IP')
    argparser.add_argument('-p', '--port', type=int, default=2000, help='Port')
    argparser.add_argument('--no-rendering', action='store_true', help='Disable rendering')
    argparser.add_argument('--filter', default='model3', help='Vehicle filter')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print('Simulation interrupted by user.')
