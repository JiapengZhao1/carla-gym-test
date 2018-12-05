import numpy as np
import pygame
import cv2
import sys


def preprocess_image(config, image):
    # This function has input:
    # a dict var which is config
    # , and an image from sensor.listen.
    # Output:
    # Image array.

    # Retrieve data from config
    render_x_res = config["render_x_res"]
    render_y_res = config["render_y_res"]
    x_res = config["x_res"]
    y_res = config["y_res"]
    use_depth_camera = config["use_depth_camera"]

    # Process image based on config data
    if use_depth_camera:
        assert use_depth_camera
        data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        data = np.reshape(data, (render_y_res, render_x_res, 4))
        data = data[:, :, :1]
        data = data[:, :, ::-1]
        data = cv2.resize(
            data, (x_res, y_res), interpolation=cv2.INTER_AREA)
        data = np.expand_dims(data, 2)
    else:
        data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        data = np.reshape(data, (render_y_res, render_x_res, 4))
        data = data[:, :, :3]
        data = data[:, :, ::-1]
        data = cv2.resize(
            data, (x_res, y_res), interpolation=cv2.INTER_AREA)
        data = (data.astype(np.float32) - 128) / 128

    return data


def get_transform_from_nearest_way_point(cur_map, cur_location, dst_location):
    # This function has input:
    # a carla.Map var,
    # a carla.Location var which is current car location,
    # a carla.Location var which is the destination location.
    # Output:
    # The carla.Transform var which from the nearest way_point to the destination location.

    # Get next possible way_points
    way_points = cur_map.get_waypoint(cur_location)
    nexts = list(way_points.next(1.0))
    print('Next(1.0) --> %d waypoints' % len(nexts))
    if not nexts:
        raise RuntimeError("No more waypoints!")

    # Calculate the way_point which is nearest to the dst_location
    smallest_dist = sys.maxsize
    for p in nexts:
        loc = p.transform.location
        diff_x = loc.x - dst_location.x
        diff_y = loc.y - dst_location.y
        diff_z = loc.z - dst_location.z
        cur_dist = np.linalg.norm([diff_x, diff_y, diff_z])
        if cur_dist < smallest_dist:
            next_point = p
    text = "road id = %d, lane id = %d"
    print(text % (next_point.road_id, next_point.lane_id))

    # debugger = self.client.get_world().debug
    # debugger.draw_point(next_point.transform.location, size=0.1, color=carla.Color(), life_time=-1.0, persistent_lines=True)

    return next_point.transform