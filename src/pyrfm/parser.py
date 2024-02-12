import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import numpy as np
import struct





def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options


def parse_tf(msg):
    """ Parse /tf2 data into a list of Python dictionaries. """
    tf_data = []
    for transform in msg.transforms:
        data = {
            'timestamp': transform.header.stamp.sec,
            'frame_id': transform.header.frame_id,
            'child_frame_id': transform.child_frame_id,
            'translation': {
                'x': transform.transform.translation.x,
                'y': transform.transform.translation.y,
                'z': transform.transform.translation.z
            },
            'rotation': {
                'x': transform.transform.rotation.x,
                'y': transform.transform.rotation.y,
                'z': transform.transform.rotation.z,
                'w': transform.transform.rotation.w
            }
        }
        tf_data.append(data)
    return tf_data


# def parse_PointCloud2(msg):
#     """ Parse /PointCloud2 data into a Python3 list of x,y,z endpoints. """
#     # Parse header information
#     height = msg.height
#     width = msg.width
#     point_step = msg.point_step
#     row_step = msg.row_step

#     # For debugging:
#     # print("height ", height, "\nwidth ", width, "\npoint_step ", point_step, "\nrow_step ", row_step)

#     # Extract X, Y, Z fields
#     x_offset = None
#     y_offset = None
#     z_offset = None

#     for field in msg.fields:
#         if field.name == "x":
#             x_offset = field.offset
#         elif field.name == "y":
#             y_offset = field.offset
#         elif field.name == "z":
#             z_offset = field.offset

#     # Extract end points of lidar rays
#     end_points = []

#     for row in range(height):
#         for col in range(width):
#             index = row * row_step + col * point_step

#             x = struct.unpack_from('f', msg.data, index + x_offset)[0]
#             y = struct.unpack_from('f', msg.data, index + y_offset)[0]
#             z = struct.unpack_from('f', msg.data, index + z_offset)[0]

#             # Calculate end point or perform further processing based on lidar sensor specifics
#             end_point = (x, y, z)
#             end_points.append(end_point)
#     return end_points

def parse_PointCloud2(msg):
    """ Parse /PointCloud2 data into a Python3 list of x, y, z endpoints and their range. """
    # Parse header information
    height = msg.height
    width = msg.width
    point_step = msg.point_step
    row_step = msg.row_step

    # Extract X, Y, Z fields
    x_offset = None
    y_offset = None
    z_offset = None

    for field in msg.fields:
        if field.name == "x":
            x_offset = field.offset
        elif field.name == "y":
            y_offset = field.offset
        elif field.name == "z":
            z_offset = field.offset

    # Extract end points of lidar rays
    end_points = []

    for row in range(height):
        for col in range(width):
            index = row * row_step + col * point_step

            x = struct.unpack_from('f', msg.data, index + x_offset)[0]
            y = struct.unpack_from('f', msg.data, index + y_offset)[0]
            z = struct.unpack_from('f', msg.data, index + z_offset)[0]

            # Calculate range
            range_val = (x**2 + y**2 + z**2)**0.5

            # Append (x, y, z, range) to end_points
            end_point = (x, y, z, range_val)
            end_points.append(end_point)
    return end_points

def to_rays(end_points, tf_data):  
    # Define the dtype for the rays_struct
    rays_struct = np.dtype([
        ('sx', np.float32),
        ('sy', np.float32),
        ('ex', np.float32),
        ('ey', np.float32),
        ('r', np.float32),
    ])

    # Extract necessary data from tf_data
    sx = tf_data["translation"]["x"]
    sy = tf_data["translation"]["y"]
    theta = np.arctan2(2 * (tf_data["rotation"]["w"] * tf_data["rotation"]["z"] + tf_data["rotation"]["x"] * tf_data["rotation"]["y"]),
                       1 - 2 * (tf_data["rotation"]["y"]**2 + tf_data["rotation"]["z"]**2))

    # Initialize rays_data as a list
    rays_data = np.empty(len(end_points), dtype=rays_struct)

    # Iterate over each endpoint
    for i, endpoint in enumerate(end_points):
        ex, ey, ez, r = endpoint

        # Rotate endpoint by theta
        ex_rot = sx + (ex - sx) * np.cos(theta) - (ey - sy) * np.sin(theta)
        ey_rot = sy + (ex - sx) * np.sin(theta) + (ey - sy) * np.cos(theta)

        # Assign data to rays_data for each ray
        rays_data[i] = (sx, sy, ex_rot, ey_rot, r)

    return rays_data


def filter(bag_path_input):
    storage_options, converter_options = get_rosbag_options(bag_path_input)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()  #! Necessary?
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}  #! Necessary?

    only_once = True  # get one slice only
    writeTf = True
    writePointCloud2 = True
    point_cloud_endpoints = []
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])  #! Necessary?
        msg = deserialize_message(data, msg_type)

        if topic == "/tf" and writeTf: 
            tf_data = parse_tf(msg)
            writeTf = False

        if topic == "/ouster/points" and writePointCloud2:  # PointCloud2
            end_points = parse_PointCloud2(msg)
            point_cloud_endpoints.append(end_points)
            writePointCloud2 = False
        
        if only_once and not writeTf and not writePointCloud2:
            break

    # Process point_cloud_endpoints and tf_data here. Save them to .rays file.
    # TODO: make parser for 2d point cloud (pyrfm needs that), figure out why [0] is necessary for endpoints
    rays_array = to_rays(point_cloud_endpoints[0], tf_data[0])
    rays_array.tofile('./out/standing_still-once.rays')
 

if __name__== "__main__":
    # filter("../../data/moving/lidar_tf_tf_static_points_moving_bag_1")
    filter("../../data/standing_still/lidar_tf_tf_static_points_stationary_bag_1")
