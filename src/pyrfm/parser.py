import rosbag2_py
import rclpy
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message, serialize_message

import struct

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def parse_tf(msg):
    start_point = (0, 0, 0)
    print("x ", msg.transforms[0].transform.translation.x)
    return start_point
    pass


def parse_PointCloud2(msg):
    # Parse header information
    height = msg.height
    width = msg.width
    point_step = msg.point_step
    row_step = msg.row_step
    print("height ", height, "\nwidth ", width, "\npoint_step ", point_step, "\nrow_step ", row_step)

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

            # Calculate end point or perform further processing based on lidar sensor specifics
            end_point = (x, y, z)
            end_points.append(end_point)
    return end_points


def filter(bag_path_input):

    storage_options, converter_options = get_rosbag_options(bag_path_input)

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    # writer = rosbag2_py.SequentialWriter()
    # writer.open(storage_options2,converter_options2)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    i = 0
    writeTf = True
    writePointCloud2 = True
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)

        if type_map[topic] == "tf2_msgs/msg/TFMessage" and writeTf:
            try:
                parse_tf(msg)
            except:
                print("error ")
            # writeTf = False
            
            # sx = str(msg.transforms[0].transform.translation.x)
            # sy = str(msg.transforms[0].transform.translation.y)
            pass


        if type_map[topic] == "sensor_msgs/msg/PointCloud2" and writePointCloud2:
            end_points = parse_PointCloud2(msg)
            for point in end_points:
                # print(point)  # if zero3, either barrier hit or inf
                # closer to 30cm to the lidar -> blocked (zero3)
                # farther than 50m -> inf (zero3)
                pass

            print(len(end_points))
            writePointCloud2 = False

        # if topic=="/tf":    
        #     if not ((msg.transforms[0].header.frame_id == "map" and msg.transforms[0].child_frame_id == "odom")):
        #         writer.write(topic,serialize_message(msg),t)
        # else:
        #     writer.write(topic,serialize_message(msg),t)

if __name__== "__main__":
    filter("../../data/standing_still/lidar_tf_tf_static_points_stationary_bag_1")
