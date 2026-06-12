import math

from geometry_msgs.msg import Quaternion


DEFAULT_ROBOTS = ('tb3_1', 'tb3_2', 'tb3_3')


def normalize_robot_name(name):
    return name.strip().strip('/')


def namespaced_topic(robot_name, topic):
    robot = normalize_robot_name(robot_name)
    clean_topic = topic.strip('/')
    return f'/{robot}/{clean_topic}'


def yaw_to_quaternion(yaw):
    quat = Quaternion()
    quat.z = math.sin(yaw / 2.0)
    quat.w = math.cos(yaw / 2.0)
    return quat
