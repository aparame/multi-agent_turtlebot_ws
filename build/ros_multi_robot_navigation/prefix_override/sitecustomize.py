import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/adi2440/turtlebot_ws/install/ros_multi_robot_navigation'
