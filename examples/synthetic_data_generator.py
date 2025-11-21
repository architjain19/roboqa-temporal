"""

################################################################

File: examples/synthetic_data_generator.py
Created: 2025-11-20
Created by: Archit Jain (architj@u.edu)
Last Modified: 2025-11-20
Last Modified by: Archit Jain (architj@uw.edu)

#################################################################

Copyright: RoboQA-Temporal Authors
License: MIT License

################################################################

Synthetic data generator for testing RoboQA-Temporal.
This script publishes synthetic point cloud data to a ROS2 topic.

################################################################

"""

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import rclpy
from rclpy.node import Node
import numpy as np

class SyntheticPCPublisher(Node):
    """
    A ROS2 node that publishes synthetic point cloud data.
    """
    def __init__(self):
        """
        Initialize the SyntheticPCPublisher node.
        """
        super().__init__('synthetic_pc_pub')

        # Publisher for synthetic point cloud
        self.pub = self.create_publisher(PointCloud2, '/synthetic_points', 10)
        # Timer to publish at 1 Hz
        self.timer = self.create_timer(1.0, self.timer_cb)  # 1 Hz

    def timer_cb(self):
        """
        Timer callback to publish synthetic point cloud data.
        """
        N = 200  # points per frame
        pts = (np.random.uniform(-5.0, 5.0, (N, 3))).astype(float).tolist()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        msg = point_cloud2.create_cloud_xyz32(header, pts)
        self.pub.publish(msg)
        self.get_logger().info(f'Published {N} synthetic points')

def main():
    """
    Main function to run the SyntheticPCPublisher node.
    """
    rclpy.init()
    node = SyntheticPCPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
