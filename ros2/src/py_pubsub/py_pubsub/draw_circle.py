# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# reference: https://www.youtube.com/watch?v=Yy4OgGwEAj8&ab_channel=RoboticsBack-End

import rclpy
from rclpy.node import Node
import time

from geometry_msgs.msg import Twist

class DrawCircleNode(Node):

    def __init__(self):
        super().__init__('draw_circle')
        # velocity topic for Create3 :  cmd_vel
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(2, self.send_velocity_command)
        # can't set two timers?
        # self.timer = self.create_timer(2, self.send_velocity_command2)


    def send_velocity_command(self):
        msg = Twist()
        msg.linear.x = 2.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)
        time.sleep(1)
        msg.linear.x = 0.0
        msg.angular.z = 2.0
        self.publisher_.publish(msg)
        time.sleep(1)
        # self.get_logger().info(msg)

    def send_velocity_command2(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 2.0
        self.publisher_.publish(msg)
        # self.get_logger().info(msg)

def main(args=None):
    rclpy.init(args=args)

    node = DrawCircleNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
