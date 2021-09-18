#!/usr/bin/env python
import numpy as np
import os
import sys
import random
import math
import copy

import geometry_msgs.msg
import std_msgs.msg
import rospy
import torch

from utils.pnet import PNet


class NeuralHybirdAstarPlanner(object):
    """
    Neural Motion Planner.
    """

    def __init__(self):
        """
        Constructor for NeuralHybirdAstarPlanner
        """
        super(NeuralHybirdAstarPlanner, self).__init__()  # initialize the object

        # initialize the Node
        rospy.init_node('Neural_Planner', anonymous=True, disable_signals=True)
        self.display_trajectory_publisher = rospy.Publisher('/neural_hybridastar_planner/display_planned_path',
                                                            geometry_msgs.msg.PoseArray,
                                                            queue_size=20)
        self.pnet = PNet(6, 3)
        self.pnet.load_state_dict(torch.load(os.path.join(os.curdir, 'pnet_weights.pt')))
        self.traj_goal = []
        self.traj_value = []
        self.width = 5.0
        self.height = 5.0

    def euler_from_quaternion(self, pose):
        x = pose.orientation.x 
        y = pose.orientation.y 
        z = pose.orientation.z 
        w = pose.orientation.w 
        Roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2+y**2))
        Pitch = np.arcsin(2 * (w * y - z * x))
        Yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return [Roll, Pitch, Yaw] # [r p y]

    def quaternion_from_euler(self, y, p=0, r=0):
        q3 = np.cos(r / 2) * np.cos(p / 2) * np.cos(y / 2) + \
            np.sin(r / 2) * np.sin(p / 2) * np.sin(y / 2)
        q0 = np.sin(r / 2) * np.cos(p / 2) * np.cos(y / 2) - \
            np.cos(r / 2) * np.sin(p / 2) * np.sin(y / 2)
        q1 = np.cos(r / 2) * np.sin(p / 2) * np.cos(y / 2) + \
            np.sin(r / 2) * np.cos(p / 2) * np.sin(y / 2)
        q2 = np.cos(r / 2) * np.cos(p / 2) * np.sin(y / 2) - \
            np.sin(r / 2) * np.sin(p / 2) * np.cos(y / 2)
        return [q0, q1, q2, q3] # [x y z w]

    def all_close(self,goal, actual, tolerance):
        """
        used to check if the current eff position is close to the goal position.
        :param goal:
        :param actual:
        :param tolerance:
        :return type bool:
        """
        r,p,y = self.euler_from_quaternion(goal.pose)
        goal_list = []
        goal_list.append(goal.pose.position.x)
        goal_list.append(goal.pose.position.y)
        goal_list.append(y)

        r,p,y = self.euler_from_quaternion(actual.pose)
        actual_list = []
        actual_list.append(actual.pose.position.x)
        actual_list.append(actual.pose.position.y)
        actual_list.append(y)

        for index in range(len(goal_list)):
            if abs(actual_list[index] - goal_list[index]) > tolerance:
                    return False

        return True

    def get_random_pose(self):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = '/map'
        pos = geometry_msgs.msg.Point(random.uniform(-self.width,self.width), random.uniform(-self.height,self.height), 0.0)
        ori = self.quaternion_from_euler(random.uniform(-math.pi,math.pi))
        start_pose = geometry_msgs.msg.PoseStamped(header, geometry_msgs.msg.Pose(pos, geometry_msgs.msg.Quaternion(*ori)))
        
        return start_pose

    def plan_path(self, start_states,goal_states,next_states):
        r,p,y = self.euler_from_quaternion(start_states.pose)
        current_list = []
        current_list.append(start_states.pose.position.x / self.width)
        current_list.append(start_states.pose.position.y / self.height)
        current_list.append(y / math.pi)

        r,p,y = self.euler_from_quaternion(goal_states.pose)
        goal_list = []
        goal_list.append(goal_states.pose.position.x / self.width)
        goal_list.append(goal_states.pose.position.y / self.height)
        goal_list.append(y / math.pi)

        # network inference
        states = torch.from_numpy(np.array([current_list, goal_list]).flatten()).float()
        plan = self.pnet(states)

        
        
        next_states.pose.position.x = plan[0].item() * self.width
        next_states.pose.position.y = plan[1].item() * self.height
        # print('start_states:{0:3f},{1:3f}'.format(current_list[0],current_list[1]))
        # print('next_states:{0:3f},{1:3f}'.format(plan[0].item(),plan[1].item()))

        ori = self.quaternion_from_euler(plan[2].item() * math.pi)
        next_states.pose.orientation = geometry_msgs.msg.Quaternion(*ori)

        return self.all_close(goal_states, next_states, 0.2)


def main():
    planner = NeuralHybirdAstarPlanner()

    #set start pose
    start_pose = planner.get_random_pose()
    status = []
    status.append(start_pose)
    print('start_pose:{0:3f},{1:3f}'.format(start_pose.pose.position.x,start_pose.pose.position.y))

    #set goal pose
    goal_pose = planner.get_random_pose()
    print('goal_pose:{0:3f},{1:3f}'.format(goal_pose.pose.position.x,goal_pose.pose.position.y))

    next_pose = geometry_msgs.msg.PoseStamped()
    next_pose.header.stamp = rospy.Time.now()
    next_pose.header.frame_id = '/map'

    count = 0
    while (not planner.plan_path(status[-1],goal_pose,next_pose) and count < 100):
        print('current:{0:3f},{1:3f},{2:3f}  next:{3:3f},{4:3f},{5:3f}'.format(status[-1].pose.position.x, status[-1].pose.position.y,
                                                        0,
                                                       next_pose.pose.position.x, next_pose.pose.position.y,0))
        status.append(copy.deepcopy(next_pose))
        count = count + 1
        continue
    print("NeuralHybirdAstarPlanner successful")

    path = geometry_msgs.msg.PoseArray()
    path.header.stamp = rospy.Time.now()
    path.header.frame_id = '/map'
    path.poses = []
    for index in range(len(status)):
        path.poses.append(status[index].pose)
    planner.display_trajectory_publisher.publish(path)



if __name__ == '__main__':
    main()
