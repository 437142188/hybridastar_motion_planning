#!/usr/bin/env python
import numpy as np
import os
import sys
import random
import math

import geometry_msgs.msg
import std_msgs.msg
import rospy
import hybridastar_service.srv


class HybridastarPlanner(object):
    """
    hybrid astar class
    """

    def __init__(self):

        super(HybridastarPlanner, self).__init__()  # initialize the object

        # initialize the Node
        rospy.init_node('hybridastar_planning', anonymous=True, disable_signals=True)

        service_name = '/run_astar_opt'
        rospy.wait_for_service(service_name)
        self.get_hybridastar = rospy.ServiceProxy(service_name, hybridastar_service.srv.hybrid_astar_serviceMsg)

        self.traj_goal = []
        self.traj_value = []
        self.width = 5.0
        self.height = 5.0
        self.index = 1

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


    def get_hybridastar_traj(self):
        call_hybridastar_request = hybridastar_service.srv.hybrid_astar_serviceMsgRequest()
        call_hybridastar_request.start = self.get_random_pose()
        call_hybridastar_request.goal = self.get_random_pose()
        call_hybridastar_request.vehicle_state = geometry_msgs.msg.Twist()
        # call_hybridastar_request.obstacles = 
        
        bound = std_msgs.msg.Float32MultiArray()
        bound.data.append(-self.width)
        bound.data.append(self.width)
        bound.data.append(-self.height)
        bound.data.append(self.height)
        bound.data.append(1.0)
        bound.data.append(-1.0)
        bound.data.append(0.4)
        call_hybridastar_request.bounds = bound

        try:

            result = self.get_hybridastar(call_hybridastar_request)

            for i in range(len(result.plan)):
                r,p,y = self.euler_from_quaternion(result.plan[i].pose)
                self.traj_value.append([result.plan[i].pose.position.x, result.plan[i].pose.position.y, y])

            r,p,y = self.euler_from_quaternion(call_hybridastar_request.goal.pose)
            self.traj_goal = [call_hybridastar_request.goal.pose.position.x, call_hybridastar_request.goal.pose.position.y, y]
            return True

        except rospy.ServiceException as e:
            print( "Service call failed: %s"%e)
            return False

    def get_random_pose(self):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = '/map'
        pos = geometry_msgs.msg.Point(random.uniform(-self.width,self.width), random.uniform(-self.height,self.height), 0.0)
        ori = self.quaternion_from_euler(random.uniform(-math.pi,math.pi))
        start_pose = geometry_msgs.msg.PoseStamped(header, geometry_msgs.msg.Pose(pos, geometry_msgs.msg.Quaternion(*ori)))
        
        return start_pose


    def get_plan(self, path):
        if (len(self.traj_value)):
            for i in range(len(self.traj_value) - 1):
                temp = np.array([self.traj_value[i], self.traj_goal]).flatten()
                np.save(os.path.join(path, 'states' + str(self.index) + '.npy'), temp)
                np.save(os.path.join(path, 'plan' + str(self.index) + '.npy'), self.traj_value[i + 1])
                self.index += 1
            temp = np.array([self.traj_value[-1], self.traj_goal]).flatten()
            np.save(os.path.join(path, 'states' + str(self.index) + '.npy'), temp)
            np.save(os.path.join(path, 'plan' + str(self.index) + '.npy'), self.traj_value[-1])
            self.index += 1
