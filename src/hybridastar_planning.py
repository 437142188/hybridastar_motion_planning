#!/usr/bin/env python
import argparse
import os

import rospy

from utils.hybridastar_planner import HybridastarPlanner


def main(args):
    planner = HybridastarPlanner()
    try:

        if (not os.path.exists(args.path)):
            os.makedirs(args.path)

        while (planner.index <= args.num_files):
            if(planner.get_hybridastar_traj()):
                planner.get_plan(args.path)

    except rospy.ROSInterruptException:
        print("hybridastar planning ros interrupt ")
    except KeyboardInterrupt:
        print( "hybridastar planning keyboard interrupt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../pnet_dataset', help='location of dataset directory')
    parser.add_argument('--num_files', type=int, default=10000, help='num of files')
    parser.add_argument('--step_size', type=float, default=0.01, help='The step size of the point cloud data')
    args = parser.parse_args()
    main(args)

