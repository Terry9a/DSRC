import argparse
import os
import random
import time
import glob
import numpy as np
import multiprocessing as mp
import copy
from pathlib import Path
from tqdm import tqdm

seed = 1205
random.seed(seed)
np.random.seed(seed)


def get_kitti_ringID(points):
        scan_x = points[:, 0]
        scan_y = points[:, 1]

        yaw = -np.arctan2(scan_y, -scan_x)
        proj_x = 0.5 * (yaw / np.pi + 1.0)
        new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
        proj_y = np.zeros_like(proj_x)
        proj_y[new_raw] = 1
        ringID = np.cumsum(proj_y)
        ringID = np.clip(ringID, 0, 63)
        return ringID

def parse_arguments():

    parser = argparse.ArgumentParser(description='LiDAR cross_sensor')

    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='./data_root/Kitti')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/cross_sensor/light')  # ['light','moderate','heavy']
    parser.add_argument('-n', '--num_beam_to_drop', help='number of beam to be dropped', type=int, default=16)
    arguments = parser.parse_args()

    return arguments



def drop_beams(pointcloud, num_beams_to_drop):
    ring_id = get_kitti_ringID(pointcloud)

    if num_beams_to_drop == 16:
        to_drop = np.arange(1, 64, 4)
    elif num_beams_to_drop == 32:
        to_drop = np.arange(1, 64, 2)
    elif num_beams_to_drop == 48:
        to_drop = np.arange(1, 64, 1.33)
        to_drop = to_drop.astype(int)
    else:
        raise ValueError("Invalid number of beams to drop.")

    to_keep = np.array([i for i in np.arange(64) if i not in to_drop])

    mask_to_keep = np.isin(ring_id, to_keep)
    downsampled_pointcloud = pointcloud[mask_to_keep][::2, :]
    
    return downsampled_pointcloud

def apply_cross_sensor_to_numpy(point_cloud_np):  

    num_beams_to_drop = 32
    processed_pointcloud = drop_beams(point_cloud_np, num_beams_to_drop)
    return processed_pointcloud