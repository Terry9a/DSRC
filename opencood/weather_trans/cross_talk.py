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


def lidar_crosstalk_noise(pointcloud, percentage):
    N, C = pointcloud.shape  # [m,], 4 (xyzi)
    c = int(percentage * N)
    index = np.random.choice(N, c, replace=False)
    pointcloud[index] += np.random.normal(size=(c, C)) * 3.0
    return pointcloud, index


def apply_cross_talk_to_numpy(point_cloud_np):  
    percentage = 0.01
    crosstalk_scan, index = lidar_crosstalk_noise(point_cloud_np, percentage=percentage)
    return crosstalk_scan