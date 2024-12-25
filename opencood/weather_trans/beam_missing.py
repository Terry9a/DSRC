import numpy as np

def get_pcd_ringID(points, vertical_resolution=64):
    scan_x = points[:, 0]
    scan_y = points[:, 1]

    yaw = -np.arctan2(scan_y, -scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    ringID = np.cumsum(proj_y)
    ringID = np.clip(ringID, 0, vertical_resolution - 1)
    return ringID


def apply_beam_missing_to_numpy(points):
    num_beam_to_drop = 32
    vertical_resolution=64
    ringID = get_pcd_ringID(points, vertical_resolution=vertical_resolution)
    ringID = ringID.astype(np.int64)

    drop_range = np.arange(vertical_resolution)
    drop_indices = np.random.choice(drop_range, num_beam_to_drop, replace=False)
    drop_mask = np.isin(ringID, drop_indices)
    remaining_points = points[~drop_mask]

    return remaining_points
