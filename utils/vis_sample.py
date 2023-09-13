"""
	Visualize sample results.
	Author: Haowen Wang
"""

import open3d as o3d
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io as scio
from data_utils import CameraInfo, create_point_cloud_from_depth_image

"""
	Given torch.Tensor, output open3d visualization.
	Args:
		all_points: points in the whole scene -> np.array
		seed_points: points select by network -> np.array
		seg: segmentation, for beauty -> np.array
		display: show -> boolean
	Return:
		pc_all -> o3d.geometry.PointCloud (with color info)
"""
def vis_sample(all_points, seed_idx, seg=None, display=False):

	all_points = all_points.cpu().numpy().reshape(-1, 3)
	seed_idx = seed_idx.cpu().numpy().reshape(-1)

	pc_all = o3d.geometry.PointCloud()
	pc_all.points = o3d.utility.Vector3dVector(all_points)

	if seg is not None:
		max_label = seg.max()
		colors = plt.get_cmap("Paired")(seg / (max_label if max_label > 0 else 1))
		for i in range(len(seg)):
			if seg[i] == 0:
				colors[i][0:3] = 211 / 255
			if i in seed_idx:
				colors[i][0] = 200
				colors[i][1] = 0
				colors[i][2] = 0
		pc_all.colors = o3d.utility.Vector3dVector(colors[:, :3])
	else:
		colors = np.zeros_like(all_points)
		for i in range(len(all_points)):
			if i in seed_idx:
				colors[i][0] = 200
				colors[i][1] = 0
				colors[i][2] = 0
			else:
				colors[i][0:3] = 211 / 255

	if display:
		vis = o3d.visualization.Visualizer()
		vis.create_window(window_name="RealScene Samples")
		render_option = vis.get_render_option()
		render_option.background_color = np.array([255, 255, 255])
		render_option.point_size = 6

		vis.add_geometry(pc_all)
		vis.run()

	return pc_all


"""
	Use Tensorboard to display the changing flow of selected points.
	Args:
		pcs: containing o3d.geometry.PointCloud -> list
		name: save dictionary -> string
"""
def write_tensorboard(pcs, name):
	logdir = "../vis/" + name
	if not os.path.isdir(logdir):
		os.makedirs(logdir)

	writer = SummaryWriter(logdir)
	for step in range(len(pcs)):
		writer.add_3d('point cloud', to_dict_batch([pcs[step]]), step=step)


def get_process_data(data_dir, num_point):
	depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
	seg = np.array(Image.open(os.path.join(data_dir, 'label.png')))
	meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
	intrinsic = meta['intrinsic_matrix']
	factor_depth = meta['factor_depth']

	# generate cloud
	camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
	cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

	# get valid points
	mask = depth > 0
	cloud_masked = cloud[mask]
	seg_masked = seg[mask]

	# sample points
	if len(cloud_masked) >= num_point:
		idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
	else:
		idxs1 = np.arange(len(cloud_masked))
		idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
		idxs = np.concatenate([idxs1, idxs2], axis=0)

	cloud_sampled = cloud_masked[idxs]
	seg_masked = seg_masked[idxs]

	return cloud_sampled, seg_masked


if __name__ == '__main__':
	data_dir = '../doc/example_data2'
	points_all, segmentation = get_process_data(data_dir, 4096)
	idx_seed = np.random.randint(0, 4096, 400)
	pc = vis_sample(points_all, idx_seed, segmentation)
