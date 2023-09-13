import os
import sys
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.io as scio
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'knn'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points
from loss_utils import GRASP_MAX_WIDTH, batch_viewpoint_params_to_matrix, transform_point_cloud, generate_grasp_views
from knn_modules import knn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--points', type=int, default=10000, help='number of points to sample')
parser.add_argument('--start', type=int, default=0, help='start scene to generate')
parser.add_argument('--end', type=int, default=100, help='end scene to generate')
parser.add_argument('--camera', default='realsense', help='realsense, kinect')
cfgs = parser.parse_args()

root = cfgs.dataset_root
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_grasp_labels():
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18:
            continue
        valid_obj_idxs.append(i + 1)  # here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                               label['scores'].astype(np.float32))  # without tolerance label

    return valid_obj_idxs, grasp_labels

def load_on_gpu(data_list):
    for i in range(len(data_list)):
        data_list[i] = data_list[i].to(device)  # load on gpu
    return data_list

class GenerateObjectness:
    def __init__(self, range_start, range_end, num_points, load_label=True, camera="realsense",
                 remove_outlier=True, remove_invisible=True):
        self.num_points = num_points
        self.load_label = load_label
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.camera_name = camera
        self.scene_ids = list(range(range_start, range_end))
        self.scene_ids = ['scene_{}'.format(str(x).zfill(4)) for x in self.scene_ids]
        self.valid_obj_idxs, self.grasp_labels = load_grasp_labels()
        self.grasp_point_merged = None
        self.grasp_label_merged = None
        self.grasp_offset_merged = None
        self.collision_labels = {}

        self.depth_path = []
        self.label_path = []
        self.meta_path = []
        self.scene_name = []
        self.frame_id = []

        for x in tqdm(self.scene_ids, desc='Loading data path...'):
            for img_num in range(256):
                self.depth_path.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.label_path.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.meta_path.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                self.scene_name.append(x.strip())
                self.frame_id.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def generate_objectness(self):
        for index in tqdm(range(len(self.depth_path)), desc="Generating objectness scores..."):
            depth = np.array(Image.open(self.depth_path[index]))
            seg = np.array(Image.open(self.label_path[index]))
            meta = scio.loadmat(self.meta_path[index])
            scene = self.scene_name[index]
            frame = self.frame_id[index]
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int8)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

            # generate cloud
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

            # get valid points
            depth_mask = (depth > 0)
            if self.remove_outlier:
                camera_poses = np.load(os.path.join(root, 'scenes', scene, self.camera_name, 'camera_poses.npy'))
                align_mat = np.load(os.path.join(root, 'scenes', scene, self.camera_name, 'cam0_wrt_table.npy'))
                trans = np.dot(align_mat, camera_poses[self.frame_id[index]])
                workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
                mask = (depth_mask & workspace_mask)
            else:
                mask = depth_mask
            cloud_masked = cloud[mask]
            seg_masked = seg[mask]

            # sample points
            if len(cloud_masked) >= self.num_points:
                idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            seg_sampled = seg_masked[idxs]
            objectness_label = seg_sampled.copy()
            objectness_label[objectness_label > 1] = 1

            object_poses_list = []
            grasp_points_list = []
            grasp_scores_list = []
            for i, obj_idx in enumerate(obj_idxs):
                if obj_idx not in self.valid_obj_idxs:
                    continue
                if (seg_sampled == obj_idx).sum() < 50:
                    continue
                object_poses_list.append(torch.from_numpy(poses[:, :, i]))
                points, _, scores = self.grasp_labels[obj_idx]
                collision = self.collision_labels[scene][i]  # (points, views, angles, depth) -> (Np, V, A, D)

                # remove invisible grasp points
                if self.remove_invisible:
                    visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled == obj_idx], points,
                                                                 poses[:, :, i], th=0.01)
                    points = points[visible_mask]
                    scores = scores[visible_mask]
                    collision = collision[visible_mask]

                grasp_points_list.append(torch.from_numpy(points))
                collision = collision.copy()
                scores = scores.copy()
                scores[collision] = 0
                grasp_scores_list.append(torch.from_numpy(scores))

            grasp_points_list = load_on_gpu(grasp_points_list)
            grasp_scores_list = load_on_gpu(grasp_scores_list)
            object_poses_list = load_on_gpu(object_poses_list)
            point_clouds = torch.from_numpy(cloud_sampled.astype(np.float32)).to(device)
            objectness_label = objectness_label.astype(np.int8)

            seed_xyz = point_clouds  # (N, 3)
            num_samples, _ = seed_xyz.shape

            poses = object_poses_list  # [(3, 4),]

            # get merged grasp points for label computation
            grasp_points_merged = []
            grasp_labels_merged = []
            for obj_idx, pose in enumerate(poses):
                grasp_points = grasp_points_list[obj_idx]  # (Np, 3)
                grasp_labels = grasp_scores_list[obj_idx]  # (Np, V, A, D)
                _, V, A, D = grasp_labels.size()
                # generate and transform template grasp views
                grasp_views = generate_grasp_views(V).to(pose.device)  # (V, 3)
                grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')  # may not exist in real scene
                grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')

                # assign views
                grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
                grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
                view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
                grasp_labels = torch.index_select(grasp_labels, 1, view_inds)  # (Np, V, A, D)
                # add to list
                grasp_points_merged.append(grasp_points_trans)
                grasp_labels_merged.append(grasp_labels)

            grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
            grasp_labels_merged = torch.cat(grasp_labels_merged, dim=0)  # (Np', V, A, D)

            # compute nearest neighbors
            seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Ns)
            grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Np')
            nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1  # (Ns) find the nearest grasp point

            # assign anchor points to real points
            grasp_labels_merged = torch.index_select(grasp_labels_merged, 0, nn_inds)  # (Ns, V, A, D)

            # process labels
            label_mask = (grasp_labels_merged > 0)
            u_max = grasp_labels_merged.max()
            grasp_labels_merged[label_mask] = torch.log(u_max / grasp_labels_merged[label_mask])  # use log
            grasp_labels_merged[~label_mask] = 0
            grasp_labels_merged = grasp_labels_merged.view(num_samples, -1).mean(-1)  # (N)
            grasp_labels_merged = grasp_labels_merged.cpu().numpy().astype(np.float32)
            objectness_score = grasp_labels_merged * objectness_label

            base = os.path.join(root, 'objectness_score', scene, self.camera_name)
            if not os.path.exists(base):
                os.makedirs(base)
            save_path_obj = os.path.join(base, 'objectness_score_'+str(frame).zfill(4) + '.npy')
            save_path_idx = os.path.join(base, 'objectness_sampled_' + str(frame).zfill(4) + '.npy')
            np.save(save_path_obj, objectness_score.astype(np.float32))
            np.save(save_path_idx, idxs.astype(np.int32))

            torch.cuda.empty_cache()


if __name__ == "__main__":
    gen = GenerateObjectness(cfgs.start, cfgs.end, num_points=cfgs.points, camera=cfgs.camera)
    gen.generate_objectness()
