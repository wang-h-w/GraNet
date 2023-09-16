""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time
import open3d as o3d

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import get_scene_name, create_table_points, transform_points, voxel_sample_points, \
    eval_grasp
from graspnetAPI.utils.utils import generate_scene_model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from granet_pipeline import GraNet, pred_decode
from graspnet_dataset_granet import GraspNetDataset, collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--split', required=True, help='Dataset split [seen/similar/novel]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=50, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEGIN

if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


# Create Dataset and Dataloader
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test_' + cfgs.split,
                               camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False,
                               load_label=False)
print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))

# Init the model
net = GraNet(input_feature_dim=3, batch_size=cfgs.batch_size, num_view=cfgs.num_view, num_angle=12, num_depth=4,
             cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def inference():
    batch_interval = 100
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            elif 'graph' in key:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx%256).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx, (toc-tic)/batch_interval))
            tic = time.time()

def evaluate():
    # ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test_' + cfgs.split)
    friction_list = [0.2,0.4,0.6,0.8,1.0,1.2]
    # friction_list = [0.4]
    ge = GraspEvalCostom(root=cfgs.dataset_root, camera=cfgs.camera, friction_list = friction_list, split='test_' + cfgs.split)
    if cfgs.split == 'seen':
        res, ap = ge.eval_seen(cfgs.dump_dir, proc=cfgs.num_workers)
    elif cfgs.split == 'similar':
        res, ap = ge.eval_similar(cfgs.dump_dir, proc=cfgs.num_workers)
    else:
        res, ap = ge.eval_novel(cfgs.dump_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)

class GraspEvalCostom(GraspNetEval):
    def __init__(self, root, camera, friction_list, split='test'):
        super(GraspEvalCostom, self).__init__(root, camera, split)
        self.friction_list = friction_list

    def eval_scene(self, scene_id, dump_folder, TOP_K=50, return_list=False, vis=False, max_width=0.1):
        '''
        **Input:**

        - scene_id: int of the scene index.

        - dump_folder: string of the folder that saves the dumped npy files.

        - TOP_K: int of the top number of grasp to evaluate

        - return_list: bool of whether to return the result list.

        - vis: bool of whether to show the result

        - max_width: float of the maximum gripper width in evaluation

        **Output:**

        - scene_accuracy: np.array of shape (256, 50, 6) of the accuracy tensor.
        '''
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

        list_coe_of_friction = self.friction_list

        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []

        for ann_id in range(256):
            grasp_group = GraspGroup().from_npy(
                os.path.join(dump_folder, get_scene_name(scene_id), self.camera, '%04d.npy' % (ann_id,)))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:, 1] < 0)
            max_width_mask = (gg_array[:, 1] > max_width)
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = max_width
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                     pose_list, config, table=table_trans,
                                                                     voxel_size=0.008, TOP_K=TOP_K)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:, :]),
                      end='')
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
                score_list), np.concatenate(collision_mask_list)

            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(self.root, 'scene_%04d' % scene_id, ann_id, return_poses=False,
                                                  align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 2 + 0.5  # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0.3
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype=np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                o3d.visualization.draw_geometries([pcd, *grasps_geometry])
                o3d.visualization.draw_geometries([pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries([*grasps_geometry, *model_list, t])

            # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
                indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # calculate AP
            grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, TOP_K):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                                    k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(
                            ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)

            print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (
            scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
        if not return_list:
            return scene_accuracy
        else:
            return scene_accuracy, grasp_list_list, score_list_list, collision_list_list

    def parallel_eval_scenes(self, scene_ids, dump_folder, proc=2):
        '''
        **Input:**

        - scene_ids: list of int of scene index.

        - dump_folder: string of the folder that saves the npy files.

        - proc: int of the number of processes to use to evaluate.

        **Output:**

        - scene_acc_list: list of the scene accuracy.
        '''

        res_list = []
        for scene_id in scene_ids:
            res_list.append(self.eval_scene(scene_id, dump_folder))
        return res_list


if __name__=='__main__':
    inference()
    evaluate()
