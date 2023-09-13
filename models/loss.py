""" Loss functions for training.
    Author: chenxi-wang
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE, THRESH_GOOD, THRESH_BAD,\
                       transform_point_cloud, generate_grasp_views,\
                       batch_viewpoint_params_to_matrix, huber_loss, FocalLoss

def get_loss(end_points):
    att_loss, end_points = compute_gps_loss(end_points)
    view_loss, end_points = compute_view_loss(end_points)
    grasp_loss, end_points = compute_grasp_loss(end_points)
    # object_loss, end_points = compute_objectness_loss(end_points)  # ablation

    loss = 0.5 * att_loss + 0.3 * view_loss + 0.2 * grasp_loss
    # loss = 0.5 * view_loss + 0.3 * grasp_loss + 0.2 * object_loss  # ablation
    end_points['loss/overall_loss'] = loss

    return loss, end_points

def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']  # (B, 2, Ns)
    objectness_label = end_points['objectness_label'].long()  # (B, Ns)

    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss
    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[objectness_label == 1].float().mean()

    end_points['stage1_positive_label_percentage_acc'] = (end_points['objectness_label'] == 1).float().mean()

    return loss, end_points

def compute_gps_loss(end_points):
    batch_size = end_points['ops_score'].shape[0]

    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    ops_score = end_points['ops_score']  # (B, 2, N1)
    ops_label = end_points['ops_label'].view(batch_size, -1).long()  # (B, N1)
    loss1 = criterion1(ops_score, ops_label)
    pred1 = torch.argmax(ops_score, 1)
    end_points['loss/ops_loss'] = loss1
    end_points['acc/ops_acc'] = (pred1 == ops_label.long()).float().mean()
    end_points['acc/ops_prec'] = (pred1 == ops_label.long())[pred1 == 1].float().mean()
    end_points['acc/ops_recall'] = (pred1 == ops_label.long())[ops_label == 1].float().mean()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weight('balanced', classes=np.arange(10),
                                         y=end_points['vps_label'].view(-1, 1).squeeze().cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion2 = FocalLoss(alpha=class_weights, gamma=2, reduction='mean')
    att2_score = end_points['vps_score']  # (B, 3, N2)
    obj2_label = end_points['vps_label'].view(batch_size, -1).long()  # (B, N2)
    loss2 = criterion2(att2_score, obj2_label)
    pred2 = torch.argmax(att2_score, 1)
    end_points['loss/vps_loss'] = loss2
    end_points['acc/vps_acc'] = (pred2 == obj2_label.long()).float().mean()

    loss = 0.5 * loss1 + 0.5 * loss2

    return loss, end_points

def compute_view_loss(end_points):
    criterion = nn.MSELoss(reduction='none')
    view_score = end_points['view_score']  # (B, Ns, V)
    view_label = end_points['batch_grasp_view_label']  # (B, Ns, V): select max in dim(A*D) using torch.log
    objectness_label = end_points['objectness_label']
    V = view_label.size(2)

    objectness_mask = (objectness_label > 0)
    objectness_mask = objectness_mask.unsqueeze(-1).repeat(1, 1, V)
    pos_view_pred_mask = ((view_score >= THRESH_GOOD) & objectness_mask)

    loss = criterion(view_score, view_label)
    loss = loss[objectness_mask].mean()

    end_points['loss/stage1_view_loss'] = loss
    end_points['stage1_pos_view_pred_count'] = pos_view_pred_mask.long().sum()

    return loss, end_points


def compute_grasp_loss(end_points, use_template_in_training=True):
    top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
    vp_rot = end_points['grasp_top_view_rot']  # (B, Ns, view_factor, 3, 3)
    objectness_label = end_points['objectness_label']  # (B, Ns)
    objectness_mask = objectness_label.bool()  # (B, Ns)

    # process labels (after selected top view according to network predictions)
    batch_grasp_label = end_points['batch_grasp_label']  # (B, Ns, A, D)
    batch_grasp_offset = end_points['batch_grasp_offset']  # (B, Ns, A, D, 3)
    B, Ns, A, D = batch_grasp_label.size()

    # pick the one with the highest angle score
    top_view_grasp_angles = batch_grasp_offset[:, :, :, :, 0]  # (B, Ns, A, D)
    top_view_grasp_depths = batch_grasp_offset[:, :, :, :, 1]  # (B, Ns, A, D)
    top_view_grasp_widths = batch_grasp_offset[:, :, :, :, 2]  # (B, Ns, A, D)
    target_labels_inds = torch.argmax(batch_grasp_label, dim=2, keepdim=True)  # (B, Ns, 1, D)
    target_labels = torch.gather(batch_grasp_label, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_angles = torch.gather(top_view_grasp_angles, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_depths = torch.gather(top_view_grasp_depths, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)
    target_widths = torch.gather(top_view_grasp_widths, 2, target_labels_inds).squeeze(2)  # (B, Ns, D)

    graspable_mask = (target_labels > THRESH_BAD)  # (B, Ns, D)
    objectness_mask = objectness_mask.unsqueeze(-1).expand_as(graspable_mask)
    loss_mask = (objectness_mask & graspable_mask).float()

    # 1. grasp score loss
    target_labels_inds_ = target_labels_inds.transpose(1, 2)  # (B, 1, Ns, D)
    grasp_score = torch.gather(end_points['grasp_score_pred'], 1, target_labels_inds_).squeeze(1)
    grasp_score_loss = huber_loss(grasp_score-target_labels, delta=1.0)
    grasp_score_loss = torch.sum(grasp_score_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_score_loss'] = grasp_score_loss

    # 2. inplane rotation cls loss
    target_angles_cls = target_labels_inds.squeeze(2)  # (B, Ns, D)
    criterion_grasp_angle_class = nn.CrossEntropyLoss(reduction='none')
    grasp_angle_class_score = end_points['grasp_angle_cls_pred']  # (B, A, Ns, D)
    grasp_angle_class_loss = criterion_grasp_angle_class(grasp_angle_class_score, target_angles_cls)
    grasp_angle_class_loss = torch.sum(grasp_angle_class_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_angle_class_loss'] = grasp_angle_class_loss
    grasp_angle_class_pred = torch.argmax(grasp_angle_class_score, 1)
    end_points['stage2_grasp_angle_class_acc/0_degree'] = (grasp_angle_class_pred==target_angles_cls)[loss_mask.bool()].float().mean()
    acc_mask_15 = ((torch.abs(grasp_angle_class_pred-target_angles_cls)<=1) | (torch.abs(grasp_angle_class_pred-target_angles_cls)>=A-1))
    end_points['stage2_grasp_angle_class_acc/15_degree'] = acc_mask_15[loss_mask.bool()].float().mean()
    acc_mask_30 = ((torch.abs(grasp_angle_class_pred-target_angles_cls)<=2) | (torch.abs(grasp_angle_class_pred-target_angles_cls)>=A-2))
    end_points['stage2_grasp_angle_class_acc/30_degree'] = acc_mask_30[loss_mask.bool()].float().mean()
    # print("loss mask sum: " + str(loss_mask.sum()))

    # 3. width reg loss
    grasp_width_pred = torch.gather(end_points['grasp_width_pred'], 1, target_labels_inds_).squeeze(1)  # (B, Ns, D)
    grasp_width_loss = huber_loss((grasp_width_pred-target_widths)/GRASP_MAX_WIDTH, delta=1)
    grasp_width_loss = torch.sum(grasp_width_loss * loss_mask) / (loss_mask.sum() + 1e-6)
    end_points['loss/stage2_grasp_width_loss'] = grasp_width_loss

    grasp_loss = grasp_score_loss + grasp_angle_class_loss + grasp_width_loss

    return grasp_loss, end_points
