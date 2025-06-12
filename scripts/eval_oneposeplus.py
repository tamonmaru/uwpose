# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import logging
import time

from utils.misc_utils import *
from utils.plot_utils import get_color_table, plot_one_box, draw_demo_img_corners
from utils.eval_utils import *
from utils.data_utils import letterbox_resize

from model import yolov3
from tqdm import tqdm
from pose_loss import PoseRegressionLoss

def get_path_and_poses(img_id , size ='745', cropped=True):
    # img_inf = f'color\\{img_id}.png'
    # img_uncrop = f'./oneposeplus_inference/test_seq/color_full/{img_id}.png'
    # pred_path = f'./oneposeplus_inference/test_seq/predicted_pose_test_{size}images/{img_id}.txt'
    # gt_path = f'./oneposeplus_inference/test_seq/poses_ba/{img_id}.txt'
    # k_path = f'intrin_ba\\{img_id}.txt'

    # img_inf = f'color\\{img_id}.png'
    # img_uncrop = f'./oneposeplus_inference/test_seq/color_full/{img_id}.png'
    # pred_path = f'./oneposeplus_inference/test_seq/predicted_pose_test_filtered_{size}images/{img_id}.txt'
    # gt_path = f'./oneposeplus_inference/test_seq/poses_ba/{img_id}.txt'
    # k_path = f'intrin_ba\\{img_id}.txt'

    img_inf = f'color\\{img_id}.png'
    img_uncrop = f'./oneposeplus_inference/test_seq/color_full/{img_id}.png'
    pred_path = f'./oneposeplus_inference/test_seq/predicted_pose_test_sift_{size}images/{img_id}.txt'
    gt_path = f'./oneposeplus_inference/test_seq/poses_ba/{img_id}.txt'
    k_path = f'intrin_ba\\{img_id}.txt'


    #Original Intrinsic 
    fx = 1742.75609
    fy = 1742.56896
    cx = 1056.45657
    cy= 674.98554

    K = np.array([[fx,0,cx],
                    [0,fy,cy],
                        [0,0,1]])

    pose = np.loadtxt(pred_path)
    pose_gt = np.loadtxt(gt_path)
    # K_cropped = np.loadtxt(k_path)
    if cropped :
        return img_inf, pose, pose_gt, K_cropped
    else:
        return img_uncrop, pose, pose_gt, K 
    
    #Error calculation stats
eps = 1e-5
testing_error_trans = 0.0
testing_error_angle = 0.0
testing_error_pixel = 0.0
preds_corners2D = []
gts_corners2D = []
errs_corner2D = []
errs_trans = []
errs_angle = []
errs_2d = []
errs_3d = []

roll_err = 0.0
pitch_err = 0.0
yaw_err = 0.0
count = 0
error_count = 0
model_diag = 1865.783325 

min_x,min_y,min_z = -310.211914,-626.390503,-617.810913
max_x,max_y,max_z = 310.211914,626.390503,617.810913
corners = np.array([[min_x, min_y, min_z],
                    [min_x, min_y, max_z],
                    [min_x, max_y, min_z],
                    [min_x, max_y, max_z],
                    [max_x, min_y, min_z],
                    [max_x, min_y, max_z],
                    [max_x, max_y, min_z],
                    [max_x, max_y, max_z]])

corners = np.concatenate((np.transpose(corners), np.ones((1, 8))), axis=0)
corners3D = corners

image_num = 200
width,height = 2048,1536
for idx in range (image_num):
    # if idx >=49 :
        #get image and pose path
        filename, pose, pose_gt, K  = get_path_and_poses(idx ,'50', cropped=False)
        R_pred = np.array(pose)[:3,:3]
        t_pred = np.array(pose)[:3,3:]* 1000
        R_gt = np.array(pose_gt)[:3,:3]
        t_gt = np.array(pose_gt)[:3,3:]* 1000
        
        pose_pred = np.concatenate((R_pred, t_pred), 1)
        pose_gt = np.concatenate((R_gt, t_gt), 1)
        img_ori = cv2.imread(filename)
        
        # # print(filename)
        # img_ori = cv2.resize(img_ori, (width, height))

        # # cv2.imshow('Image', img_ori)
        # # cv2.waitKey(0)

        intrinsics = K

        #project points using gt pose 
        box_gt = compute_projection(corners3D, pose_gt, intrinsics)
        box_gt = np.transpose(box_gt)

        try:
                img_ori = draw_demo_img_corners(img_ori, box_gt, (0,255,0), nV=8)
        except:
                print("Something went wrong")

        #project points using pred pose 
        bbox_3d = compute_projection(corners3D, pose_pred, intrinsics)
        corners2D_pr = np.transpose(bbox_3d)

        # print('TRANSLATION_PREDICT:',trans)
        # print('TRANSLATION GT:',t_gt)
        # Compute translation error
        # trans_dist = np.sqrt(np.sum(np.square(t_gt - t_pred)))
        trans_dist = np.linalg.norm(t_pred-t_gt)
        # print("trans_dist:", trans_dist)
        corner_norm = np.linalg.norm(box_gt - corners2D_pr, axis=1)
        # print(f'shape_boxgt :{box_gt.shape}, shape cornerpr:{corners2D_pr.shape}')
        # print('CORNER NORM',corner_norm)
        corner_dist = np.mean(corner_norm)
        if math.isnan(corner_dist):
            print('nan in reprojection error')
            error_count+=1 
            continue

        errs_corner2D.append(corner_dist)

        if math.isnan(trans_dist):
            print('nan in translational error')
            error_count+=1 
            continue

        # if corner_dist > 100:
        #     print(f'More than 100 reprojection error in image:{idx}')
        #     error_count += 1
        #     # cv2.imwrite(f'./oneposeplus_inference/sfm_set/745images/inference_images/{idx}_oneposeplus_result.png', img_ori)
        #     continue
        try:
            img_ori = draw_demo_img_corners(img_ori, corners2D_pr, (0, 0, 255), nV=8)
        except:
            print("Something went wrong")
            
        cv2.imwrite(f'./oneposeplus_inference/test_seq/inference_plot_sift/{idx}_oneposeplus_result.png', img_ori)
        print(idx)


        trans_dist = trans_dist/10 #unit adjustment (mm to cm)
        errs_trans.append(trans_dist) 
        # Compute angle error
        angle_dist = calcAngularDistancetrace(R_gt, R_pred)
        errs_angle.append(angle_dist)

        indiv_angles = calcAngularDistance(R_gt, R_pred)
        roll_err += indiv_angles[0]
        pitch_err += indiv_angles[1]
        yaw_err += indiv_angles[2]

        # Compute pixel error
        Rt_gt = np.concatenate((R_gt, t_gt), axis=1)
        Rt_pr = np.concatenate((R_pred, t_pred), axis=1)
        proj_2d_gt = compute_projection(corners3D, Rt_gt, intrinsics)
        proj_2d_pred = compute_projection(corners3D, Rt_pr, intrinsics)
        norm = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
        pixel_dist = np.mean(norm)
        errs_2d.append(pixel_dist)

        # print('Pixel Dist: ', pixel_dist)

        # Compute 3D distances
        transform_3d_gt = compute_transformation(corners3D, Rt_gt)
        transform_3d_pred = compute_transformation(corners3D, Rt_pr)
        norm3d = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
        vertex_dist = np.mean(norm3d)
        errs_3d.append(vertex_dist)

        
        testing_error_trans += trans_dist 
        testing_error_angle += angle_dist
        testing_error_pixel += pixel_dist
        count = count + 1

px_threshold = 10
acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (image_num)
acc15 = len(np.where(np.array(errs_2d) <= 15)[0]) * 100. / (image_num)
acc20 = len(np.where(np.array(errs_2d) <= 20)[0]) * 100. / (image_num)
acc5cm5deg = len(np.where((np.array(errs_trans) <= 5) & (np.array(errs_angle) <= 5))[0]) * 100. / (
        len(errs_trans) + eps)
acc10cm10deg = len(np.where((np.array(errs_trans) <= 10) & (np.array(errs_angle) <= 10))[0]) * 100. / (
        len(errs_trans) + eps)
acc15cm15deg = len(np.where((np.array(errs_trans) <= 15) & (np.array(errs_angle) <= 15))[0]) * 100. / (
        len(errs_trans) + eps)
# acc3d10 = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(lines) + eps)
# acc5cm5deg = np.mean(
#             (np.array(errs_trans) < 5) & (np.array(errs_angle) < 5)
#         )
corner_acc = len(np.where(np.array(errs_corner2D) <= px_threshold)[0]) * 100. / (image_num + eps)
mean_err_2d = np.mean(errs_2d)
mean_corner_err_2d = np.mean(errs_corner2D)

# Print test statistics
print('Correct Predictions: %d' % len(errs_2d))
print('Results of {}'.format('Aqua'))
print('   Acc using {} px 2D Projection = {:.4f}%'.format(px_threshold, acc))
print('   Acc using {} px 2D Projection = {:.4f}%'.format(15, acc15))
print('   Acc using {} px 2D Projection = {:.4f}%'.format(20, acc20))

# print('   Acc using 10% threshold - {} vx 3D Transformation = {:.4f}%'.format(diam * 0.1, acc3d10))
print('   Acc using 5 cm 5 degree metric = {:.4f}%'.format(acc5cm5deg))
print('   Acc using 10 cm 10 degree metric = {:.4f}%'.format(acc10cm10deg))
print('   Acc using 15 cm 15 degree metric = {:.4f}%'.format(acc15cm15deg))
print("   Mean 2D pixel error is %f, Mean vertex error is %f, mean corner error is %f" % (
    mean_err_2d, np.mean(errs_3d), mean_corner_err_2d))
print('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (
    testing_error_trans / count, testing_error_angle / count, testing_error_pixel / count))
print('ADD metrics:',np.mean(np.array(errs_3d) < (0.1*model_diag)))
print('Correct prediction: %f' % (count/image_num))
print('Roll error: %f' % (roll_err/count))
print('Pitch error: %f' % (pitch_err/count))
print('Yaw error: %f' % (yaw_err/count))

print('Total errors: ', error_count)