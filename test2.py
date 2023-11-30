import copy
import math

import requests

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights,MaskRCNN_ResNet50_FPN_V2_Weights

#this is the file with auxillary functions. stereo_image_utils.py. Should be in the same
#directory as the notebook
import stereo_image_utils
from stereo_image_utils import get_detections, get_cost, draw_detections, annotate_class2 
from stereo_image_utils import get_horiz_dist_corner_tl, get_horiz_dist_corner_br, get_dist_to_centre_tl, get_dist_to_centre_br

fl = 2.043636363636363
tantheta = 0.7648732789907391

frame_l = cv2.imread("C:/Users/dibya/Downloads/Minor project/esp32_stereo_camera/python_notebooks/project/left_30.jpg")
frame_r = cv2.imread("C:/Users/dibya/Downloads/Minor project/esp32_stereo_camera/python_notebooks/project/right_30cm.jpg")

print(frame_l)

cnt = 0


weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
_ = model.eval()



imgs = [cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB),cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)]
if cnt == 0:
    cnt = 1
                
    det, lbls, scores, masks = get_detections(model,imgs)
#                 if (len(det[1])==len(det[0])):
#                     det[1] = det[1][:-1]
    sz1 = frame_r.shape[1]
    centre = sz1/2
    print(det)
    print(np.array(weights.meta["categories"])[lbls[0]])
    print(np.array(weights.meta["categories"])[lbls[1]])
    cost = get_cost(det, lbls = lbls,sz1 = centre)
    tracks = scipy.optimize.linear_sum_assignment(cost)

    dists_tl =  get_horiz_dist_corner_tl(det)
    dists_br =  get_horiz_dist_corner_br(det)

    final_dists = []
    dctl = get_dist_to_centre_tl(det[0],cntr = centre)
    dcbr = get_dist_to_centre_br(det[0], cntr = centre)

    for i, j in zip(*tracks):
        if dctl[i] < dcbr[i]:
            final_dists.append((dists_tl[i][j],np.array(weights.meta["categories"])[lbls[0]][i]))

        else:
            final_dists.append((dists_br[i][j],np.array(weights.meta["categories"])[lbls[0]][i]))
    
    #final distances as list
    fd = [i for (i,j) in final_dists]
    #find distance away
    dists_away = (7.05/2)*sz1*(1/tantheta)/np.array((fd))+fl
    cat_dist = []
    for i in range(len(dists_away)):
        cat_dist.append(f'{np.array(weights.meta["categories"])[lbls[0]][(tracks[0][i])]} {dists_away[i]:.1f}cm')
        print(f'{np.array(weights.meta["categories"])[lbls[0]][(tracks[0][i])]} is {dists_away[i]:.1f}cm away')
    t1 = [list(tracks[1]), list(tracks[0])]
    frames_ret = []
    for i, imgi in enumerate(imgs):
        img = imgi.copy()
        deti = det[i].astype(np.int32)
        draw_detections(img,deti[list(tracks[i])], obj_order=list(t1[1]))
        annotate_class2(img,deti[list(tracks[i])],lbls[i][list(tracks[i])],cat_dist)
        frames_ret.append(img)
    cv2.imshow("left_eye", cv2.cvtColor(frames_ret[0],cv2.COLOR_RGB2BGR))
    cv2.imshow("right_eye", cv2.cvtColor(frames_ret[1],cv2.COLOR_RGB2BGR))

    while True:
        key1 = cv2.waitKey(1)
        if key1 == ord('p'):
            break