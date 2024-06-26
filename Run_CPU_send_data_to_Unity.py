from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

from Socket import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--cal-dir", default="camera_cal/",
                help="Relative path to the directory containing camera calibration form KITTI. \
                Default is camera_cal/")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():

    FLAGS = parser.parse_args()

    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        #model = Model.Model(features=my_vgg.features, bins=2).cuda()
        model = Model.Model(features=my_vgg.features, bins=2)
        
        #checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        checkpoint = torch.load(weights_path + '/%s' % model_lst[-1], map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)


    cal_dir = FLAGS.cal_dir
    img_file = "/home/user1/Desktop/3D-BoundingBox-master-Khadem/3D-BoundingBox-master/eval/image_2/0.png"
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    truth_img = cv2.imread(img_file)
    img = np.copy(truth_img)
    yolo_img = np.copy(truth_img)

    detections = yolo.detect(yolo_img)

    for detection in detections:
        print(detection.detected_class)

        try:
            detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
        except:
            continue

        theta_ray = detectedObject.theta_ray
        input_img = detectedObject.img
        proj_matrix = detectedObject.proj_matrix
        box_2d = detection.box_2d
        detected_class = detection.detected_class



        # Start 3D
        input_tensor = torch.zeros([1, 3, 224, 224])

        input_tensor[0,:,:,:] = input_img

        [orient, conf, dim] = model(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

        dim += averages.get_item(detected_class)

        argmax = np.argmax(conf)
        orient = orient[argmax, :]
        cos = orient[0]
        sin = orient[1]
        alpha = np.arctan2(sin, cos)
        alpha += angle_bins[argmax]
        alpha -= np.pi

        if FLAGS.show_yolo:
            location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
        else:
            location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

        # if not FLAGS.hide_debug:
        print('Estimated pose: %s'%location)

        # I Added this
        print(f"Orientation (alpha) of detected {detected_class}: {alpha:.2f} radians")

        alpha_deg = (np.degrees(alpha) + 360) % 360
        print(f"Orientation (alpha) of detected {detected_class}: {alpha_deg:.2f} degrees")


        # if FLAGS.show_yolo:
        #     numpy_vertical = np.concatenate((truth_img, img), axis=0)
        #     cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        # else:
        #     cv2.imshow('3D detections', img)

        e= Elements([Element("1", Position(location[0],location[1],location[2]), Rotation(0,alpha_deg,0))])

        send_positions_single(e)

        #if FLAGS.video:
        #    cv2.waitKey(1)
        #else:
        if cv2.waitKey(0) != 32: # space bar
            exit()
            



if __name__ == '__main__':
    main()

