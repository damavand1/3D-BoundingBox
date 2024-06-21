import torch
import numpy as np
import cv2
import os
import time
from torch.autograd import Variable
from torchvision.models import vgg
from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
#from yolo.yolo import cv_Yolo

from yolo.yolo8 import cv_Yolo

from Socket import *
import argparse

ParameterSendToUnity = True

parser = argparse.ArgumentParser()

parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration from KITTI. \
                    Default is camera_cal/")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detections on a separate image")

def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location)  # 3d boxes
    return location

def main():
    FLAGS = parser.parse_args()
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s' % model_lst[-1])
        #my_vgg = vgg.vgg19_bn(pretrained=True)
        #my_vgg = vgg.vgg19_bn(weights=vgg.VGG19_BN_Weights.IMAGENET1K_V1)
        my_vgg = vgg.vgg19_bn(weights=vgg.VGG19_BN_Weights.DEFAULT)
        

        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s' % model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    #yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    #yolo = cv_Yolo(yolo_path)
    yolo = cv_Yolo()
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    #video_path = "/home/user1/Desktop/Pirooz testing/Driving in Central Prague Czechia - the Heart of Europe - 4K City Drive.mp4"
    video_path = "/home/user1/Desktop/Pirooz testing/20240222_103752.mp4"
    
    cap = cv2.VideoCapture(video_path)
    #cap = cv2.VideoCapture(2)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #target_width_ForShowToUser = 640
    #target_height_ForShowToUser = int(original_height * target_width_ForShowToUser / original_width)

    target_width_ForResizeBeforePassTo_Model = 1242 # * 375 -> is Kitti dataset Height
    #target_width_ForResizeBeforePassTo_Model = 1392 # * 375 -> is Kitti dataset Height
    target_height_ForResizeBeforePassTo_Model = int(original_height * target_width_ForResizeBeforePassTo_Model / original_width)
    
    target_height_ForCut=375

    print("target_height_ForModel (should be 375 but is): " + str(target_height_ForResizeBeforePassTo_Model))

    if ParameterSendToUnity:
        sock = connect_to_server()
        if not sock:
            return

    cal_dir = FLAGS.cal_dir
    calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    calib_file = calib_path + "calib_cam_to_cam.txt"

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.imread('/home/user1/Desktop/Pirooz VisualCortex/Kitti/training/image_2/000016.png')  
        #frame = cv2.imread('/home/user1/Desktop/Pirooz VisualCortex/Kitti/training/image_2/000036.png')            

        cv2.imshow('Pirooz - Visual Cortex - V1', frame)
        
        #frame = cv2.resize(frame, 
         #                  (target_width_ForResizeBeforePassTo_Model,
        #                    target_height_ForResizeBeforePassTo_Model))
        
        #frame = frame[-375:, :] # Dev: Crop (cut in vertical to fit with KITTI dataset pictures)
        #frame = frame[-512:, :]
        
        img = np.copy(frame)
        yolo_img = np.copy(frame)

        detections = yolo.detect(yolo_img)

        if len(detections) == 0:
            continue

        ListOfElements = []

        # Batch processing for detections
        input_tensors = []
        detected_classes = []
        proj_matrices = []
        box_2ds = []
        theta_rays = []

        for detection in detections:
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_rays.append(detectedObject.theta_ray)
            input_tensors.append(detectedObject.img)
            proj_matrices.append(detectedObject.proj_matrix)
            box_2ds.append(detection.box_2d)
            detected_classes.append(detection.detected_class)

        if len(input_tensors) == 0:
            continue

        input_tensors = torch.stack(input_tensors).cuda()
        orientations, confs, dims = model(input_tensors)
        orientations = orientations.cpu().data.numpy()
        confs = confs.cpu().data.numpy()
        dims = dims.cpu().data.numpy()

        if len(detected_classes) != len(dims):
            print("Mismatch in the number of detections and dimensions calculated.")
            continue

        for i in range(len(detected_classes)):
            dim = dims[i] + averages.get_item(detected_classes[i])
            argmax = np.argmax(confs[i])
            orient = orientations[i][argmax]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            #location = plot_regressed_3d_bbox(img, proj_matrices[i], box_2ds[i], dim, alpha, theta_rays[i])
            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrices[i], box_2ds[i], dim, alpha, theta_rays[i], frame)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrices[i], box_2ds[i], dim, alpha, theta_rays[i])

            #if not FLAGS.hide_debug:
            print('Estimated pose: %s'%location)

            alpha_deg = (np.degrees(alpha) + 360) % 360

            class_map = {"car": "1", "motorbike": "3", "truck": "7", "pedestrian": "5"}
            element_id = class_map.get(detected_classes[i], "0")
            currElement = Element(element_id, Position(location[0], 0, location[2]), Rotation(0, int(alpha_deg) + 90, 0))
            ListOfElements.append(currElement)

        #img2 = cv2.resize(img, (target_width_ForShowToUser, target_height_ForShowToUser))                    
        img2=img
        #jj
        if FLAGS.show_yolo:    
            #frame2 = cv2.resize(frame, (target_width_ForShowToUser, target_height_ForShowToUser))
            frame2=frame
            
            numpy_vertical = np.concatenate((frame2, img2), axis=0)
            cv2.imshow('Pirooz - Visual Cortex - V1', numpy_vertical)
        else:
            #img2 = cv2.resize(img, (target_width_ForShowToUser, target_height_ForShowToUser))
            cv2.imshow('Pirooz - Visual Cortex - V1', img2)
            #cv2.imshow('3D detections', img)


        e = Elements(ListOfElements)
        if ParameterSendToUnity:
            send_positions_single(sock, e)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == '__main__':
    main()
