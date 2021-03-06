import numpy as np
import cv2
import glob
import os
import pickle
from ORB_matching import matching
from load_data import load_data
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bovw import bovw
from loop_closure import loopclosure
from relative_scale import comput_relative_scale

def plot_pose(pose_array,mapmax,mapmin):
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    plot_camera_pose3d(pose_array,ax1,mapmax,mapmin)
    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plot_camera_pose2d(pose_array,ax2,mapmax,mapmin)
    plt.show()
def plot_camera_pose3d(pose_array,ax,mapmax,mapmin):

    ax.plot3D(pose_array[:,0,0],pose_array[:,0,1],pose_array[:,0,2], c='r', marker='o')
    ax.dist =9
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([mapmin,mapmax])
    ax.set_ylim([mapmin,mapmax])
    ax.set_zlim([mapmin,mapmax])

    # plt.show()
def save_to_pickle(content,filename):
    pickle.dump(content,open(filename,"wb"))
def plot_camera_pose2d(pose_array,ax,mapmax,mapmin):
    ax.plot(pose_array[:,0,0],pose_array[:,0,1], c='r', marker='o')
    ax.axis((mapmin,mapmax,mapmin,mapmax))
    # plt.show()

class map:
    def __init__(self,K,D):
        self.K = K
        self.D = D

def main():
    # camera matrix
    fx = 3551.342810
    fy = 3522.689669
    cx = 2033.513326
    cy = 1455.489194
    

    # fx = 718.8560
    # fy = 718.8560
    # cx = 607.1928
    # cy = 185.2157

    K = np.float64([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])
    
    D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469])
    
    #load images
    # dataset1_dir = '/home/linjian/datasets/Data_trajectory/2018-08-21/22_47_20_load/'
    # dataset2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-22/'
    # dataset1_dir ='/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/'
    dataset1_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/16h-26m-42s load/'
    filelist1 = glob.glob(dataset1_dir+'*.jpg')
    filelist1 = sorted(filelist1)
    img_num = len(filelist1)

    #load scale as speed of the wheel
    loaded_data = load_data(dataset1_dir)
    scale = loaded_data.get_speed()

    #initialization
    rotation_array =[]
    transformation_array =[]
    pose_array =[]
    R = np.eye(3)
    t = np.zeros((1, 3))
    rotation_array.append(R)
    transformation_array.append(t)
    pose_array.append(t)
    
    #bag of virtual words init
    detector = cv2.ORB_create()
    bovw_class = bovw(detector)

    #init loop closure class
    loopclosure_class = loopclosure()

    #init the relative scale list
    relative_scale_list = []

    #init keypoints and descriptors
    keypoints_list = []
    descriptors_list = []

    #keyframe flags


    #initialize input images 
    img1 = cv2.imread(filelist1[0])
    img2 = cv2.imread(filelist1[1]) 
    keyframe_index = 1
    # for i in range(0,50): 
    for i in range(1,img_num):
   
        #initialize matching class with camera parameters
        matching_class = matching(K,D)
        #insert images 
        matching_class.load_image(img1,img2)
        #create a detector
        detector = cv2.ORB_create()

        #scan matching
        enough_match,matches = matching_class.match_images(detector)
        if matches == -1 or scale[i-1] < 0.01:
            print('not a good keyframe')
            keyframe_index = keyframe_index+1
            if keyframe_index >img_num-1:
                break
            img2 = cv2.imread(filelist1[keyframe_index])
            continue
        kp1_match,kp2_match = matches
        keypoints_list.append(matching_class.kp1)
        descriptors_list.append(matching_class.des1)

        #calculate the relative scale
        try:
            relative_scale = comput_relative_scale(kp1_match,kp2_match)
            relative_scale_list.append(relative_scale)
            print("for the ",i,"image relative scale is ",relative_scale)
            print("for the ",i,"image absolute scale is ",scale[i-1])
            print("for the ",i,"image calculated absolute scale is ",scale[i-2]/relative_scale_list[i-1]*relative_scale)      
        except:
            print("An exception occurred")
        #add into bovw
        bovw_class.add_histogram(matching_class.des1)

        #calculate rotation and transformation
        dR = matching_class.getRotation()
        rotation_array.append(dR)
        dt = np.transpose(matching_class.getTransformation())
        transformation_array.append(dt)
        R = dR.dot(R)
        t = t+dt.dot(R)*scale[i-1]
        pose_array.append(t)

        #find loop closure
        lc_index,lc_cost = bovw_class.find_lc(matching_class.des2)
        print(lc_cost)
        if (lc_cost < 0.01):
            img_lc = cv2.imread(filelist1[lc_index])
            cv2.imshow('Loop closure matched',img_lc)
            #scale calculate, 1st calutate the good matches, then relative scale
            # lc_scale = comput_relative_scale(,)
            # print('lc scale is ', scale[i-2]/relative_scale_list[i-1]*lc_scale)
        cv2.waitKey(1)  
        img1 = img2
        keyframe_index = keyframe_index+1
        if keyframe_index >img_num-1:
            break
        img2 = cv2.imread(filelist1[keyframe_index])             


    bovw_class.save_bovw_lib()
    save_to_pickle(filelist1,"image_file_list")
    #convert lists to array
    rotation_array = np.asarray(rotation_array)
    transformation_array = np.asarray(transformation_array)
    pose_array=np.asarray(pose_array)
    mapmax = np.amax(pose_array) +2
    mapmin = np.amin(pose_array) -2
    #plot
    # plot_camera_pose3d(pose_array)
    # plot_camera_pose2d(pose_array)
    plot_pose(pose_array,mapmax,mapmin)
    print('there are ', str(len(pose_array)),'number of camera poses')
    # ax.plot3D(pose_array, yline, zline, 'gray')
    # plt.show()
    #kp1_match = np.array([kp1[mat.queryIdx].pt for mat in good])
    #kp2_match = np.array([kp2[mat.trainIdx].pt for mat in good])
    #
    #kp1_match_ud = cv2.undistortPoints(np.expand_dims(kp1_match,axis=1),K,D)
    #kp2_match_ud = cv2.undistortPoints(np.expand_dims(kp2_match,axis=1),K,D)
    #

    #E, mask_e = cv2.findEssentialMat(kp1_match_ud, kp2_match_ud, focal=1.0, pp=(0., 0.), 
    #                               method=cv2.RANSAC, prob=0.999, threshold=0.001)
    #
    #print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(good),"matches")
    #
    #points, R, t, mask_RP = cv2.recoverPose(E, kp1_match_ud, kp2_match_ud, mask=mask_e)
    #print("points:",points,"\trecover pose mask:",np.sum(mask_RP!=0))
    #print("R:",R,"t:",t.T)
    
    #bool_mask = mask_RP.astype(bool)
    #img_valid = cv2.drawMatches(gr1,kp1,gr2,kp2,good, None, 
    #                            matchColor=(0, 255, 0), 
    #                            matchesMask=bool_mask.ravel().tolist(), flags=2)
    
    #plt.imshow(img_valid)
    #plt.show()
    
    # ret1, corners1 = cv2.findChessboardCorners(matching_class.gr1, (16,9),None)
    # ret2, corners2 = cv2.findChessboardCorners(matching_class.gr2, (16,9),None)
    
    # corners1_ud = cv2.undistortPoints(corners1,K,D)
    # corners2_ud = cv2.undistortPoints(corners2,K,D)
    
    # #Create 3 x 4 Homogenous Transform
    # Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    # print ("Pose_1: ", Pose_1)
    # Pose_2 = np.hstack((matching_class.R,matching_class.t))
    # print ("Pose_2: ", Pose_2)
    
    # # Points Given in N,1,2 array 
    # landmarks_hom = cv2.triangulatePoints(Pose_1, Pose_2, 
    #                                      kp1_match_ud[mask_RP[:,0]==1], 
    #                                      kp2_match_ud[mask_RP[:,0]==1]).T
    # landmarks_hom_norm = landmarks_hom /  landmarks_hom[:,-1][:,None]
    # landmarks = landmarks_hom_norm[:, :3]
    
    # corners_hom = cv2.triangulatePoints(Pose_1, Pose_2, corners1_ud, corners2_ud).T
    # corners_hom_norm = corners_hom /  corners_hom[:,-1][:,None]
    # corners_12 = corners_hom_norm[:, :3]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')         # important!
    # title = ax.set_title('3D Test')
    # ax.set_zlim3d(-5,10)
    
    # # Plot triangulated featues in Red
    # graph, = ax.plot(landmarks[:,0], landmarks[:,1], landmarks[:,2], linestyle="", marker="o",color='r')
    # # Plot triangulated chess board in Green
    # graph, = ax.plot(corners_12[:,0], corners_12[:,1], corners_12[:,2], linestyle="", marker=".",color='g')
    
    # # Plot pose 1
    # plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=0.5)
    # #Plot pose 2
    # plot_pose3_on_axes(ax, R, t.T, axis_length=1.0)
    # ax.set_zlim3d(-2,5)
    # ax.view_init(-70, -90)
    # plt.show()


if __name__ == "__main__":
    main()
