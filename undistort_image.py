import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def undistort(img_dir,):
#     image = cv2.imread(img_dir)
#     dst = cv2.undistort(image,mtx,dist,None,newcameramtx)
#     cv2.imshow('feature matching',dst)
#     cv2.waitKey(1)

if __name__ == "__main__":
    dataset1_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/16h-26m-42s load/'
    filelist1 = glob.glob(dataset1_dir+'*.jpg')
    filelist1 = sorted(filelist1)
    img_num = len(filelist1)

    # fx = 3551.342810
    # fy = 3522.689669
    # cx = 2033.513326
    # cy = 1455.489194
    

    fx = 718.8560
    fy = 7180.8560
    cx = 607.1928
    # cx = 1000000.0
    cy = 185.2157

    mtx = np.float64([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])
    
    # dist = np.float64([-0.276796, 0.113400, -0.000349, -0.000469])
    dist = np.float64([-0.0, 0.0, -0.0, -0.0])
    # for i in filelist1:
    img = cv2.imread(filelist1[0])

    h,w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    fig, axs = plt.subplots(1,2)
    # plt.subplot(221)
    axs[0] = fig.add_subplot(211)
    axs[0].imshow(img)
    axs[1] = fig.add_subplot(212)
    axs[1].imshow(dst)
    plt.show()
