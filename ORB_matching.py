import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pose3_on_axes(axes, gRp, origin, axis_length=0.1):
    """Plot a 3D pose on given axis 'axes' with given 'axis_length'."""
    # get rotation and translation (center)
    #gRp = pose.rotation().matrix()  # rotation from pose to global
    #t = pose.translation()
    #origin = np.array([t.x(), t.y(), t.z()])

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin, x_axis, axis=0) 
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'r-')
    y_axis = origin + gRp[:, 1] * axis_length 
    line = np.append(origin, y_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'g-')

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin, z_axis, axis=0)
    axes.plot(line[:, 0], line[:, 1], line[:, 2], 'b-')



class matching:
    def __init__(self,K,D):
        self.K = K
        self.D = D

    def load_image(self,img1_dir,img2_dir):
        img1 = cv2.imread(img1_dir)
        img2 = cv2.imread(img2_dir)
        self.img1 = img1
        self.img2 = img2
        # Convert images to greyscale
        self.gr1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        self.gr2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    def get_gray_imgs(self):
        return self.gr1,self.gr2

    def match_images(self,detector):
        self.detector = detector
        self.kp1, self.des1 = detector.detectAndCompute(self.gr1,None)
        self.kp2, self.des2 = detector.detectAndCompute(self.gr2,None)
        print ("Points detected: ",len(self.kp1), " and ", len(self.kp2))
    
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number=6,key_size=12,multi_probe_level=1)
        search_params = dict(checks = 50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(self.des1,self.des2,k=2)

        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #
        #matches = bf.match(self.des1,self.des2)
        self.find_goodmatches(matches)
    
    
    def find_goodmatches(self,matches):
        # store all the good matches as per Lowe's ratio test.
        self.good = []
        #print(matches)
        for m,n in matches:
                if m.distance < 0.7*n.distance:
                       self.good.append(m)
        if len(self.good) > 10:
            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in self.good ]).reshape(-1,1,2)
            dst_pts = np.float32([ self.kp2[m.trainIdx].pt for m in self.good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            
            #print(img1.shape)
            #h,w = img1.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            #dst = cv2.perspectiveTransform(pts,M)
            #
            #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
        else:
            print("not enough matches are found")
            matchesMask = None
        self.draw_matches(matchesMask)
    
    
    def draw_matches(self,matchesMask):
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        img3 = cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.good,None,**draw_params)
        
        plt.imshow(img3, 'gray'),plt.show()

    def undistort_images(self):
        self.img1 = cv2.undistortPoints(self.img1,self.K,self.D)
        self.img2 = cv2.undistortPoints(self.img2,self.K,self.D)


    def caluclate_transformation(self):
        E, mask_e = cv2.findEssentialMat(self.kp1, self.kp2, focal=1.0, pp=(0., 0.), 
                                       method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(good),"matches")
        
        points, R, t, mask_RP = cv2.recoverPose(E, kp1_match_ud, kp2_match_ud, mask=mask_e)
        print("points:",points,"\trecover pose mask:",np.sum(mask_RP!=0))
        print("R:",R,"t:",t.T)
def main():
    fx = 3551.342810
    fy = 3522.689669
    cx = 2033.513326
    cy = 1455.489194
    
    K = np.float64([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])
    
    D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);
    
    
    #img1_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/1534891645.64.jpg'
    img1_dir = '/home/linjian/datasets/Data_trajectory/2018-08-21/22_47_20_load/1534891645.64.jpg'
    #img2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-21/22_47_20_load/1534891655.03.jpg'
    img2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958914.09.jpg'
    #img2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958918.48.jpg'
    #img2_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958913.29.jpg'
    
    #initialize matching class with camera parameters
    matching_class = matching(K,D)
    #insert images 
    matching_class.load_image(img1_dir,img2_dir)
    #create a detector
    detector = cv2.ORB_create()
    #scan matching
    matching_class.match_images(detector)
    
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
    
    #ret1, corners1 = cv2.findChessboardCorners(gr1, (16,9),None)
    #ret2, corners2 = cv2.findChessboardCorners(gr2, (16,9),None)
    #
    #corners1_ud = cv2.undistortPoints(corners1,K,D)
    #corners2_ud = cv2.undistortPoints(corners2,K,D)
    #
    ##Create 3 x 4 Homogenous Transform
    #Pose_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    #print ("Pose_1: ", Pose_1)
    #Pose_2 = np.hstack((R, t))
    #print ("Pose_2: ", Pose_2)
    #
    ## Points Given in N,1,2 array 
    #landmarks_hom = cv2.triangulatePoints(Pose_1, Pose_2, 
    #                                     kp1_match_ud[mask_RP[:,0]==1], 
    #                                     kp2_match_ud[mask_RP[:,0]==1]).T
    #landmarks_hom_norm = landmarks_hom /  landmarks_hom[:,-1][:,None]
    #landmarks = landmarks_hom_norm[:, :3]
    #
    #corners_hom = cv2.triangulatePoints(Pose_1, Pose_2, corners1_ud, corners2_ud).T
    #corners_hom_norm = corners_hom /  corners_hom[:,-1][:,None]
    #corners_12 = corners_hom_norm[:, :3]
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')         # important!
    #title = ax.set_title('3D Test')
    #ax.set_zlim3d(-5,10)
    #
    ## Plot triangulated featues in Red
    #graph, = ax.plot(landmarks[:,0], landmarks[:,1], landmarks[:,2], linestyle="", marker="o",color='r')
    ## Plot triangulated chess board in Green
    #graph, = ax.plot(corners_12[:,0], corners_12[:,1], corners_12[:,2], linestyle="", marker=".",color='g')
    #
    ## Plot pose 1
    #plot_pose3_on_axes(ax,np.eye(3),np.zeros(3)[np.newaxis], axis_length=0.5)
    ##Plot pose 2
    #plot_pose3_on_axes(ax, R, t.T, axis_length=1.0)
    #ax.set_zlim3d(-2,5)
    #ax.view_init(-70, -90)
    #plt.show()


if __name__ == "__main__":
    main()
