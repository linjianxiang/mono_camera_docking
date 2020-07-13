import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class matching:
    def __init__(self,K,D):
        self.K = K
        self.D = D

    def load_image(self,img1,img2):
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
        # print ("Points detected: ",len(self.kp1), " and ", len(self.kp2))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.des1,self.des2)
        self.find_goodbfmatches(matches)





    def find_goodbfmatches(self,matches):
        matches = sorted(matches, key = lambda x:x.distance)[0:100]
        # print(len(matches))
        
        kp1_match = np.array([self.kp1[mat.queryIdx].pt for mat in matches])
        kp2_match = np.array([self.kp2[mat.trainIdx].pt for mat in matches])
        
        # kp1_match = cv2.undistortPoints(np.expand_dims(kp1_match,axis=1),self.K,self.D)
        # kp2_match = cv2.undistortPoints(np.expand_dims(kp2_match,axis=1),self.K,self.D)


        #the camera matrix can be used during finding essential matrix ex. findEssential(kp1_match, kp2_match, cameramatrix, method=cv2.RANSAC, prob=0.999, threshold=0.001) 
        E, mask_e = cv2.findEssentialMat(kp1_match, kp2_match, focal=1.0, pp=(0., 0.), 
                                       method=cv2.RANSAC, prob=0.999, threshold=0.001)

        # print ("Essential matrix: used ",np.sum(mask_e) ," of total ",len(matches),"matches")
        
        
        #R, output rotation matrix.
        #t, output translation vector
        #mask_RP, input/output mask for inliers in points1 and point2. If it is not empty then it marks inliners in points1 and points2
        points, self.R, self.t, mask_RP = cv2.recoverPose(E, kp1_match, kp2_match, mask=mask_e)
        print("points:",points,"\trecover pose mask:",np.sum(mask_RP!=0))
        # print("R:",self.R,"t:",self.t.T)
        
        if(points < 20):
            print("not enough matches, less than 20")


        # draw matchings
        bool_mask = mask_RP.astype(bool)
        img_valid = cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,matches, None, 
                                    matchColor=(0, 255, 0), 
                                    matchesMask=bool_mask.ravel().tolist(), flags=2)
        
        # plt.imshow(img_valid)
        # plt.show()
        cv2.imshow('feature matching',img_valid)
        cv2.waitKey(1)


    
    
    def find_goodflannmatches(self,matches):
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
    
    
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        img3 = cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,self.good,None,**draw_params)
        
        plt.imshow(img3, 'gray'),plt.show()

    def getTransformation(self):
        return self.t

    def getRotation(self):
        return self.R


def main():
    fx = 3551.342810
    fy = 3522.689669
    cx = 2033.513326
    cy = 1455.489194
    
    K = np.float64([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])
    
    D = np.float64([-0.276796, 0.113400, -0.000349, -0.000469]);
    
    # dataset1_dir = '/home/linjian/datasets/Data_trajectory/2018-08-21/22_47_20_load/'
    # dataset2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-22/'

    # filelist1 = glob.glob(dataset1_dir+'*.jpg')
    # filelist2 = glob.glob(dataset1_dir+'*.jpg')

    #img1_dir = filelist1[30]
    img1_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/1534891645.64.jpg'
    # img1_dir = '/home/linjian/datasets/Data_trajectory/2018-08-21/22_47_20_load/1534891645.64.jpg'
    #img2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-21/22_47_20_load/1534891655.03.jpg'
    # img2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958914.09.jpg'
    #img2_dir = '/home/linjian/datasets/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958918.48.jpg'
    img2_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958913.29.jpg'
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)
    
    #initialize matching class with camera parameters
    matching_class = matching(K,D)
    #insert images 
    matching_class.load_image(img1,img2)
    #create a detector
    detector = cv2.ORB_create()
    #scan matching
    matching_class.match_images(detector)

if __name__ == "__main__":
    main()
