import numpy as np
import cv2
from matplotlib import pyplot as plt

#Insert an object with known size into the scene that are filming with the 2D camera. Then, from the 3D point cloud can compute the scale. In the 3D point cloud, we have to identify the object, and then we can e.g. identify two corner points in the object. And between these corner points you know the distance in the real world, so can now compute the scale.

# same map points R_i2,p ^T R_i1,p?
class loopclosure:
    def __init__(self,):
        print('init loop closure class')

    def if_good_match(self,des1,kp1,des2,kp2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        self.find_goodbfmatches(matches)

    def find_goodbfmatches(self,matches,kp1,kp2):
        matches = sorted(matches, key = lambda x:x.distance)[0:100]
        # print(len(matches))
        
        kp1_match = np.array([kp1[mat.queryIdx].pt for mat in matches])
        kp2_match = np.array([kp2[mat.trainIdx].pt for mat in matches])
        
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
            return -1

        return True
        # # draw matchings
        # bool_mask = mask_RP.astype(bool)
        # img_valid = cv2.drawMatches(self.img1,self.kp1,self.img2,self.kp2,matches, None, 
        #                             matchColor=(0, 255, 0), 
        #                             matchesMask=bool_mask.ravel().tolist(), flags=2)
        
        # # plt.imshow(img_valid)
        # # plt.show()
        # cv2.imshow('feature matching',img_valid)
