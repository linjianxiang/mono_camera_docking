import numpy as np
import random

def comput_relative_scale(matched_kp1, matched_kp2):
    # ref_scale_first = naive_average_method(reference_kp,matched_kp1)
    ref_scale_second = naive_average_method(matched_kp1,matched_kp2)
    # print("the first scale is ", ref_scale_first, "second is ",ref_scale_second)
    # return ref_scale_first*ref_scale_second
    return ref_scale_second

def naive_average_method(matched_kp1,matched_kp2):
    kp_number = matched_kp1.shape[0]
    pairs = np.random.randint(kp_number,size =(kp_number,2))
    # print(pairs)
    dis1 = []
    dis2 = []
    for i in pairs:
        first = i[0]
        second = i[1]
        dis1.append(np.linalg.norm(matched_kp1[first,:] - matched_kp1[second,:]))
        dis2.append(np.linalg.norm(matched_kp2[first,:] - matched_kp2[second,:]))
    dis1 = np.array(dis1)
    dis2 = np.array(dis2)
    # print("dis 1 " ,dis1)
    # print("dis 2 ",dis2)
    scale = dis1/dis2
    # print("scale cal ",scale)
    notnan_num = sum(not i for i in np.isnan(scale))
    # print("totoal not nan is ", notnan_num)
    # print("sum scale is ", np.nansum(scale))
    average = np.nansum(scale)/notnan_num
    # print("the relative scale is ", average)
    return average
    







if __name__ == "__main__":
    kp1 = np.array([[1.,2.,3.,4.,5.,6.,7.,8.],[1.,2.,3.,4.,5.,6.,7.,8.]])
    print(kp1[:,0])
    kp2 = 2*np.array([[1.,2.2,3.,3.5,5.,6.,7.5,8.],[1.,2.2,3.,3.5,5.,6.,7.5,8.]])
    comput_relative_scale(kp1,kp2)