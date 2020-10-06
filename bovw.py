import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import glob
import pickle
import os
import sys
import re

def save_trained_bovw(kmeans):
    pickle.dump(kmeans,open("bovw_database.pkl","wb"))

def load_bovw():
    return pickle.load(open("bovw_database.pkl","rb"))

def save_bovw_lib(his):
    pickle.dump(his,open("bovw_lib.pkl","wb"))

def load_bovw_lib():
    return pickle.load(open("bovw_lib.pkl","rb"))

def generate_db_trainingset(top_folder_dir):
    # top_folder_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory'
    
    image_folders_dir = ([x[0] for x in os.walk(top_folder_dir)])
    # print(image_folders)
    f = open("filelist.txt","w")
    print(len(image_folders_dir))
    for folder_dir in image_folders_dir:
        filelist = glob.glob(folder_dir+'/*.jpg')
        image_number = len(filelist)
        if (image_number >200):
            # print(folder_dir)
            use_image = image_number//200
            for i in range(0,image_number,use_image):
                f.write(filelist[i]+'\n')
    f.close()
    
class bovw:
    def __init__(self,detector):
        self.cluster_number = 200
        self.detector = detector
        self.his = []
        #load bovw and
        if (os.path.isfile('./bovw_database.pkl')):
            self.kmeans = load_bovw()
    def extract_descriptors(self,filelist):
        descriptor_list = []
        image_descriptors = []
        # for image_path in self.filelist[100:170]:
        for image_path in filelist:
            image = cv2.imread(image_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kp,des = self.detector.detectAndCompute(image,None)
            # self.add_histogram(des)
            image_descriptors.append(des)
            for descriptor in des:
                descriptor_list.append(descriptor)
        return descriptor_list,image_descriptors

    def descriptor_cluster(self,descriptor_list):
        kmeans = KMeans(n_clusters =self.cluster_number,n_init=40)
        kmeans.fit(descriptor_list)
        return kmeans

    def build_histogram(self,image_descriptors, cluster_alg):
        histogram_array = np.zeros([len(image_descriptors),len(cluster_alg.cluster_centers_)])
        for idx, descriptors in enumerate(image_descriptors):
            histogram = self.build_single_histogram(descriptors,cluster_alg)
            histogram_array[idx] = histogram
        return histogram_array
    
    # def build_histogram_list(self,cluster_alg):

    def build_single_histogram(self,descriptors,cluster_alg):
        histogram = np.zeros(len(cluster_alg.cluster_centers_))
        
        cluster_result =  cluster_alg.predict(descriptors)
        for i in cluster_result:
            histogram[i] += 1.0
        return histogram

    def add_histogram(self, des):
        his = self.build_single_histogram(des, self.kmeans)
        # his = np.transpose(his)
        self.his.append(his)

    def save_bovw_lib(self):
        self.his = np.array(self.his)
        self.his = np.transpose(self.his)
        pickle.dump(self.his,open("bovw_lib.pkl","wb"))


    ## weighing tf-idf function
    def reweight_tf_idf(self,histograms):
        re_hists  = np.zeros(histograms.shape)
        N = histograms.shape[0]
        # print(histograms)
        n_i = np.sum(histograms > 0, axis=0)
        # print(n_i)
        for hist_id in range(histograms.shape[0]):
            n_d  = np.sum(histograms[hist_id])
            for bin_id in range(len(histograms[hist_id])): 
                # print(histograms[hist_id, bin_id]/n_d)
                # print(N/n_i[bin_id])
                # print(np.log(N/n_i[bin_id]))
                re_hists[hist_id, bin_id] = histograms[hist_id, bin_id]/ n_d * np.log(N/n_i[bin_id])
    #             print(re_hists[hist_id, bin_id], histograms[hist_id, bin_id], n_d, N, n_i[bin_id])
        return re_hists

    def normToUnitLength(self,v):
        # print('v shape',v.shape)
        v_length = np.linalg.norm(v,axis = 0)
        # print('vlengh is ', v_length)
        v_norm = v
        v_norm = v_norm / v_length
        return v_norm

    def compute_cost_matrices(self,histograms,input_histogram):
        cost_matrix_eucl  = np.zeros(histograms.shape[1])
        cost_matrix_cos  = np.zeros(histograms.shape[1])
        histograms_trans = np.transpose(histograms)
        for row, hist_row in enumerate(histograms_trans):
            # print(hist_row)
            # print(input_histogram)
            eucl_dist = np.linalg.norm(hist_row-input_histogram)
            cost_matrix_eucl[row] = eucl_dist
            cos_sim = 1- np.dot(hist_row, input_histogram) / (np.linalg.norm(hist_row)* np.linalg.norm(input_histogram))
            # print(np.dot(hist_row, input_histogram))
            # print((np.linalg.norm(hist_row)* np.linalg.norm(input_histogram)))
            # print(cos_sim)
            cost_matrix_cos[row] = cos_sim
        return cost_matrix_eucl, cost_matrix_cos



    def comput_histgram(self,image_descriptors):
        if (not os.path.isfile('./bovw_database.pkl')):
            print("The bovw lib 'bovw_database.pkl' exist")
            return -1
        #extract descriptors from image sets
        descriptor_list,image_descriptors = self.extract_descriptors(self.fileilst)
        his = self.build_histogram(image_descriptors,self.kmeans)
        his = np.transpose(his)
        save_bovw_lib(his)

    def train_db(self,dataset_dir):
        if (os.path.isfile('./bovw_database.pkl')):
            print("The bovw db 'bovw_database.pkl' exist")
            return

        f = open(dataset_dir,"r")
        filelist = f.read().split('\n')
        filelist.pop()
        print("there are ", len(filelist)," images in datase")
        f.close()
        # print(filelist)
        descriptor_list,image_descriptors = self.extract_descriptors(filelist)
        kmeans = self.descriptor_cluster(descriptor_list)
        print(type(kmeans))
        # print(his)
        #save the trained model and histgram
        save_trained_bovw(kmeans)


    def train(self,dataset_dir):
        if (not os.path.isfile('./bovw_database.pkl')):
            print("The bovw lib 'bovw_database.pkl' does not exist, please ")
            sys.exit()
            return

        filelist = glob.glob(dataset_dir+'*.jpg')
        # self.filelist = sorted(filelist)
        filelist.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.filelist  = filelist
        if (os.path.isfile('./bovw_database.pkl') and os.path.isfile('bovw_lib.pkl')):
            print("The bovw lib 'bovw_database.pkl' exist")
            return
        
        print('number of files :' ,len(self.filelist))
        descriptor_list,image_descriptors = self.extract_descriptors(self.filelist)
        kmeans = self.kmeans
        his = self.build_histogram(image_descriptors,kmeans)
        print('his number is ' ,his.shape)
        his = np.transpose(his)
        print(type(kmeans))
        # print(his)
        #save the trained model and histgram
        save_trained_bovw(kmeans)
        # save_bovw_lib(his)
        print("his shape is", his.shape)
        # self.his = np.array(self.his)
        self.his = his
        # self.his = np.transpose(self.his)

        # print(self.his.shape)
        # print(self.his)
        # print(his)
        save_bovw_lib(self.his)



    def find_match(self,input_image_path):
        #load histogram
        # his = np.array(self.his)
        his = load_bovw_lib()       
        print('full his size is ',his.shape)
        # self.kmeans = load_bovw()
        image = cv2.imread(input_image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp,des = self.detector.detectAndCompute(image,None)
        input_his = np.expand_dims(self.build_single_histogram(des, self.kmeans),axis = 1)
        #normalize the histogram
        his = self.normToUnitLength(his)
        input_his = self.normToUnitLength(input_his)
        #compute cost of histogram, contains cost of euclidean distance and cos angle
        costs = self.compute_cost_matrices(his,input_his)
        # print('cost',costs)
        #find the minmum cost as the best match
        mincosts = np.amin(costs,axis=1)
        result_euclidean = np.where(costs[0] == mincosts[0])[0][0]
        result_cos = np.where(costs[1] == mincosts[1])[0][0]
        print("the minimum cost is ", str(mincosts), "its index is ",result_euclidean,'and',result_cos)
        ##plot matched images
        # image_matched = self.filelist[result_euclidean]
        # image_matched = cv2.imread(image_matched)
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(image,cmap='gray')
        # axs[1].imshow(image_matched)
        # plt.show()
        
        
        # image_matched = self.filelist[result_cos]
        # print(image_matched)
        # image_matched = cv2.imread(image_matched)
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(image,cmap='gray')
        # axs[1].imshow(image_matched)
        # plt.show()
        
        #compute the input image index
        input_index = self.filelist.index(input_image_path)
        print('image from ',input_index)
        print('its cost are', costs[0][input_index],'and',costs[1][input_index])
        return result_cos,mincosts[1] #return the index and cost of minimum
       
        
    def find_lc(self,des):
        #load histogram
        # his = np.array(self.his)
        his = self.his
        his = np.array(his)
        his = np.transpose(his)

        input_his = np.expand_dims(self.build_single_histogram(des, self.kmeans),axis = 1)
        #normalize the histogram
        his = self.normToUnitLength(his)
        input_his = self.normToUnitLength(input_his)
        #compute cost of histogram, contains cost of euclidean distance and cos angle
        self.costs = self.compute_cost_matrices(his,input_his)
        # print('cost',costs)
        #find the minmum cost as the best match
        mincosts = np.amin(self.costs,axis=1)
        # result_euclidean = np.where(costs[0] == mincosts[0])[0][0]
        result_cos_index = np.where(self.costs[1] == mincosts[1])[0][0]
        # print("the minimum cost is ", str(mincosts), "its index is ",result_euclidean,'and',result_cos)
        ##plot matched images
        # image_matched = self.filelist[result_euclidean]
        # image_matched = cv2.imread(image_matched)
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(image,cmap='gray')
        # axs[1].imshow(image_matched)
        # plt.show()
        
        
        # image_matched = self.filelist[result_cos]
        # print(image_matched)
        # image_matched = cv2.imread(image_matched)
        # fig,axs = plt.subplots(2)
        # axs[0].imshow(image,cmap='gray')
        # axs[1].imshow(image_matched)
        # plt.show()
        return result_cos_index,mincosts[1] #return the index and cost of minimum
       
    def get_lowest_costs_index(self,num):
        cost = np.sort(self.costs,axis=1)
        # print("cost is ", cost, " shape is ", cost.shape)
        if num > cost.shape[1]:
            num = cost.shape[1]
        indices = []
        for i in range(num):
            # print(num)
            # print(i)
            # print(self.costs[1],"and ", cost[0])
            # print(np.where(self.costs[1] == cost[1][i]))
            indices.append(np.where(self.costs[1] == cost[1][i])[0][0])
        return indices       

    def get_costs(self):
        return  self.costs 
        
    def test(self, input_image_path):
        #load histogram
        his = load_bovw_lib()
        print('his number is ' ,his.shape)
        self.kmeans = load_bovw()
        image = cv2.imread(input_image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp,des = self.detector.detectAndCompute(image,None)
        input_his = np.expand_dims(self.build_single_histogram(des, self.kmeans),axis = 1)
        #normalize the histogram
        his = self.normToUnitLength(his)
        input_his = self.normToUnitLength(input_his)
        #compute cost of histogram, contains cost of euclidean distance and cos angle
        costs = self.compute_cost_matrices(his,input_his)
        print('cost number is ' ,costs[1].shape)
        self.costs = costs
        # print('cost',costs)
        #find the minmum cost as the best match
        mincosts = np.amin(costs,axis=1)
        result_euclidean = np.where(costs[0] == mincosts[0])[0][0]
        result_cos = np.where(costs[1] == mincosts[1])[0][0]
        print("the minimum cost is ", str(mincosts), "its index is ",result_euclidean,'and',result_cos)
        ##plot matched images
        image_matched = self.filelist[result_euclidean]
        image_matched = cv2.imread(image_matched)
        fig,axs = plt.subplots(2)
        axs[0].imshow(image,cmap='gray')
        axs[1].imshow(image_matched)
        plt.show()
        
        
        image_matched = self.filelist[result_cos]
        print(image_matched)
        image_matched = cv2.imread(image_matched)
        fig,axs = plt.subplots(2)
        axs[0].imshow(image,cmap='gray')
        axs[1].imshow(image_matched)
        plt.show()
        
        #compute the input image index
        input_index = self.filelist.index(input_image_path)
        print('image from ',input_index)
        print('its cost are', costs[0][input_index],'and',costs[1][input_index])

    
if __name__ == "__main__":
    # detector = cv2.ORB_create()
    # training_set_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/16h-26m-42s load/'
    # # training_set_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/17h-28m-10s unload/'
    # bovw_class = bovw(detector)
    # # bovw_class.train_db(training_set_dir)
    # bovw_class.train_db('filelist.txt')

    # image_set_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/16h-26m-42s load/'
    # # image_set_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/'
    # bovw_class.train(image_set_dir)


    # #testing
    # test_image_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/16h-26m-42s load/1534955250.9.jpg'
    # # test_image_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/1534891666.84.jpg'
    # # bovw_class.test(test_image_dir)

    # #find match
    # bovw_class.find_match(test_image_dir)

    top_dir = '/home/linjian/dataset/tum_image'
    # #generate db training set
    generate_db_trainingset(top_dir)
    detector = cv2.ORB_create()
    bovw_class = bovw(detector)
    bovw_class.train_db('filelist.txt')


    # image_set_dir = top_dir+'/sequence_30/resized_images/'
    image_set_dir ='/home/linjian/dataset/docking_dataset/image/Data_trajectory/load_unload/'
    bovw_class.train(image_set_dir)


    # #testing
    # test_image_dir = image_set_dir+'2.0.jpg'
    test_image_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/1534891669.83.jpg'
    bovw_class.test(test_image_dir)
    lc_indices = bovw_class.get_lowest_costs_index(100)
    print(lc_indices)
    print(bovw_class.costs[1][lc_indices])


