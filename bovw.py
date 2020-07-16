import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import glob
import pickle
import os.path

def save_trained_bovw(kmeans):
    pickle.dump(kmeans,open("bovw_database.pkl","wb"))

def load_bovw():
    return pickle.load(open("bovw_database.pkl","rb"))

def save_bovw_lib(kmeans):
    pickle.dump(kmeans,open("bovw_lib.pkl","wb"))

def load_bovw_lib():
    return pickle.load(open("bovw_lib.pkl","rb"))


class bovw:
    def __init__(self,dataset_dir,detector):
        filelist = glob.glob(dataset_dir+'*.jpg')
        self.filelist = sorted(filelist)
        self.detector = detector

        #load bovw and
        if (os.path.isfile('./bovw_database.pkl')):
            self.kmeans = load_bovw()
    def extract_descriptors(self):
        descriptor_list = []
        image_descriptors = []
        # for image_path in self.filelist[100:170]:
        for image_path in self.filelist:
            image = cv2.imread(image_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            kp,des = self.detector.detectAndCompute(image,None)
            image_descriptors.append(des)
            for descriptor in des:
                descriptor_list.append(descriptor)
        return descriptor_list,image_descriptors

    def descriptor_cluster(self,descriptor_list):
        kmeans = KMeans(n_clusters =20,n_init=10)
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
        descriptor_list,image_descriptors = self.extract_descriptors()
        his = self.build_histogram(image_descriptors,self.kmeans)
        his = np.transpose(his)
        save_bovw_lib(his)


    def train(self):
        if (os.path.isfile('./bovw_database.pkl') and os.path.isfile('bovw_lib.pkl')):
            print("The bovw lib 'bovw_database.pkl' exist")
            return
        descriptor_list,image_descriptors = self.extract_descriptors()
        kmeans = self.descriptor_cluster(descriptor_list)
        his = self.build_histogram(image_descriptors,kmeans)
        his = np.transpose(his)
        print(type(kmeans))
        # print(his)
        #save the trained model and histgram
        save_trained_bovw(kmeans)
        save_bovw_lib(his)

    def test(self, input_image_path):
        #load histogram
        his = load_bovw_lib()
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
        # plt.imshow(image_matched), plt.show()
        
        # image_matched = self.filelist[result_cos]
        # image_matched = cv2.imread(image_matched)
        # plt.imshow(image_matched), plt.show()
        
        #compute the input image index
        input_index = self.filelist.index(input_image_path)
        print('image from ',input_index)
        print('its cost are', costs[0][input_index],'and',costs[1][input_index])

    
if __name__ == "__main__":
    detector = cv2.ORB_create()
    # training_set_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/16h-26m-42s load/'
    training_set_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/17h-28m-10s unload/'
    bovw_class = bovw(training_set_dir,detector)
    bovw_class.train()
    # bovw_class.draw_keypoints()

    #testing
    test_image_dir = '/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-22/17h-28m-10s unload/1534958913.87.jpg'
    bovw_class.test(test_image_dir)
