import numpy as np
import pandas as pd

class load_data:
    def __init__(self,dir):
        self.csv_dir = dir+'camera8102.csv'
        # f = open(csv_dir,'r')
        # csv_reader = csv.DictReader(f)
        # # for row in csv_reader:
        # #     print(row)
        # # print(csv_reader['Wheel Angle'])
        # f.close()

    def get_wheel_angle(self):
        df = pd.read_csv(self.csv_dir)
        wheel_angle = df['Wheel Angle']
        wheel_angle = wheel_angle.to_numpy()
        return wheel_angle

    def get_speed(self):
        df = pd.read_csv(self.csv_dir)
        speed = df['Speed']
        speed = speed.to_numpy()
        # print(speed[100])
        return speed



if __name__ == "__main__":
    dataset1_dir ='/home/linjian/dataset/docking_dataset/image/Data_trajectory/2018-08-21/22_47_20_load/'
    data = load_data(dataset1_dir)
    scale = data.get_speed()
    print(scale[10])