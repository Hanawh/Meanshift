import numpy as np
import cv2
import os
import argparse
import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Meanshift In Tracking')
parser.add_argument("--data_dir", type = str, default='data/FaceOcc1')
parser.add_argument("--mode", type = str, default='train')
args = parser.parse_args()

class Meanshift():
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'img')
        self.gt_dir = os.path.join(data_dir,'groundtruth_rect.txt')
        self.output_dir = os.path.join('result', data_dir.split('/')[-1])
        self.imgs, self.gt = self.data_process()
        self.x, self.y, self.w, self.h = self.gt[0] # x,y是左上角坐标
        self.hist1 = None # 目标模型直方图
        self.C = None # 权值归一化系数
        self.weight = None # 目标模型权重
        if not os.path.exists('result'):
            os.mkdir('result')
    
    def data_process(self): 
        imgs = glob.glob(os.path.join(self.img_dir + '/*.jpg'))
        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('.')[-2])) 
        gt = []
        with open(self.gt_dir, 'r') as f:
            for l in f.readlines():
                try:
                    x, y, width, height = l.split(',')
                except:
                    x, y, width, height = l.split('\t')
                gt.append([int(x),int(y),int(width),int(height)]) 
        return imgs, gt

    # step1:目标模型的描述
    def init_target(self):
        init_img = cv2.imread(self.imgs[0])
        crop = init_img[self.y : (self.y + self.h), self.x : (self.x + self.w), :]
        center_w = self.w / 2.0
        center_h = self.h / 2.0
        H = center_w * center_w + center_h * center_h # 带宽
        # 初始化权值矩阵
        mesh1, mesh2 = np.meshgrid(np.arange(self.w), np.arange(self.h))
        weight = ((mesh2 - center_h) * (mesh2 - center_h) + (mesh1 - center_w) * (mesh1 - center_w))
        self.weight = 1 - weight*1.0/H # epanneehnikov核函数
        self.C = 1.0/np.sum(self.weight) 
        # 初始化加权后的目标直方图
        hist1 = np.zeros(4096, dtype=np.float32)
        for i in range(self.h):
            for j in range(self.w):
                q_r = np.fix(float(crop[i, j, 0]) / 16)
                q_g = np.fix(float(crop[i, j, 1]) / 16)
                q_b = np.fix(float(crop[i, j, 2]) / 16)
                q_temp = int(q_r * 256 + q_g * 16 + q_b)
                hist1[q_temp] += self.weight[i, j]
        self.hist1 = hist1 * self.C


    def train(self):
        self.init_target()
        rect = np.array(self.gt[0])
        rect[2] = np.ceil(rect[2])
        rect[3] = np.ceil(rect[3])
        for ind in range(1, len(self.imgs)):
            image = cv2.imread(self.imgs[ind])
            shift = np.array([1, 1])
            iter = 0
            while(np.sum(shift * shift) > 0.5 and iter < 20):
                iter += 1
                crop_cur = image[int(rect[1]):int(rect[1] + rect[3]), int(rect[0]):int(rect[0] + rect[2])]
                # step2:候补模型的描述
                hist2 = np.zeros(4096, dtype=np.float32)
                q_temp = np.zeros((self.h, self.w), dtype=np.int32)
                for i in range(crop_cur.shape[0]):
                    for j in range(crop_cur.shape[1]):
                        q_r = np.fix(float(crop_cur[i, j, 0]) / 16)
                        q_g = np.fix(float(crop_cur[i, j, 1]) / 16)
                        q_b = np.fix(float(crop_cur[i, j, 2]) / 16)
                        q_temp[i, j] = int(q_r * 256 + q_g * 16 + q_b) # 该像素的直方图表示的索引
                        hist2[q_temp[i, j]] += self.weight[i, j]
                hist2 = hist2 * self.C
                temp_w = np.zeros(4096, dtype=np.float32)
                mask = (hist2 != 0)
                temp_w[mask] = np.sqrt(self.hist1[mask] / hist2[mask])

                w_sum = 0
                shift = np.array([0,0])
                for i in range(self.h):
                    for j in range(self.w):
                        w_sum += temp_w[int(q_temp[i, j])]
                        shift[0] += temp_w[int(q_temp[i, j])] * (i - self.h / 2.0 - 0.5)
                        shift[1] += temp_w[int(q_temp[i, j])] * (j - self.w / 2.0 - 0.5)
                shift = shift / w_sum
                rect[0] = rect[0] + shift[1]
                rect[1] = rect[1] + shift[0]
                rect[0:2] = np.ceil(rect[0:2])
            
            cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2)
            gt = self.gt[ind+1]
            cv2.rectangle(image, (int(gt[0]), int(gt[1])), (int(gt[0] + gt[2]), int(gt[1] + gt[3])), (0, 255, 0), 2)
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
            cv2.imwrite(self.output_dir+'/%05d'%ind+'.jpg', image)    
    
    def make_result(self):
        cv2.namedWindow('result')
        path = sorted(glob.glob(self.output_dir + "/*.jpg"))
        for ind in range(len(path)):
            img = cv2.imread(path[ind]) #结果图
            cv2.imwrite(self.output_dir+'/%05d'%ind+'.jpg', img)
            cv2.imshow('result', img)
            cv2.waitKey(100)


if __name__ == '__main__':
    meanshift = Meanshift(args.data_dir)
    if(args.mode == "train"):
        meanshift.train()
    else:
        meanshift.make_result()


        
