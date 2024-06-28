import torch
from nuscenes.nuscenes import NuScenes
import os
from PIL import Image 
import numpy as np
from torchvision import transforms

def cal_mean():
    channel_mean_lst=[]
    count=0
    period=0
    for sample_token in sample_tokens:
        data=nusc.get("sample",sample_token)["data"]
        for camera in cameras:
            color_path=os.path.join(data_path,nusc.get("sample_data",data[camera])["filename"])
            color = Image.open(color_path).convert('RGB')
            img_tensor = torch.tensor(np.array(color)).float()
            channel_mean_lst.append(img_tensor.mean(dim=[0, 1]))
        count+=1
        if count%200==0:
            period+=1
            print(f"{period}% images finished")

    count=0
    period=0
    mean_r=mean_g=mean_b=0
    nums=len(channel_mean_lst)
    for channel_mean in channel_mean_lst:
        mean_r+=channel_mean[0]
        mean_g+=channel_mean[1]
        mean_b+=channel_mean[2]
        count+=1
        if count%1200==0:
            period+=1
            print(f"{period}% means calculated")
    mean_r/=nums
    mean_g/=nums
    mean_b/=nums
    print(mean_r)
    print(mean_g)
    print(mean_b)
    mean_r=float(mean_r)
    mean_g=float(mean_g)
    mean_b=float(mean_b)

    final_means=np.array([mean_r,mean_g,mean_b])
    np.savetxt("mean.txt",final_means,fmt="%f")
    print("OK!!!!!!!!!!!!!")
    
def cal_std():
    mean = torch.tensor([0.3775, 0.3823, 0.3741]).view(-1,1,1)
    channel_std_lst=[]
    count=0
    period=0
    for sample_token in sample_tokens:
        data=nusc.get("sample",sample_token)["data"]
        for camera in cameras:
            color_path=os.path.join(data_path,nusc.get("sample_data",data[camera])["filename"])
            color = Image.open(color_path).convert('RGB')
            totensor = transforms.ToTensor()
            img_tensor=totensor(color)
            std=((img_tensor-mean)**2).mean(dim=[1,2])
            channel_std_lst.append(std)
        count+=1
        if count%200==0:
            period+=1
            print(f"{period}% images finished")

    count=0
    period=0
    std_r=std_g=std_b=0
    nums=len(channel_std_lst)
    for channel_std in channel_std_lst:
        std_r+=channel_std[0]
        std_g+=channel_std[1]
        std_b+=channel_std[2]
        count+=1
        if count%1200==0:
            period+=1
            print(f"{period}% stds calculated")
    std_r/=nums
    std_g/=nums
    std_b/=nums
    print(std_r)
    print(std_g)
    print(std_b)
    std_r=float(std_r)
    std_g=float(std_g)
    std_b=float(std_b)

    final_stds=np.array([std_r,std_g,std_b])
    np.savetxt("std.txt",final_stds,fmt="%f")
    print("OK!!!!!!!!!!!!!")
    
if __name__=="__main__":
    cameras=["CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT"]

    data_path="/data_nvme/nuscenes"
    nusc = NuScenes(version="v1.0-trainval",
                                dataroot=data_path, verbose=False)

    sample_tokens=[]
    with open("../datasets/nusc/train.txt","r",encoding="utf-8") as f:
        for line in f:
            sample_tokens.append(line[:-1])
            
    print("data loaded")
    
    # cal_mean()
    cal_std()


