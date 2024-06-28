import cv2
import numpy as np
import os
from PIL import Image
import numpy as np
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
import os

def histogram_equalization_rgb_pil(img_pil, factor=1.0):
    # 将PIL图像转换为numpy数组
    img = np.array(img_pil)

    # 分离RGB通道
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    # 对每个通道进行直方图均衡化
    R_eq = cv2.equalizeHist(R)
    G_eq = cv2.equalizeHist(G)
    B_eq = cv2.equalizeHist(B)

    # 根据强度因子调整每个通道的强度
    R_adjusted = cv2.convertScaleAbs(R_eq, alpha=factor)
    G_adjusted = cv2.convertScaleAbs(G_eq, alpha=factor)
    B_adjusted = cv2.convertScaleAbs(B_eq, alpha=factor)

    # 合并均衡后并调整强度的通道
    img_eq = cv2.merge((R_adjusted, G_adjusted, B_adjusted))

    # 将处理后的numpy数组转换回PIL图像
    img_eq_pil = Image.fromarray(img_eq)

    # 返回均衡后的PIL图像
    return img_eq_pil

def histogram_equalization_rgb(image_path, factor=1.0):
    # 读取图像
    img = cv2.imread(image_path)
    # 将图像从BGR转换为RGB（因为OpenCV默认使用BGR）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 分离RGB通道
    R, G, B = cv2.split(img)

    # 对每个通道进行直方图均衡化
    R_eq = cv2.equalizeHist(R)
    G_eq = cv2.equalizeHist(G)
    B_eq = cv2.equalizeHist(B)

    # 根据强度因子调整每个通道的强度
    R_adjusted = cv2.convertScaleAbs(R_eq, alpha=factor)
    G_adjusted = cv2.convertScaleAbs(G_eq, alpha=factor)
    B_adjusted = cv2.convertScaleAbs(B_eq, alpha=factor)

    # 合并均衡后并调整强度的通道
    img_eq = cv2.merge((R_adjusted, G_adjusted, B_adjusted))

    # 将结果图像转换回BGR格式，以便使用OpenCV显示或保存
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_RGB2BGR)

    # 返回均衡后的图像
    return img_eq

def universal_denoise_pil(img_pil):
    # 将 PIL 图像转换为 numpy 数组
    img_np = np.array(img_pil)
    # 将图像数据转换为浮点型，适用于 skimage
    img_float = img_as_float(img_np)
    # 估计噪声的标准差，用于非局部均值去噪
    sigma_est = np.mean(estimate_sigma(img_float,channel_axis=2))
    # 应用非局部均值去噪
    # patch_size：用于去噪的邻近区域大小
    # patch_distance：搜索窗口的大小
    # h：决定过滤强度的参数，h越大去噪效果越明显
    denoise = denoise_nl_means(img_float, h=1.10 * sigma_est, fast_mode=True,
                               patch_size=5, patch_distance=6,channel_axis=2)
    # 将去噪后的图像转换回 uint8 类型
    img_denoised = img_as_ubyte(denoise)
    # 转换回 PIL 图像格式
    img_denoised_pil = Image.fromarray(img_denoised)
    return img_denoised_pil



def needs_denoising(img_pil, threshold=0.01):
    # 将 PIL 图像转换为 numpy 数组
    img_np = np.array(img_pil)
    # 将图像数据转换为浮点型，适用于 skimage
    img_float = img_as_float(img_np)
    # 估计噪声的标准差
    sigma_est = np.mean(estimate_sigma(img_float, channel_axis=2))
    # 根据阈值判断是否需要去噪
    return sigma_est > threshold

# dir="./dirty"
# cam_lst=['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

# def denoise():
#     for cor in os.listdir(dir):
#         cams_path=os.path.join(dir,cor,"samples")
#         for cam in cam_lst:
#             imgs_path=os.path.join(cams_path,cam)
#             for img in os.listdir(imgs_path):
#                 one_img_path =os.path.join(imgs_path,img)
#                 equalized_image = histogram_equalization_rgb(one_img_path)
#                 os.makedirs(os.path.join("./denoise",cor,"samples",cam),exist_ok=True)
#                 cv2.imwrite(os.path.join("./denoise",cor,"samples",cam,img),equalized_image)

# for cor in os.listdir(dir):
#     cor_img_path=os.path.join(dir,cor,"samples","CAM_BACK")
#     one_cor_img=os.listdir(cor_img_path)[0]
#     denoise_img_path=os.path.join("./denoise",cor,"samples","CAM_BACK")
#     cor_img=cv2.imread(os.path.join(cor_img_path,one_cor_img))
#     cor_img = cv2.resize(cor_img, (int(cor_img.shape[1]/2), int(cor_img.shape[0]/2)))
#     denoise_img=cv2.imread(os.path.join(denoise_img_path,one_cor_img))
#     denoise_img = cv2.resize(denoise_img, (int(denoise_img.shape[1]/2), int(denoise_img.shape[0]/2)))
#     combined_img = np.hstack((cor_img, denoise_img))
#     cv2.imwrite(os.path.join("./compare",cor+".jpg"),combined_img)