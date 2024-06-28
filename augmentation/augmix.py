import numpy as np
from PIL import Image
import torch

from augmentation.augmentations import augs
from augmentation.config import *

def apply_op(image, op, severity):
	image = np.clip(image * 255., 0, 255).astype(np.uint8)
	pil_img = Image.fromarray(image)  # Convert to PIL.Image
	pil_img = op(pil_img, severity)
	return np.asarray(pil_img) / 255.

def normalize(image):
	"""Normalize input image channel-wise to zero mean and unit variance."""
	image = image.transpose(2, 0, 1)  # Switch to channel-first
	mean, std = np.array(MEAN), np.array(STD)
	image = (image - mean[:, None, None]) / std[:, None, None]
	return image.transpose(1, 2, 0)

def augmix(image, severity=3, width=2, depth=-1, alpha=1.):
	"""
 
	Args:
		image: PIL 读取的 RGB 图
		severity: 程度(1 到 10)
		width: augmentation chain 的宽度
		depth: augmentation chain 的深度。 -1 从 [1,3] 随机取一个深度。
		alpha: Beta and Dirichlet 分布的参数

	Returns:
		mixed: PIL.Image.Image 的 RGB 图
	"""
 
	image=np.asarray(image, dtype=np.float32) / 255.0
	ws = np.float32(
			np.random.dirichlet([alpha] * width))
	m = np.float32(np.random.beta(alpha, alpha))

	mix = np.zeros_like(image)
	for i in range(width):
		image_aug = image.copy()
		d = depth if depth > 0 else np.random.randint(1, 4)
		for _ in range(d):
			op = np.random.choice(augs)
			image_aug = apply_op(image_aug, op, severity)
		# Preprocessing commutes since all coefficients are convex
		mix += ws[i] * normalize(image_aug)

	mixed = (1 - m) * normalize(image) + m * mix
 
	mixed = np.clip(mixed * 255, 0, 255).astype(np.uint8)
	mixed = Image.fromarray(mixed).convert('RGB')
	return mixed