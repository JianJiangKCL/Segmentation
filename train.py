from skimage import io, transform
from sklearn.model_selection import train_test_split
import argparse
import glob
import matplotlib.pyplot as plt
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import albumentations as A
from albumentations.pytorch import ToTensor
from tqdm import tqdm as tqdm
from model import UNet
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)


class NucleiDataset(Dataset):
	def __init__(self, root, folders, transform=None):
		self.root = root
		self.folders = folders
		self.transforms = transform

	def __len__(self):
		return len(self.folders)

	def __getitem__(self, idx):
		image_folder = os.path.join(self.root, self.folders[idx], 'images/')
		mask_folder = os.path.join(self.root, self.folders[idx], 'masks/')

		image_path = os.path.join(image_folder, os.listdir(image_folder)[0])
		img = io.imread(image_path)[:, :, :3].astype('float32')
		mask = self.get_mask(mask_folder).astype('uint8')

		augmented = self.transforms(image=img, mask=mask)
		img, mask = augmented['image'], augmented['mask']
		return (img, mask)

	def get_mask(self, mask_folder):
		masks = io.imread_collection(mask_folder + '*.png').concatenate()
		mask = np.max(masks, axis=0)
		return mask


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def forward(self, input, target):
		N = target.size(0)
		smooth = 1.

		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)

		intersection = input_flat * target_flat

		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N

		return loss


class DiceBCELoss(nn.Module):
	def __init__(self):
		super(DiceBCELoss, self).__init__()
		self.bce_ratio = 0.5
		self.bce_criterion = nn.BCELoss().cuda()
		self.dice_criterion = DiceLoss().cuda()

	def forward(self, input, target):
		bce_loss = self.bce_criterion(input, target)
		dice_loss = self.dice_criterion(input, target)
		return bce_loss * self.bce_ratio + dice_loss


def compute_iou(y_true, y_pred):
	eps = 1e-6
	y_true = y_true > 0.5
	y_pred = y_pred > 0.5

	intersection = (y_true & y_pred).float().sum((1, 2, 3))
	union = (y_true | y_pred).float().sum((1, 2, 3))

	iou = (intersection + eps) / (union + eps)
	return iou


def train(loader, model, opt, criterion):
	avg_loss = 0.0
	avg_iou = 0.0

	model.train()
	for i, (imgs, masks) in enumerate(loader):
		imgs, masks = imgs.cuda(), masks.cuda()
		opt.zero_grad()
		pred = model(imgs)
		loss = criterion(pred, masks)
		iou = compute_iou(masks.data, pred.data)
		loss.backward()
		opt.step()
		avg_loss += loss.item() * len(masks)
		avg_iou += iou.sum().item()

	avg_loss /= len(loader.dataset)
	avg_iou /= len(loader.dataset)
	return avg_loss, avg_iou


def valid(loader, model, opt, criterion):
	avg_loss = 0.0
	avg_iou = 0.0
	model.eval()

	with torch.no_grad():
		for i, (imgs, masks) in enumerate(loader):
			imgs, masks = imgs.cuda(), masks.cuda()
			pred = model(imgs)
			loss = criterion(pred, masks)
			iou = compute_iou(masks.data, pred.data)
			avg_loss += loss.item() * len(masks)
			avg_iou += iou.sum().item()

	avg_loss /= len(loader.dataset)
	avg_iou /= len(loader.dataset)
	return avg_loss, avg_iou

def main(args):
	img_paths = os.listdir(args.dataset)
	train_paths, valid_paths = train_test_split(img_paths, test_size=args.split_ratio, random_state=args.random_seed)

	train_dataset = NucleiDataset(args.dataset, train_paths, A.Compose([
		A.Resize(args.img_size, args.img_size),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		A.HorizontalFlip(p=0.25),
		A.VerticalFlip(p=0.25),
		ToTensor()]))

	valid_dataset = NucleiDataset(args.dataset, valid_paths, A.Compose([
		A.Resize(args.img_size, args.img_size),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensor()]))

	trainloader = torch.utils.data.DataLoader(train_dataset,
	                                          batch_size=args.batch_size,
	                                          shuffle=True,
	                                          num_workers=0,
	                                          pin_memory=True)

	validloader = torch.utils.data.DataLoader(valid_dataset,
	                                          batch_size=args.batch_size,
	                                          shuffle=False,
	                                          num_workers=0,
	                                          pin_memory=True)


	image, mask = train_dataset.__getitem__(0)
	print(image.shape)
	print(mask.shape)
	unet = UNet().cuda()
	opt = torch.optim.Adam(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	# BCE_criterion = nn.BCELoss().cuda()
	# DICE_criterion = DiceLoss().cuda()
	DICE_BCE_criterion = DiceBCELoss().cuda()

	for i in range(1, args.n_epochs + 1):
		train_loss, train_iou = train(trainloader, unet, opt, DICE_BCE_criterion)
		valid_loss, valid_iou = valid(validloader, unet, opt, DICE_BCE_criterion)
		print('Epoch:{:02d}/{:02d}  Loss:{:.3f} IOU:{:.3f}  Valid_Loss:{:.3f} Valid_IOU:{:.3f}'.format(
			i, args.n_epochs, train_loss, train_iou, valid_loss, valid_iou))
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='D:\Dataset\data-science-bowl-2018\stage1_train')
	parser.add_argument('--img_size', type=int, default=128)
	parser.add_argument('--n_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--lr', type=float, default=3e-2)
	parser.add_argument('--weight_decay', type=float, default=1e-4)

	parser.add_argument('--split_ratio', type=float, default=0.2)

	parser.add_argument('--random_seed', type=float, default=42)

	args, _ = parser.parse_known_args()



	main(args)