import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import normalize

def conv_block( in_ch, out_ch):
	return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
						 nn.BatchNorm2d(out_ch),
						 nn.ReLU(inplace=True),
						 nn.Conv2d(out_ch, out_ch, 3, 1, 1),
						 nn.BatchNorm2d(out_ch),
						 nn.ReLU(inplace=True),
						 )


def up_block(in_ch, out_ch):
	return nn.Sequential(
		nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
		nn.BatchNorm2d(out_ch),
		nn.ReLU(inplace=True)
	)


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		kernels = 32
		self.up_conv6 = up_block(kernels * 16, kernels * 8)
		self.conv6 = conv_block(kernels * 16, kernels * 8)
		self.up_conv7 = up_block(kernels * 8, kernels * 4)
		self.conv7 = conv_block(kernels * 8, kernels * 4)
		self.up_conv8 = up_block(kernels * 4, kernels * 2)
		self.conv8 = conv_block(kernels * 4, kernels * 2)
		self.up_conv9 = up_block(kernels * 2, kernels)
		self.conv9 = conv_block(kernels * 2, kernels)

		self.conv10 = nn.Conv2d(kernels, 1, 1)

	def forward(self, latents):
		x1, x2, x3, x4, x5 = latents
		x6 = self.up_conv6(x5)
		c6 = torch.cat((x6, x4), dim=1)
		h6 = self.conv6(c6)
		#         (128, 32, 32) -> (64, 64, 64)
		x7 = self.up_conv7(h6)
		c7 = torch.cat((x7, x3), dim=1)
		h7 = self.conv7(c7)
		#         (64, 64, 64) -> (32, 128, 128)
		x8 = self.up_conv8(h7)
		c8 = torch.cat((x8, x2), dim=1)
		h8 = self.conv8(c8)
		#         (32, 128, 128) -> (16, 256, 256)
		x9 = self.up_conv9(h8)
		c9 = torch.cat((x9, x1), dim=1)
		h9 = self.conv9(c9)
		#         (16, 256, 256) -> (1, 256, 256)
		h10 = self.conv10(h9)
		out = F.sigmoid(h10)
		return out
class projection_MLP(nn.Module):
	def __init__(self, in_dim, hidden_dim=1024, out_dim=256):
		super().__init__()
		''' page 3 baseline setting
		Projection MLP. The projection MLP (in f) has BN ap-
		plied to each fully-connected (fc) layer, including its out- 
		put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
		This MLP has 3 layers.
		'''
		self.layer1 = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(inplace=True)
		)
		self.layer2 = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(inplace=True)
		)
		self.layer3 = nn.Sequential(
			nn.Linear(hidden_dim, out_dim),
			nn.BatchNorm1d(out_dim)
		)
		self.num_layers = 3
	def set_layers(self, num_layers):
		self.num_layers = num_layers

	def forward(self, x):
		if self.num_layers == 3:
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
		elif self.num_layers == 2:
			x = self.layer1(x)
			x = self.layer3(x)
		else:
			raise Exception
		x = F.sigmoid(x)
		return x
class UNet(nn.Module):

	def __init__(self):
		super(UNet, self).__init__()

		kernels = 32

		self.conv1 = self.conv_block(3, kernels)
		self.conv2 = self.conv_block(kernels, kernels * 2)
		self.conv3 = self.conv_block(kernels * 2, kernels * 4)
		self.conv4 = self.conv_block(kernels * 4, kernels * 8)
		self.conv5 = self.conv_block(kernels * 8, kernels * 16)

		self.down_pooling = nn.MaxPool2d(2)

		self.fore_decoder = Decoder()
		self.back_decoder = Decoder()
		self.add_module('Foreground_decoder', self.fore_decoder)
		self.add_module('background_decoder', self.back_decoder)
		# self.up_conv6 = self.up_block(kernels * 16, kernels * 8)
		# self.conv6 = self.conv_block(kernels * 16, kernels * 8)
		# self.up_conv7 = self.up_block(kernels * 8, kernels * 4)
		# self.conv7 = self.conv_block(kernels * 8, kernels * 4)
		# self.up_conv8 = self.up_block(kernels * 4, kernels * 2)
		# self.conv8 = self.conv_block(kernels * 4, kernels * 2)
		# self.up_conv9 = self.up_block(kernels * 2, kernels)
		# self.conv9 = self.conv_block(kernels * 2, kernels)
		#
		# self.conv10 = nn.Conv2d(kernels, 1, 1)
		self.proj5 = projection_MLP(64, 1024, 64)
		self._initialize_weights()

	def up_block(self, in_ch, out_ch):
		return nn.Sequential(
			nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def conv_block(self, in_ch, out_ch):
		return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
							 nn.BatchNorm2d(out_ch),
							 nn.ReLU(inplace=True),
							 nn.Conv2d(out_ch, out_ch, 3, 1, 1),
							 nn.BatchNorm2d(out_ch),
							 nn.ReLU(inplace=True),
							 )

	def forward(self, x):
		#         Encoder Part
		#         (3, 256, 256) -> (16, 128, 128)
		x1 = self.conv1(x)
		h1 = self.down_pooling(x1)
		#         (16, 128, 128) -> (32, 64, 64)
		x2 = self.conv2(h1)
		h2 = self.down_pooling(x2)
		#         (32, 64, 64) -> (64, 32, 32)
		x3 = self.conv3(h2)
		h3 = self.down_pooling(x3)
		#         (64, 32, 32) -> (128, 16, 16)
		x4 = self.conv4(h3)
		h4 = self.down_pooling(x4)

		#         (128, 16, 16) -> (256, 16, 16)
		# 512x8x8
		x5 = self.conv5(h4)
		size_x5 = x5.size()
		x5 = x5.view(-1, size_x5[-1]**2)
		x5p = self.proj5(x5.detach()).reshape(*size_x5)
		x5 = x5.reshape(*size_x5)
		#         Decoder Part
		#         (256, 16, 16) -> (128, 32, 32)
		# x6 = self.up_conv6(x5)
		# c6 = torch.cat((x6, x4), dim=1)
		# h6 = self.conv6(c6)
		# #         (128, 32, 32) -> (64, 64, 64)
		# x7 = self.up_conv7(h6)
		# c7 = torch.cat((x7, x3), dim=1)
		# h7 = self.conv7(c7)
		# #         (64, 64, 64) -> (32, 128, 128)
		# x8 = self.up_conv8(h7)
		# c8 = torch.cat((x8, x2), dim=1)
		# h8 = self.conv8(c8)
		# #         (32, 128, 128) -> (16, 256, 256)
		# x9 = self.up_conv9(h8)
		# c9 = torch.cat((x9, x1), dim=1)
		# h9 = self.conv9(c9)
		# #         (16, 256, 256) -> (1, 256, 256)
		# h10 = self.conv10(h9)
		# out = F.sigmoid(h10)
		latents = (x1, x2, x3, x4, x5)
		fore_out = self.fore_decoder(latents)
		size = fore_out.size()
		# fore_out = fore_out.view(size[0], -1)
		# back_out = self.proj5(fore_out.detach()).flatten()
		fore_out = fore_out.flatten()
		# back_out = self.back_decoder((x1, x2, x3, x4, x5p)).flatten()
		back_out = self.fore_decoder((x1, x2, x3, x4, x5p)).flatten().detach()
		indices = (fore_out < back_out).nonzero()
		ph_out = fore_out + (1 - back_out)
		ph_out[indices] = ph_out[indices] - fore_out[indices]
		back_indices = (fore_out > back_out).nonzero()
		ph_out[back_indices] = ph_out[back_indices] + back_out[back_indices] - 1
		# print('---------------------')
		# print(indices.size())
		# print(fore_out.size())
		ph_out = ph_out.reshape(*size)
		return ph_out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)
