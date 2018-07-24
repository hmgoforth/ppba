import sys
sys.path.insert(0, '../python') # to reach DeepLKBatch
import DeepLKBatch as dlk
import sift_ransac_homography as srh

import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import requests
from PIL import Image
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from sys import argv
import argparse
import time
from math import cos, sin, pi, sqrt, ceil
import sys
import autograd.numpy as np
from autograd import grad, jacobian
from functools import reduce
from pdb import set_trace as st
import argparse
import glob
import re
import matplotlib.pyplot as plt
from forwardAdditiveLKBA import p_to_H, H_to_p, gen_rand_p

def test_algnmnt_dir():
	parser = argparse.ArgumentParser()
	parser.add_argument("big_dir")
	parser.add_argument("small_dir")
	args = parser.parse_args()

	small_files = glob.glob(args.small_dir + 'IMG*')
	big_dir = glob.glob(args.big_dir + 'IMG*')

	small_num = [re.search('\d{4}', fn).group() for fn in small_files]
	big_files = [fn for fn in big_dir if re.search('\d{4}', fn).group() in small_num]

	big_files_srt = sorted(big_files, key=lambda x: float(re.search('\d{4}', x).group()))
	sm_files_srt = sorted(small_files, key=lambda x: float(re.search('\d{4}', x).group()))

	sz = 200

	test_img = Image.open(sm_files_srt[0])
	im_w, im_h = test_img.size
	asp_rat = im_h / im_w

	img1_btch = torch.zeros(len(sm_files_srt), 3, round(asp_rat * sz), sz)
	img2_btch = torch.zeros(len(sm_files_srt), 3, round(asp_rat * sz), sz)

	# plt.figure()

	min_scale = 0.75
	max_scale = 0.85
	angle_range = 9 # degrees
	projective_range = 0
	translation_range = 15 # pixels

	num_img = len(big_files_srt)
	# num_img = 1

	for i in range(num_img):
		print('processing ... {:d}/{:d}'.format(i + 1, len(big_files_srt)))

		img1 = Image.open(big_files_srt[i])
		img2 = Image.open(sm_files_srt[i])

		img1_rz = img1.resize((sz, round(asp_rat * sz)))
		img2_rz = img2.resize((sz, round(asp_rat * sz)))

		# plt.subplot(1,2,1)
		# plt.imshow(img1_rz)
		# plt.subplot(1,2,2)
		# plt.imshow(img2_rz)
		# plt.show()

		img1_btch[i, :, :, :] = transforms.ToTensor()(img1_rz)
		img2_btch[i, :, :, :] = transforms.ToTensor()(img2_rz)

	p_gt = torch.zeros(len(sm_files_srt), 8, 1)

	for i in range(num_img):
		p_gt[i, :, 0] = gen_rand_p(min_scale, max_scale, angle_range, projective_range, translation_range)

	img2_btch_w, _ = dlk.warp_hmg(img2_btch, p_gt)

	MODEL_PATH = "../models/conv_02_17_18_1833.pth"
	dlk_net = dlk.DeepLK(dlk.custom_net(MODEL_PATH))

	img1_btch_nmlz = Variable(dlk.normalize_img_batch(img1_btch))
	img2_w_btch_nmlz = Variable(dlk.normalize_img_batch(img2_btch_w))

	p_lk, _ = dlk_net(img1_btch_nmlz, img2_w_btch_nmlz, tol=1e-4, max_itr=75, conv_flag=0)

	plt.figure()

	img1_btch_lkw, _ = dlk.warp_hmg(img1_btch, p_lk.data)

	for i in range(num_img):

		plt.subplot(1, 3, 1)
		plt.imshow(transforms.ToPILImage()(img1_btch[i,:,:,:]))
		plt.title('ORG IMG1')

		plt.subplot(1, 3, 2)
		plt.imshow(transforms.ToPILImage()(img2_btch_w[i,:,:,:]))
		plt.title('ORG IMG2')

		plt.subplot(1, 3, 3)
		plt.imshow(transforms.ToPILImage()(img1_btch_lkw[i,:,:,:]))
		plt.title('WRP IMG1')

		plt.axis('off')

		plt.show()

	st()

def test_algnmnt():
	parser = argparse.ArgumentParser()
	parser.add_argument("img_1")
	parser.add_argument("img_2")
	args = parser.parse_args()

	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	img1 = Image.open(args.img_1)
	img1 = img1.convert('RGB')
	print(img1.mode)
	w1, h1 = img1.size

	img2 = Image.open(args.img_2)
	print(img2.mode)
	w2, h2 = img2.size

	ratio1 = h1/w1
	ratio2 = h2/w2

	ratio_both = min(ratio1, ratio2)

	img1 = img1.crop((0, 0, w1, round(ratio_both * w1)))
	img2 = img2.crop((0, 0, w2, round(ratio_both * w2)))

	sz = 534

	img1 = img1.resize((sz, round(ratio_both * sz)))
	img2 = img2.resize((sz, round(ratio_both * sz)))

	img2.show() # map template (img), M_feat
	img1.show() # template, T_feat

	img1_tens = Variable(preprocess(img1)).repeat(1,1,1,1)
	img2_tens = Variable(preprocess(img2)).repeat(1,1,1,1)

	MODEL_PATH = "../models/conv_02_17_18_1833.pth"

	dlk_net = dlk.DeepLK(dlk.custom_net(MODEL_PATH))

	img1_nmlz_tens = dlk.normalize_img_batch(img1_tens)
	img2_nmlz_tens = dlk.normalize_img_batch(img2_tens)

	p_wp, H_wp = dlk_net(img2_nmlz_tens, img1_nmlz_tens, tol=1e-4, max_itr=200, conv_flag=1)

	img1_warp, _ = dlk.warp_hmg(img1_tens, p_wp)

	transforms.ToPILImage()(img1_warp[0,:,:,:].data).show()

	# st()

	# img2 = Image.open(args.img_2).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
	# img2 = Variable(preprocess(img2.resize((sz_sm, sz_sm))))

	# transforms.ToPILImage()(img1.data).show()

if __name__ == "__main__":
	# test_algnmnt()
	test_algnmnt_dir()
