import sys
sys.path.insert(0, '../python') # to reach DeepLKBatch
import DeepLKBatch as dlk
import sift_ransac_homography as srh
import ppba

import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import requests
from PIL import Image
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

def fwdAddLK(img, tmpl, tol):
	batch_size, k, h, w = img.size()

	p = torch.zeros(1, 8, 1)
	dp = torch.zeros(1, 8, 1)

	crit = 0

	itn = 1

	grad_func = dlk.GradientBatch()
	inv_func = dlk.InverseBatch()

	img_gradx, img_grady = grad_func(img)

	while (itn == 1) or (crit > tol):
		img_w, mask_w = dlk.warp_hmg(img, p)
		mask_w.unsqueeze_(1)
		mask_w = mask_w.repeat(1, k, 1, 1)
		tmpl_mask = tmpl.mul(mask_w)

		res = tmpl_mask - img_w
		res = res.view(batch_size, k * h * w, 1)

		img_gradx_w, _ = dlk.warp_hmg(img_gradx, p)
		img_grady_w, _ = dlk.warp_hmg(img_grady, p)

		img_gradx_w = img_gradx_w.view(batch_size, k * h * w, 1)
		img_grady_w = img_grady_w.view(batch_size, k * h * w, 1)

		x = torch.arange(w)
		y = torch.arange(h)
		X, Y = dlk.meshgrid(x, y)
		H = dlk.param_to_H(p)
		xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), torch.ones(1, X.numel())), 0)
		xy = xy.repeat(batch_size, 1, 1)
		xy_warp = H.bmm(xy)

		# extract warped X and Y, normalizing the homog coordinates
		X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
		Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]

		X_warp = X_warp.view(X_warp.numel(), 1)
		Y_warp = Y_warp.view(Y_warp.numel(), 1)

		X_warp = X_warp.repeat(batch_size, k, 1)
		Y_warp = Y_warp.repeat(batch_size, k, 1)

		dIdp = torch.cat((
			X_warp.mul(img_gradx_w), 
			Y_warp.mul(img_gradx_w),
			img_gradx_w,
			X_warp.mul(img_grady_w),
			Y_warp.mul(img_grady_w),
			img_grady_w,
			-X_warp.mul(X_warp).mul(img_gradx_w) - X_warp.mul(Y_warp).mul(img_grady_w),
			-X_warp.mul(Y_warp).mul(img_gradx_w) - Y_warp.mul(Y_warp).mul(img_grady_w)),2)

		dIdp_t = dIdp.transpose(1, 2)

		invH = inv_func(dIdp_t.bmm(dIdp))

		dp = invH.bmm(dIdp_t.bmm(res))

		crit = float(dp.norm(p=2,dim=1,keepdim=True).max())

		p = p + dp

		itn = itn + 1

		print('itn: {:d}, crit: {:.2f}'.format(itn, crit))

	print('finished at iteration ', itn)

	return p

def fwdAddLKParamComp(img, tmpl, tol, mot_par, q_ind):
	batch_size, k, h, w = img.size()

	dq = torch.zeros(batch_size, 8, 1)

	crit = 0

	itn = 1

	grad_func = dlk.GradientBatch()
	inv_func = dlk.InverseBatchFun

	img_gradx, img_grady = grad_func(img)

	while (itn == 1) or (crit > tol):
		H_tot = reduce(np.dot, np.flip(dlk.param_to_H(mot_par).numpy(), axis=0))
		H_tot = np.expand_dims(H_tot, 0)
		pq = dlk.H_to_param(torch.from_numpy(H_tot))

		img_w, mask_w = dlk.warp_hmg(img, pq)
		mask_w.unsqueeze_(1)
		mask_w = mask_w.repeat(1, k, 1, 1)
		tmpl_mask = tmpl.mul(mask_w)

		res = tmpl_mask - img_w
		res = res.view(batch_size, k * h * w, 1)

		#### - dp/dq

		dpdq = torch.zeros(batch_size, 8, 8)
		mot_par_np = np.squeeze(mot_par.numpy())

		def compute_pq(q_param):
			q_mat = ppba.p_to_H(q_param)
			p_mat = ppba.p_to_H(mot_par_np)

			p_mat = np.concatenate((
				p_mat[0:q_ind, :, :],
				q_mat,
				p_mat[q_ind + 1:, :, :]
				), axis=0)

			p_mat_reduce = np.eye(3)

			for i in range(p_mat.shape[0]):
				p_mat_reduce = np.dot(p_mat[i], p_mat_reduce)

			p_par = ppba.H_to_p(p_mat_reduce)
			return p_par

		# using auto-grad library for computing jacobian (8x8)
		grad_pq = jacobian(compute_pq)

		q_par = np.transpose(mot_par[q_ind, :, :].numpy())

		st()
		# evaluate 8x8 jacobian and store it
		dpdq[0, :, :] = \
			torch.from_numpy(grad_pq(q_par).squeeze(axis=0).squeeze(axis=1))

		# print(dpdq[0, :, :])

		#### - dI/dw and dW/dp

		img_gradx_w, _ = dlk.warp_hmg(img_gradx, pq)
		img_grady_w, _ = dlk.warp_hmg(img_grady, pq)

		img_gradx_w = img_gradx_w.view(batch_size, k * h * w, 1)
		img_grady_w = img_grady_w.view(batch_size, k * h * w, 1)

		x = torch.arange(w)
		y = torch.arange(h)
		X, Y = dlk.meshgrid(x, y)
		H_pq = dlk.param_to_H(pq)
		xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), torch.ones(1, X.numel())), 0)
		xy = xy.repeat(batch_size, 1, 1)
		xy_warp = H_pq.bmm(xy)

		# extract warped X and Y, normalizing the homog coordinates
		X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
		Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]

		X_warp = X_warp.view(X_warp.numel(), 1)
		Y_warp = Y_warp.view(Y_warp.numel(), 1)

		X_warp = X_warp.repeat(batch_size, k, 1)
		Y_warp = Y_warp.repeat(batch_size, k, 1)

		dIdp = torch.cat((
			X_warp.mul(img_gradx_w), 
			Y_warp.mul(img_gradx_w),
			img_gradx_w,
			X_warp.mul(img_grady_w),
			Y_warp.mul(img_grady_w),
			img_grady_w,
			-X_warp.mul(X_warp).mul(img_gradx_w) - X_warp.mul(Y_warp).mul(img_grady_w),
			-X_warp.mul(Y_warp).mul(img_gradx_w) - Y_warp.mul(Y_warp).mul(img_grady_w)),2)

		#### - dIdq

		dIdq = dIdp.bmm(dpdq)

		#### - compute dq

		dIdq_t = dIdq.transpose(1, 2)

		invH = inv_func(dIdq_t.bmm(dIdq))

		dq = invH.bmm(dIdq_t.bmm(res))

		crit = float(dq.norm(p=2,dim=1,keepdim=True).max())

		mot_par[q_ind, :, :] = mot_par[q_ind, :, :] + dq[0, :, :]

		itn = itn + 1

		print('itn: {:d}, crit: {:.2f}'.format(itn, crit))

	print('finished at iteration ', itn)

	return mot_par


def test_fwdAddLK():
	parser = argparse.ArgumentParser()
	parser.add_argument("img")

	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	args = parser.parse_args()

	img = Image.open(args.img)
	img_w, img_h = img.size
	aspect = img_w / img_h
	img_h_sm = 200
	img_w_sm = ceil(aspect * img_h_sm)

	img_tens = preprocess(img.resize((img_w_sm, img_h_sm)))
	img_tens = torch.unsqueeze(img_tens, 0)

	p_gt = torch.FloatTensor([[
		[0],
		[0],
		[10],
		[0],
		[0],
		[10],
		[0],
		[0]
		]])

	img_tens_w, mask_tens_w = dlk.warp_hmg(img_tens, p_gt)

	# transforms.ToPILImage()(img_tens[0,:,:,:]).show()
	# transforms.ToPILImage()(img_tens_w[0,:,:,:]).show()

	p_falk = fwdAddLK(img_tens, img_tens_w, 1e-3)

	img_tens_falk_w, _ = dlk.warp_hmg(img_tens, p_falk)

	transforms.ToPILImage()(img_tens_falk_w[0,:,:,:]).show()

def test_fwdAddLKParamComp():
	parser = argparse.ArgumentParser()
	parser.add_argument("img")

	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	args = parser.parse_args()

	img = Image.open(args.img)
	img_w, img_h = img.size
	aspect = img_w / img_h
	img_h_sm = 200
	img_w_sm = ceil(aspect * img_h_sm)

	img_tens = preprocess(img.resize((img_w_sm, img_h_sm)))
	img_tens = torch.unsqueeze(img_tens, 0)

	p_gt = torch.FloatTensor([[
		[-.1],[0],[10],[0],[-.1],[10],[0],[0]
		]])
	
	img_tens_w, mask_tens_w = dlk.warp_hmg(img_tens, p_gt)

	# transforms.ToPILImage()(img_tens[0,:,:,:]).show()
	# transforms.ToPILImage()(img_tens_w[0,:,:,:]).show()

	# p_falk = fwdAddLK(img_tens, img_tens_w, 1e-3)

	mot_par = torch.FloatTensor([
		[[-.05],[0.01],[5],[0],[-.05],[5],[0.0006],[0.0005]],
		[[0],[0],[2],[0],[0],[2],[0],[0]],
		[[0],[0],[0],[0],[0],[0],[0],[0]],
		])

	q_ind = 2

	mot_par_falkpc = fwdAddLKParamComp(img_tens, img_tens_w, 1e-3, mot_par, q_ind)

	print(mot_par_falkpc)

	print(reduce(np.dot, np.flip(dlk.param_to_H(mot_par).numpy(), axis=0)))

	# img_tens_falk_w, _ = dlk.warp_hmg(img_tens, p_falk)

	# transforms.ToPILImage()(img_tens_falk_w[0,:,:,:]).show()

if __name__ == "__main__":
	# test_fwdAddLK()
	test_fwdAddLKParamComp()

