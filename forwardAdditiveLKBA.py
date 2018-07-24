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
import random
import sys
import autograd.numpy as np
from autograd import grad, jacobian
from functools import reduce
from pdb import set_trace as st
import argparse
import glob
import matplotlib.pyplot as plt

# example call: python3 forwardAdditiveLKBA.py seq ../test_img/aerial-img.jpg 
# crops aerial-img.jpg into several random crops that are composed as a sequence, then aligns them

def optimize(I, P, T, V, tol, coeff_mult):
	'''
	Args:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template

	Returns:
		P_opt: Bundle adjusted warp parameter sequence, 3D numpy array, num_frame x 8 x 1
	'''

	num_frame = I.shape[0]

	def size_fn(a):
		return a.size

	size_fn_vec = np.vectorize(size_fn)

	V_sz = size_fn_vec(V)
	sigma = V_sz.sum()

	_, img_c, img_h, img_w = I.shape

	I_gradx = np.gradient(I, axis=3)
	I_grady = np.gradient(I, axis=2)

	dP = np.zeros([num_frame - 1, 8, 1])
	crit = 0

	itn = 1

	coeff = 1

	while ((crit > tol) or (itn == 1)) and (itn < 200):
		P_fk = compute_Pfk(P, T, V, V_sz)

		r_ba = compute_ri(I, T, V, P_fk, V_sz)

		J_ba = compute_Ji(I_gradx, I_grady, P, T, V, P_fk, V_sz)

		J_ba_trans = J_ba.swapaxes(2,3)

		J_ba_trans_J_ba = np.matmul(J_ba_trans, J_ba) 

		Hess = np.sum(J_ba_trans_J_ba, axis=1)

		invHess = np.linalg.inv(Hess)

		J_ba_trans_r_ba = np.matmul(J_ba_trans, r_ba)

		J_ba_trans_r_ba_sum = np.sum(J_ba_trans_r_ba, axis=1)

		dP = np.matmul(invHess, J_ba_trans_r_ba_sum)

		P = P + dP * coeff

		dp_norm = np.linalg.norm(dP, ord=2, axis=1)
		crit = np.amax(dp_norm)

		if (itn % 10 == 0):
			coeff = coeff * coeff_mult

		print('itn:  {:d}'.format(itn))
		print('crit: {:.5f}'.format(crit))

		# print('coeff: {:.2f}'.format(coeff))
		# print(P)
		# print(dP)

		itn = itn + 1

	return P

def optimize_wmap(I, P, T, V, P_init, M_feat, T_feat, tol):
	'''
	Args:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		M: Map image sequence, 4D numpy array, k x C x H x W
		T: Numpy vector containing indices from I which are templates, length k
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_init: Absolute warp parameters relating map to first frame, 1 x 8 x 1
		M_feat: Sequence of deep feature map images, 4D numpy array, k x C x H x W
		T_feat: Template images from I deep feature extractions, 4D numpy array, k x Cf x H x W
	
	Returns:
		P_opt: Bundle adjusted warp parameter sequence, 3D numpy array, num_frame x 8 x 1
	'''
	
	'''
	dP = zeros
	Mf_gradx, Mf_grady = gradient(M_feat)
	P_mk0 = compute_Pmk(P_init, P, T)

	while (crit > tol):
		P_fk = compute_Pfk(P, T, V, V_sz)
		P_mk = compute_Pmk(P_init, P, T)

		r_i = compute_ri(I, T, V, P_fk, V_sz)
		r_m = compute_rm(M_feat, T_feat, P_mk0, P_mk)

		
	'''

def compute_rm(M_feat, T_feat, P_mk0, P_mk):
	'''
	Args:
		M_feat: Sequence of deep feature map images, 4D numpy array, k x Cf x H x W
		T_feat: Template images from I deep feature extractions, 4D numpy array, k x Cf x H x W
		P_mk0: Numpy array, initial warp parameters from map images to templates, k x 8 x 1
		P_mk: Numpy array, current warp parameters from map images to templates, k x 8 x 1

	Returns:
		rm: Numpy array, residuals of map images with templates, num_frame - 1 (duplicated dim.) x k x (Cf x H x W) x 1
	'''

	'''
	warp M_feat with P_mk.bmm(inv(P_mk0))
	rm = T_feat - M_feat
	reshape, tile rm and return
	'''

def compute_ri(I, T, V, P_fk, V_sz):
	'''
	Args:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		r_ba: Numpy array, residuals of images with templates, num_frame - 1 x sigma x (C x H x W) x 1
	'''

	sigma = V_sz.sum()

	num_frame, img_c, img_h, img_w = I.shape

	T_f = np.repeat(I[T, :, :, :], V_sz, axis=0)

	I_f = np.zeros((sigma, img_c, img_h, img_w))

	I_f_ind = 0
	for i in range(len(V)):
		for f in V[i]:
			I_f[I_f_ind, :, :, :] = I[f, :, :, :]
			I_f_ind = I_f_ind + 1

	P_kf_tens = torch.from_numpy(P_fk).float()
	I_f_tens = torch.from_numpy(I_f).float()

	I_f_warp_tens, I_f_mask_tens = dlk.warp_hmg(I_f_tens, P_kf_tens)

	I_f_warp = I_f_warp_tens.numpy()
	I_f_mask = I_f_mask_tens.numpy()

	I_f_mask = np.tile(np.expand_dims(I_f_mask, 1), (1, img_c, 1, 1))

	T_f_mask = np.multiply(T_f, I_f_mask)

	r_ba =  T_f_mask - I_f_warp

	r_ba = r_ba.reshape((sigma, img_c * img_h * img_w, 1))

	r_ba = np.tile(r_ba, (num_frame - 1, 1, 1, 1))

	return r_ba

def compute_Ji(I_gradx, I_grady, P, T, V, P_fk, V_sz):
	'''
	Args:
		I_gradx: x-gradient of image sequence, 4D numpy array, num_frame x C x H x W
		I_grady: y-gradient of image sequence, 4D numpy array, num_frame X C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame - 1 x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		Ji: Numpy array, residuals of images with templates, num_frame - 1 x sigma x (C x H x W) x 8
	'''

	sigma = V_sz.sum()
	num_frame, img_c, img_h, img_w = I_gradx.shape

	I_gradx_f = np.zeros((sigma, img_c, img_h, img_w))
	I_grady_f = np.zeros((sigma, img_c, img_h, img_w))

	I_f_ind = 0
	for i in range(len(V)):
		for f in V[i]:
			I_gradx_f[I_f_ind, :, :, :] = I_gradx[f, :, :, :]
			I_grady_f[I_f_ind, :, :, :] = I_grady[f, :, :, :]
			I_f_ind = I_f_ind + 1

	gradI_warpjac = compute_gradI_warpjac(I_gradx_f, I_grady_f, P_fk)
	gradI_warpjac = np.tile(gradI_warpjac, (num_frame - 1, 1, 1, 1))

	gradPfk_PF = np.zeros((num_frame - 1, sigma, 8, 8))

	for F in range(num_frame - 1):
		frame_id = 0 # iterates through sigma axis
		k_ind = 0
		for k in T:
			for f in V[k_ind]:
				# f and k index images I, F indexes P
				if (f < k) and (F < k) and (f <= F):
					H_fk = p_to_H(P[f : k, :, :])

					H_F_ind = F - f

					def Pfk(P_F):
						H_F = p_to_H(P_F)

						H_fk_temp = np.concatenate((
							H_fk[0:H_F_ind, :, :],
							H_F,
							H_fk[H_F_ind + 1:, :, :]
							), axis=0)

						H_fk_mat = np.eye(3)

						for i in range(H_fk_temp.shape[0]):
							H_fk_mat = np.dot(H_fk_temp[i], H_fk_mat)

						P_fk = H_to_p(H_fk_mat)
						P_fk = P_fk.squeeze(0)
						return P_fk

					# using auto-grad library for computing jacobian (8x8)
					grad_P_fk = jacobian(Pfk)

					P_F = H_to_p(H_fk[H_F_ind, :, :])
					P_F = P_F.squeeze(0)

					# evaluate 8x8 jacobian and store it
					gradPfk_PF[F, frame_id, :, :] = \
						grad_P_fk(P_F).squeeze(axis=1).squeeze(axis=2)

				elif (f > k) and (F >= k) and (f > F):
					H_kf = p_to_H(P[k : f, :, :])

					H_F_ind = F - k

					def Pfk(P_F):
						H_F = p_to_H(P_F)

						H_kf_temp = np.concatenate((
							H_kf[0:H_F_ind, : , :],
							H_F,
							H_kf[H_F_ind + 1:, :, :]
							), axis=0)

						H_kf_temp_inv = np.linalg.inv(H_kf_temp)

						H_fk_mat = np.eye(3)

						for i in range(H_kf_temp_inv.shape[0]):
							H_fk_mat = np.dot(H_fk_mat, H_kf_temp_inv[i])

						P_fk = H_to_p(H_fk_mat)
						P_fk = P_fk.squeeze(0)
						return P_fk

					grad_P_fk = jacobian(Pfk)

					P_F = H_to_p(H_kf[H_F_ind, :, :])
					P_F = P_F.squeeze(0)

					gradPfk_PF[F, frame_id, :, :] = \
						grad_P_fk(P_F).squeeze(axis=1).squeeze(axis=2)

				else:
					gradPfk_PF[F, frame_id, :, :] = np.zeros((8,8))

				frame_id = frame_id + 1

			k_ind = k_ind + 1

	J = np.matmul(gradI_warpjac, gradPfk_PF)

	return J

def compute_Pmk(P_init, P, T):
	'''
	Args:
		P_init: Absolute warp parameters relating map to first frame, 1 x 8 x 1
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		T: Numpy vector containing indices from I which are templates, length k

	Returns:
		P_mk: Warp parameters from map to templates, k x 8 x 1
	'''

	'''
	create P_mk using reduce(dot, [P[0:T], P_init]) etc. etc.
	return P_mk
	'''

def compute_Pfk(P, T, V, V_sz):
	'''
	Args:
		P: Warp parameter sequence, 3D numpy array, num_frame - 1 x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		Pfk: Numpy array, warp parameters from images to templates, sigma x 8 x 1
	'''

	sigma = V_sz.sum()

	P_fk_all = np.zeros((sigma, 8, 1))

	frame_id = 0 # iterates through sigma axis
	k_ind = 0
	for k in T:
		for f in V[k_ind]:
			# f and k index images I, F indexes P
			if (f < k):
				H_fk = p_to_H(P[f : k, :, :])

				H_fk_mat = np.eye(3)

				for i in range(H_fk.shape[0]):
					H_fk_mat = np.dot(H_fk[i], H_fk_mat)

				P_fk = H_to_p(H_fk_mat)

			else:
				H_kf = p_to_H(P[k : f, :, :])

				H_kf_inv = np.linalg.inv(H_kf)

				H_fk_mat = np.eye(3)

				for i in range(H_kf_inv.shape[0]):
					H_fk_mat = np.dot(H_fk_mat, H_kf_inv[i])

				P_fk = H_to_p(H_fk_mat)

			P_fk_all[frame_id, :, :] = P_fk
			frame_id = frame_id + 1

		k_ind = k_ind + 1

	return P_fk_all

def compute_gradI_warpjac(I_gradx, I_grady, P_fk):
	'''
	Args:
		I_gradx: x-gradient of image sequence with duplicates representing vis. neighborhoods, 4D numpy array, sigma x C x H x W
		I_grady: y-gradient of image sequence with duplicates representing vis. neighborhoods, 4D numpy array, sigma X C x H x W
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1

	Returns:
		gradI_warpjac: Numpy array, dI/dW * dW/dPfk, sigma x (C x H x W) x 8
	'''

	batch_size, c, h, w = I_gradx.shape

	P_fk_tens = torch.from_numpy(P_fk).float()
	I_gradx_tens = torch.from_numpy(I_gradx).float()
	I_grady_tens = torch.from_numpy(I_grady).float()

	img_gradx_w, _ = dlk.warp_hmg(I_gradx_tens, P_fk_tens)
	img_grady_w, _ = dlk.warp_hmg(I_grady_tens, P_fk_tens)

	img_gradx_w = img_gradx_w.view(batch_size, c * h * w, 1)
	img_grady_w = img_grady_w.view(batch_size, c * h * w, 1)

	x = torch.arange(w)
	y = torch.arange(h)
	X, Y = dlk.meshgrid(x, y)
	H_pq = dlk.param_to_H(P_fk_tens)
	xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), torch.ones(1, X.numel())), 0)
	xy = xy.repeat(batch_size, 1, 1)
	xy_warp = H_pq.bmm(xy)

	# extract warped X and Y, normalizing the homog coordinates
	X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
	Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]

	X_warp = X_warp.unsqueeze(dim=2)
	Y_warp = Y_warp.unsqueeze(dim=2)

	X_warp = X_warp.repeat(1, c, 1)
	Y_warp = Y_warp.repeat(1, c, 1)

	gradI_warpjac = torch.cat((
		X_warp.mul(img_gradx_w), 
		Y_warp.mul(img_gradx_w),
		img_gradx_w,
		X_warp.mul(img_grady_w),
		Y_warp.mul(img_grady_w),
		img_grady_w,
		-X_warp.mul(X_warp).mul(img_gradx_w) - X_warp.mul(Y_warp).mul(img_grady_w),
		-X_warp.mul(Y_warp).mul(img_gradx_w) - Y_warp.mul(Y_warp).mul(img_grady_w)),2)

	return gradI_warpjac.numpy()

def p_to_H(p):
	if len(p.shape) < 3:
		p = np.expand_dims(p, 0)

	batch_sz, _, _ = p.shape
	
	batch_one = np.zeros((batch_sz, 1, 1))
	
	p_ = np.concatenate((p, batch_one), axis=1)

	iden = np.tile(np.eye(3), (batch_sz, 1, 1))

	H = np.reshape(p_, (batch_sz, 3, 3)) + iden

	return H

def H_to_p(H):
	if len(H.shape) < 3:
		H = np.expand_dims(H, 0)

	batch_sz, _, _ = H.shape

	H = H / np.reshape(H[:, 2, 2], (-1, 1, 1))

	iden = np.tile(np.eye(3), (batch_sz, 1, 1))

	p = np.reshape(H - iden, (batch_sz, 9, 1))

	p = p[:, 0:-1, :]

	return p

def test_2img(args):
	print("Running 2 image test ...")
	img_sz = 200
	img_tens = open_img_as_tens(args.img, img_sz)

	# size scale range
	min_scale = 0.9
	max_scale = 1.1

	# rotation range (-angle_range, angle_range)
	angle_range = 4 # degrees

	# projective variables (p7, p8)
	projective_range = 0

	# translation (p3, p6)
	translation_range = 8 # pixels

	# p_gt = torch.FloatTensor([[
	# 	[0],[0],[10],[0],[0],[10],[0],[0]
	# 	]])

	p_gt = torch.zeros(1, 8, 1)

	p_gt[0, :, 0] = gen_rand_p(min_scale, max_scale, angle_range, projective_range, translation_range)
	
	img_tens_w, mask_tens_w = dlk.warp_hmg(img_tens, p_gt)

	I = torch.cat((img_tens, img_tens_w), 0).numpy()

	P = np.array([
		[[0], [0], [0], [0], [0], [0], [0], [0]]
		])

	T = np.array([1])

	V = np.array(
		[
		 np.array([0, 1]),
		 np.array([0])
		]
	)

	V = np.delete(V, 0)

	tol = 1e-4

	P_opt = optimize(I, P, T, V, tol, 1)
		# I: Image sequence, 4D numpy array, num_frame x C x H x W
		# P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		# T: Numpy vector containing indices from I which are templates
		# V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template

	print(P_opt)

	# transforms.ToPILImage()(img_tens[0,:,:,:]).show()
	# transforms.ToPILImage()(img_tens_w[0,:,:,:]).show()

	loss = corner_loss(P_opt, p_gt, img_sz)
	print('loss: {:.2f}'.format(loss))

	plt.figure()
	plt.subplot(2, 2, 1)
	plt.imshow(plt_tens_to_np(img_tens[0,:,:,:]))
	plt.title('Original')

	plt.subplot(2, 2, 2)
	plt.imshow(plt_tens_to_np(img_tens_w[0,:,:,:]))
	plt.title('Warped')

	img_tens_uw, _ = dlk.warp_hmg(img_tens, torch.from_numpy(P_opt).float())

	plt.subplot(2, 2, 4)
	plt.imshow(plt_tens_to_np(img_tens_uw[0,:,:,:]))
	plt.title('LK Warped')

	plt.show()

def plt_axis_match_tens(tens):
	np_img = tens.numpy()
	temp = np.swapaxes(np_img, 0, 2)
	temp = np.swapaxes(temp, 0, 1)
	return temp

def plt_axis_match_np(np_img):
	temp = np.swapaxes(np_img, 0, 2)
	temp = np.swapaxes(temp, 0, 1)
	return temp

def test_3img(args):
	print("Running 3 image test ...")

	img_tens = open_img_as_tens(args.img, 200)

	p_gt = torch.FloatTensor([
		[[0.12],[0],[9],[0],[0.12],[10],[0],[0]],
		[[0.09],[0],[10],[0],[0.09],[7],[0],[0]]
	])
	
	I = warp_seq(img_tens, p_gt)

	P = np.array([
		[[0], [0], [5], [0], [0], [5], [0], [0]],
		[[0], [0], [5], [0], [0], [5], [0], [0]]
	])

	T = np.array([2])

	V = np.array(
		[
		 np.array([0]),
		 np.array([0, 1])
		]
	)		

	V = np.delete(V, 0)

	tol = 1e-4

	P_opt = optimize(I, P, T, V, tol)
		# I: Image sequence, 4D numpy array, num_frame x C x H x W
		# P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		# T: Numpy vector containing indices from I which are templates
		# V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template

	print(P_opt)


def test_multi_img(args):
	print("Running multi image test ...")

	img_tens = open_img_as_tens(args.img, 150)

	p_gt = torch.FloatTensor([
		[[0],[0],[5],[0],[0],[5],[0],[0]],
		[[0],[0],[5],[0],[0],[5],[0],[0]],
		[[0],[0],[5],[0],[0],[5],[0],[0]],
		[[0],[0],[5],[0],[0],[5],[0],[0]]
	])

	I = warp_seq(img_tens, p_gt)

	P = np.array([
		[[0], [0], [0], [0], [0], [0], [0], [0]],
		[[0], [0], [0], [0], [0], [0], [0], [0]],
		[[0], [0], [0], [0], [0], [0], [0], [0]],
		[[0], [0], [0], [0], [0], [0], [0], [0]]
	])

	T = np.array([0, 2, 4])

	V = np.array(
		[
		 np.array([0]),
		 np.array([1]),
		 np.array([1, 3]),
		 np.array([3])
		]
	)		

	V = np.delete(V, 0)

	tol = 1e-4

	P_opt = optimize(I, P, T, V, tol)
		# I: Image sequence, 4D numpy array, num_frame x C x H x W
		# P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		# T: Numpy vector containing indices from I which are templates
		# V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template

	print(P_opt)

def test_img_seq(args):

	img_sz = 200
	img_tens = open_img_as_tens(args.img, img_sz)
	num_seq = 3

	# size scale range
	min_scale = 0.9
	max_scale = 1.1

	# rotation range (-angle_range, angle_range)
	angle_range = 4 # degrees

	# projective variables (p7, p8)
	projective_range = 0

	# translation (p3, p6)
	translation_range = 8 # pixels

	p_gt = torch.zeros(num_seq, 8, 1)

	p_gt[0, :, :] = torch.FloatTensor([[-0.4,0,0,0,-0.4,0,0,0]]).t()

	for i in range(p_gt.shape[0] - 1):
		p_gt[i + 1, :, :] = gen_rand_p(min_scale, max_scale, angle_range, projective_range, translation_range)

	p_gt_comp = np.zeros((num_seq, 8, 1))

	for i in range(p_gt_comp.shape[0]):
		H_gt = p_to_H(p_gt[0 : i + 1, :, :])
		H_gt_comp_i = reduce(np.dot, np.flip(H_gt, axis=0))
		p_gt_comp_i = H_to_p(H_gt_comp_i)
		p_gt_comp[i] = p_gt_comp_i

	I = torch.zeros(img_tens.shape)
	I = np.tile(I, (num_seq, 1, 1, 1))

	for i in range(p_gt_comp.shape[0]):
		img_tens_w, _ = dlk.warp_hmg(img_tens, torch.from_numpy(p_gt_comp[i : i + 1, :, :]).float())
		I[i, :, :, :] = img_tens_w

	P = np.zeros((num_seq - 1, 8, 1))

	T = np.array([num_seq - 1])

	V = np.array(
		[
		 np.arange(50),
		 np.arange(num_seq - 1),
		]
	)		

	V = np.delete(V, 0)

	# T = np.array([2, 5, 8])

	# V = np.array(
	# 	[
	# 	 np.arange(50),
	# 	 np.array([0, 1, 3, 4]),
	# 	 np.array([3, 4, 6, 7]),
	# 	 np.array([6, 7, 9, 10, 11])
	# 	]
	# )	
	# V = np.delete(V, 0)

	tol = 1e-4

	P_opt_dep = optimize(I, P, T, V, tol, 1)

	# T = np.array([1])

	# V = np.array(
	# 	[
	# 	 np.arange(50),
	# 	 np.array([0, 2]),
	# 	]
	# )		

	# V = np.delete(V, 0)

	# T = np.array([1, 3, 5, 7, 9, 10])

	# V = np.array(
	# 	[
	# 	 np.arange(50),
	# 	 np.array([0, 2]),
	# 	 np.array([2, 4]),
	# 	 np.array([4, 6]),
	# 	 np.array([6, 8]),
	# 	 np.array([8, 10]),
	# 	 np.array([11])
	# 	]
	# )
	# V = np.delete(V, 0)

	# P_opt_odom = optimize(I, P, T, V, tol, 1)

	print('')
	print('Corners Dep Calc:')
	loss_dep = corner_loss(P_opt_dep, p_gt[1:, :, :], img_sz)

	print('')
	print('Corners No-op Calc:')
	loss_noop = corner_loss(P, p_gt[1:, :, :], img_sz)

	# print('')
	# print('Corner Odom Calc:')
	# loss_odom = corner_loss(P_opt_odom, p_gt[1:, :, :], img_sz)

	print('')
	print('loss noop: {:.3f}'.format(loss_noop))

	print('')
	print('loss dep: {:.3f}'.format(loss_dep))

	# print('')
	# print('loss odom: {:.3f}'.format(loss_odom))

	print('')
	print('P_opt_dep:')
	print(P_opt_dep)

	# print('P_opt_odom:')
	# print(P_opt_odom)

	print('')
	print('P_gt:')
	print(p_gt[1:,:,:].numpy())

	plt.figure()

	for i in range(num_seq):
		plt.subplot(2, num_seq, i + 1)

		plt.imshow(plt_axis_match_np(I[i, :, :, :]))
		plt.title('I[{:d}]'.format(i))

		plt.subplot(2, num_seq, i + 1 + num_seq)
		plt.title('I_LK[{:d}]'.format(i))

		if (i == 0):
			I_w = torch.from_numpy(I[i : i + 1, :, :, :])
		else:
			I_w, _ = dlk.warp_hmg(I_w, torch.from_numpy(P_opt_dep[i - 1 : i, :, :]).float())

		plt.imshow(plt_axis_match_tens(I_w[0, :, :, :]))

	plt.show()

def corner_loss(p, p_gt, img_sz):
	# p [in, torch tensor] : batch of regressed warp parameters
	# p_gt [in, torch tensor] : batch of gt warp parameters
	# loss [out, float] : sum of corner loss over minibatch

	batch_size, _, _ = p.shape

	# compute corner loss
	H_p = p_to_H(p)
	H_gt = p_to_H(p_gt)

	corners = np.array([[-img_sz/2, img_sz/2, img_sz/2, -img_sz/2],
						[-img_sz/2, -img_sz/2, img_sz/2, img_sz/2],
						[1, 1, 1, 1]])

	corners = np.tile(corners, (batch_size, 1, 1))

	corners_w_p = np.matmul(H_p, corners)
	corners_w_gt = np.matmul(H_gt, corners)

	corners_w_p = corners_w_p[:, 0:2, :] / corners_w_p[:, 2:3, :]
	corners_w_gt = corners_w_gt[:, 0:2, :] / corners_w_gt[:, 2:3, :]

	print('')
	print('Corners Warp:')
	print(corners_w_p)

	print('')
	print('Corners GT:')
	print(corners_w_gt)

	print('')
	print('Corner Dist:')
	print(corners_w_p - corners_w_gt)

	loss = np.sum(((corners_w_p - corners_w_gt) ** 2), (0, 1, 2))

	return loss

def gen_rand_p(min_scale, max_scale, angle_range, projective_range, translation_range):

	# create random ground truth
	scale = random.uniform(min_scale, max_scale)
	angle = random.uniform(-angle_range, angle_range)
	projective_x = random.uniform(-projective_range, projective_range)
	projective_y = random.uniform(-projective_range, projective_range)
	translation_x = random.uniform(-translation_range, translation_range)
	translation_y = random.uniform(-translation_range, translation_range)
	rad_ang = angle / 180 * pi

	p = torch.FloatTensor([
		scale + cos(rad_ang) - 2,
		-sin(rad_ang),
		translation_x,
		sin(rad_ang),
		scale + cos(rad_ang) - 2,
		translation_y,
		projective_x, 
		projective_y])

	return p


def warp_seq(img_tens, p_gt):

	I = img_tens
	for i in range(p_gt.shape[0]):
		img_tens_w, mask_tens_w = dlk.warp_hmg(I[i : i+1, :, :, :], p_gt[i : i+1, :, :])
		I = torch.cat((I, img_tens_w), 0)

	I = I.numpy()

	return I

def open_img_as_tens(img_path, sz):
	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	img = Image.open(img_path)
	img_w, img_h = img.size
	aspect = img_w / img_h
	img_h_sm = sz
	img_w_sm = ceil(aspect * img_h_sm)

	img_tens = preprocess(img.resize((img_w_sm, img_h_sm)))
	img_tens = torch.unsqueeze(img_tens, 0)

	return img_tens


if __name__ == "__main__":
	np.set_printoptions(precision=4)

	parser = argparse.ArgumentParser()
	parser.add_argument("test")
	parser.add_argument("img")
	args = parser.parse_args()

	if (args.test == '3img'):
		test_3img(args)
	elif (args.test == 'multi'):
		test_multi_img(args)
	elif (args.test == '2img'):
		test_2img(args)
	elif (args.test == 'seq'):
		test_img_seq(args)
