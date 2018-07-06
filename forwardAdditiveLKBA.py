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

'''
optimize(I, P, T, V):
	# I: numpy N x C x H x W
	# P: numpy N - 1 x 8 x 1
	# T: numpy K
	# V: numpy sum_k sum_f 1 (sigma)
	
	dP = numpy zeros(num_frame, 8, 1)
	crit = 0
	tol = 1e-5

	itn = 1

	I_gradx, I_grady = compute gradients of I, copy out to sigma

	while (crit > tol) or (itn == 1):
		P_fk = compute_Pfk(P, T, V)

		r_ba = compute_r(I, P, T, V, P_fk)
		# J_ba: N - 1 x sigma x (C x H x W) x 1
	
		J_ba = compute_J(I_gradx, I_grady, P, T, V, P_fk)
		# J_ba: N - 1 x sigma x (C x H x W) x 8

		J_tJ = J_ba.tranpose() * J_ba
		# J_tJ: N - 1 x sigma x 8 x 8

		invH = inv(J_tJ.sum(axis = 1))
		# invH: N - 1 x 8 x 8

		J_tr = (J_ba.transpose() * r_ba).sum(axis = 1)
		# J_tr: N - 1 x 8 x 1

		dP = invH.bmm(J_tr)

		crit = smallest norm of dP

		P = P + dP

		itn = itn + 1
'''

def optimize(I, P, T, V):
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
	tol = 1e-4

	itn = 1

	while (crit > tol) or (itn == 1):
		P_fk = compute_Pfk(P, T, V, V_sz)

		r_ba = compute_r(I, P, T, V, P_fk, V_sz)

		J_ba = compute_J(I_gradx, I_grady, P, T, V, P_fk, V_sz)

		J_ba_trans = J_ba.swapaxes(2,3)

		J_ba_trans_J_ba = np.matmul(J_ba_trans, J_ba) 

		Hess = np.sum(J_ba_trans_J_ba, axis=1)

		invHess = np.linalg.inv(Hess)

		J_ba_trans_r_ba = np.matmul(J_ba_trans, r_ba)

		J_ba_trans_r_ba_sum = np.sum(J_ba_trans_r_ba, axis=1)

		dP = np.matmul(invHess, J_ba_trans_r_ba_sum)

		P = P + dP

		crit = np.amax(np.linalg.norm(dP, ord=2, axis=1))

		print('itn:  {:d}'.format(itn))
		print('crit: {:.2f}'.format(crit))

		itn = itn + 1

	return P

'''
compute_r(I, P, T, V, P_fk)
	
	extract T
	populate image store matrix

	warp image storage matrix with P_fk

	res = mask Templates, subtract templates from I_f_warp

	reshape and tile res N - 1

	return res
'''

def compute_r(I, P, T, V, P_fk, V_sz):
	'''
	Args:
		I: Image sequence, 4D numpy array, num_frame x C x H x W
		P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		T: Numpy vector containing indices from I which are templates
		V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template
		P_fk: Numpy array, warps images I to respective templates T, sigma x 8 x 1
		V_sz: Numpy array, 1D, indicating number of images in each visibility neighborhood

	Returns:
		r_ba: Numpy array, residuals of 
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

'''
compute_J(I_gradx, I_grady, P, T, V, P_fk)

	initialize gradI_warpjac = np.zeros((num_frame - 1, sigma, h*w*c,  8))

	gradI_warpjac = compute_gradI_warpjac(I_gradx, I_grady, P_fk)

	initialize gradPkf_PF = np.zeros((num_frame - 1, sigma, 8, 8))

	populate gradPkf_PF with for loop

	return gradI_warpjac.bmm(gradPkf_PF)
'''

def compute_J(I_gradx, I_grady, P, T, V, P_fk, V_sz):

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
						return P_fk

					grad_P_fk = jacobian(Pfk)

					P_F = H_to_p(H_kf[H_F_ind, :, :])

					gradPfk_PF[F, frame_id, :, :] = \
						grad_P_fk(P_F).squeeze(axis=1).squeeze(axis=2)

				else:
					gradPfk_PF[F, frame_id, :, :] = np.zeros((8,8))

				frame_id = frame_id + 1

			k_ind = k_ind + 1

	J = np.matmul(gradI_warpjac, gradPfk_PF)

	return J

'''
compute_Pfk(P, T, V):
	initialize P_fk motion parameter matrix, np.zeros((sigma, 8, 1))

	populate P_fk motion parameter matrix with for loops

	return P_fk
'''

def compute_Pfk(P, T, V, V_sz):

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

				# evaluate 8x8 jacobian and store it

			else:
				H_kf = p_to_H(P[k : f, :, :])

				H_kf_inv = np.linalg.inv(H_kf)

				H_fk_mat = np.eye(3)

				for i in range(H_kf_inv.shape[0]):
					H_fk_mat = np.dot(H_fk_mat, H_kf_inv[i])

				P_fk = H_to_p(H_fk_mat)

			P_fk_all[frame_id, :, :] = P_fk

	return P_fk_all

def compute_gradI_warpjac(I_gradx, I_grady, P_fk):
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

	X_warp = X_warp.view(X_warp.numel(), 1)
	Y_warp = Y_warp.view(Y_warp.numel(), 1)

	X_warp = X_warp.repeat(batch_size, c, 1)
	Y_warp = Y_warp.repeat(batch_size, c, 1)

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



def test_2img():
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
		[0],[0],[10],[0],[0],[10],[0],[0]
		]])
	
	img_tens_w, mask_tens_w = dlk.warp_hmg(img_tens, p_gt)

	I = torch.cat((img_tens, img_tens_w), 0).numpy()

	P = np.array([
		[[0], [0], [8], [0], [0], [8], [0], [0]]
		])

	T = np.array([1])

	V = np.array(
		[
		 np.array([0, 1]),
		 np.array([0])
		]
	)

	V = np.delete(V, 0)

	P_opt = optimize(I, P, T, V)
		# I: Image sequence, 4D numpy array, num_frame x C x H x W
		# P: Warp parameter sequence, 3D numpy array, num_frame x 8 x 1
		# T: Numpy vector containing indices from I which are templates
		# V: Numpy array of numpy arrays, indicating the visibility neighborhood for each template

	print(P_opt)

# next test: align 4 images together


if __name__ == "__main__":
	test_2img()

