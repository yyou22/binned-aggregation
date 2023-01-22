from __future__ import print_function
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import argparse
import pickle
import math
import csv
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='dimesionality reduction on FGSM-perturbed CIFAR10 datasets')
parser.add_argument('--model', default='resnet', help='[resnet|TRADES], model that is being attacked')

args = parser.parse_args()

def TSNE_(data):

	tsne = TSNE(n_components=2)
	data = tsne.fit_transform(data)

	return data

def PCA_(data):

	pca = PCA(n_components=2)
	data = pca.fit_transform(data)

	return data

def main():

	Y_data = np.load("../data/Y.npy")
	Y_data = Y_data.reshape((Y_data.shape[0], 1))

	X_data_list = []
	Y_hat_list = []

	for i in range(0, 4):
		path = '../data/' + args.model + '/00'  + str(i) + '/'

		path_x = path + 'features.npy'
		X_data_list.append(np.load(path_x))

		path_y = path + 'Y_hat.npy'
		Y_hat_list.append(np.load(path_y))

	X_data = np.concatenate((X_data_list[0], X_data_list[1], X_data_list[2], X_data_list[3]))

	if args.model != "TRADES":
		X_data = X_data.reshape(X_data.shape[0], 512)
	else:
		X_data = X_data.reshape(X_data.shape[0], 512*4*4)

	X_data_t = TSNE_(X_data)

	tx, ty = X_data_t[:, 0].reshape(100*4, 1), X_data_t[:, 1].reshape(100*4, 1) #change size
	tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
	ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

	tx_list = np.array_split(tx, 4)
	ty_list = np.array_split(ty, 4)

	X_data_p = PCA_(X_data)

	px, py = X_data_p[:, 0].reshape(100*4, 1), X_data_p[:, 1].reshape(100*4, 1) #change size
	px = (px-np.min(px)) / (np.max(px) - np.min(px))
	py = (py-np.min(py)) / (np.max(py) - np.min(py))

	px_list = np.array_split(px, 4)
	py_list = np.array_split(py, 4)

	type_ = ['%.5f'] * 14 + ['%d'] * 2

	for i in range(0, 4):

		path = '../data/' + args.model + '/00'  + str(i) + '/'

		confid_level = np.load(path + '/confid_level.npy')

		Y_hat = Y_hat_list[i].reshape((Y_hat_list[i].shape[0], 1))

		result = np.concatenate((tx_list[i], ty_list[i], px_list[i], py_list[i], confid_level, Y_hat, Y_data), axis=1)
		np.savetxt(path + "/data.csv", result, header="xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

if __name__ == "__main__":
	main()