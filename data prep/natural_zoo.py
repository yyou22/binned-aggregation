from __future__ import print_function
from cifar10_models.vgg import vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18, resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

from models.resnet import *
from feature_extractor import FeatureExtractor
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='FGSM Attack on CIFAR-10 with VGG Models')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--model', default='TRADES', help='[vgg16|vgg19|resnet|TRADES], model that is being attacked')

args = parser.parse_args()

start_idx = 0

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

def natural(model, X_data, Y_data):

	model.eval()

	transform_ = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(mean, std),
			]
		)

	wrong = 0
	confid_level = []
	pred_ = []

	for idx in range(start_idx, len(Y_data)):

		x_data = transform_(X_data[idx])
		x_data = x_data.numpy()

		# load original image
		image = np.array(np.expand_dims(x_data, axis=0), dtype=np.float32)

		# load label
		label = np.array([Y_data[idx]], dtype=np.int64)

		# transform to torch.tensor
		data = torch.from_numpy(image)
		target = torch.from_numpy(label)

		X, y = Variable(data, requires_grad = True), Variable(target)

		# output of model
		out = model(X)

		confid_ = F.softmax(out.data[0])
		confid_level.append(confid_.numpy())

		init_pred = out.data.max(1)[1]

		pred_.append(init_pred)

		#print("pred: ", init_pred)
		#print("target: ", target)

		if init_pred != target:
			wrong += 1

	confid_level = np.array(confid_level)
	pred_ = np.array(pred_)

	path = './ZOO/data/' + args.model + '/00'

	if not os.path.exists(path):
		os.makedirs(path)

	np.save(path + '/confid_level.npy', confid_level)
	np.save(path + '/Y_hat.npy', pred_)

	f = open(path + '/error.pckl', 'wb')
	pickle.dump(wrong, f)
	f.close()

	print("Natural Error:" + str(wrong))

def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True)
	elif args.model == "vgg19":
		model = vgg19_bn(pretrained=True)
	elif args.model == "resnet":
		model = resnet34(pretrained=True)
	elif args.model == "TRADES":
		model = ResNet34()
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))

	X_data = np.load("./ZOO/data/X.npy")
	Y_data = np.load("./ZOO/data/Y.npy")

	natural(model, X_data, Y_data)

if __name__ == "__main__":
	main()
