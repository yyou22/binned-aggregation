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

parser = argparse.ArgumentParser(description='VGG Extracted Features on adversarial CIFAR10 dataset')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='resnet', help='[resnet|TRADES], model that is being attacked')

args = parser.parse_args()

start_idx = 0

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

epsilon = args.epsilon

def feature_(model, X_data):

	model.eval()

	transform_ = T.Compose(
			[
				T.ToTensor(),
				T.Normalize(mean, std),
			]
		)

	features = []

	for idx in range(start_idx, X_data.shape[0]):

		x_data_ = X_data[idx]#.transpose(1 , 2 , 0)

		x_data = transform_(x_data_)
		x_data = x_data.numpy()

		# load image
		image = np.array(np.expand_dims(x_data, axis=0), dtype=np.float32)
		data = torch.from_numpy(image)
		X = Variable(data)

		out = model(X)

		#print(out.shape)

		features.append(out[0].detach().numpy())

	if args.natural:
		path = '../data/' + args.model + '/000'
	else:
		path = '../data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if not os.path.exists(path):
		os.makedirs(path)

	features = np.array(features)

	np.save(path + '/features.npy', features)


def main():

	if args.model == "resnet":
		#model = vgg16_bn(pretrained=True).features
		model = resnet34(pretrained=True)
		model = FeatureExtractor(model)
	elif args.model == "TRADES":
		#model = vgg19_bn(pretrained=True).features
		model = ResNet34()
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))
		model = FeatureExtractor(model)

	if args.natural:
		path = '../data/' + args.model + '/000'
		X_data = np.load("../data/X.npy")
	else:
		path = '../data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))
		X_data = np.load(path + '/adv_X.npy')

	feature_(model, X_data)

if __name__ == "__main__":
	main()
