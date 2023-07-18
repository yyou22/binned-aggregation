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
import torchvision

from models.resnet import *
from feature_extractor import FeatureExtractor
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='VGG Extracted Features on adversarial CIFAR10 dataset')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.3, type=float, help='epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg16', help='[vgg16|vgg19], model that is being attacked')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')

args = parser.parse_args()

mean = [0.4914, 0.4822, 0.4465]
std = [0.2471, 0.2435, 0.2616]

inv_mean = [-0.4914/0.2471, -0.4822/0.2435, -0.4465/0.2616]
inv_std = [1/0.2471, 1/0.2435, 1/0.2616]

epsilon = args.epsilon

start_idx = 0

# set up data loader
#transform_test = transforms.Compose([transforms.ToTensor(),])
#testset = torchvision.datasets.CIFAR10(root='/content/data', train=False, download=True, transform=transform_test)
#test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

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

		if args.natural:
			x_data_ = X_data[idx]
		else:
			x_data_ = X_data[idx].transpose(1 , 2 , 0)

		x_data = transform_(x_data_)
		x_data = x_data.numpy()

		# load image
		image = np.array(np.expand_dims(x_data, axis=0), dtype=np.float32)
		data = torch.from_numpy(image)
		X = Variable(data)

		out = model(X)

		features.append(out[0].detach().numpy())

	if args.natural:
		path = './ZOO/data/' + args.model + '/00'
	else:
		path = './ZOO/data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if not os.path.exists(path):
		os.makedirs(path)

	features = np.array(features)

	np.save(path + '/features.npy', features)

def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True).features
	elif args.model == "vgg19":
		model = vgg19_bn(pretrained=True).features
	elif args.model == "resnet":
		model = resnet34(pretrained=True)
		model = FeatureExtractor(model)
	elif args.model == "TRADES":
		model = ResNet34()
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt"))
		model = FeatureExtractor(model)

	path = './ZOO/data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if args.natural:
		X_data = np.load('./ZOO/data/X.npy')
	else:
		X_data = np.load(path + '/adv_X.npy').transpose(0,3,1,2)

	#tensor_x = torch.Tensor(X_data)
	#tensor_y = torch.Tensor(testset.targets)

	#my_dataset = TensorDataset(tensor_x,tensor_y) # create datset
	#my_dataloader = DataLoader(my_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs) # create dataloader
	feature_(model, X_data)

if __name__ == "__main__":
	main()

