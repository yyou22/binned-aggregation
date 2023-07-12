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

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='/content/data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def feature_(model, test_loader):

	model.eval()

	normalize = T.Normalize(mean, std)

	features = []

	for data, target in test_loader:

		data, target = data.to(device), target.to(device)

		data = normalize(data)

		X, y = Variable(data, requires_grad = True), Variable(target)

		# output of model
		out = model(X)

		features.extend(out.cpu().detach().numpy())

	path = '../data/ZOO/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if not os.path.exists(path):
		os.makedirs(path)

	features = np.array(features)

	np.save(path + '/features.npy', features)

def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True).features.to(device)
	elif args.model == "vgg19":
		model = vgg19_bn(pretrained=True).features.to(device)
	elif args.model == "resnet":
		model = resnet34(pretrained=True).to(device)
		model = FeatureExtractor(model)
	elif args.model == "TRADES":
		model = ResNet34().to(device)
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))
		model = FeatureExtractor(model)

	path = '../data/ZOO/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))
	X_data = np.load(path + '/adv_X.npy').transpose(0,3,1,2)

	tensor_x = torch.Tensor(X_data)
	tensor_y = torch.Tensor(testset.targets)

	my_dataset = TensorDataset(tensor_x,tensor_y) # create datset
	my_dataloader = DataLoader(my_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs) # create dataloader
	feature_(model, my_dataloader)

if __name__ == "__main__":
	main()

