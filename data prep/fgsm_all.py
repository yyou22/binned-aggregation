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
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pickle

from models.resnet import *

parser = argparse.ArgumentParser(description='FGSM Attack on CIFAR-10 with VGG Models')
parser.add_argument('--natural', action='store_true', help='natural prediction on the unperturbed dataset')
parser.add_argument('--epsilon', default=0.03, type=float, help='epsilon, the maximum amount of perturbation that can be applied')
parser.add_argument('--model', default='vgg16', help='[vgg16|vgg19|resnet|trades], model that is being attacked')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()

torch.manual_seed(0)

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

def margin_loss(logits,y):
	logit_org = logits.gather(1,y.view(-1,1))
	logit_target = logits.gather(1,(logits - torch.eye(10, device=y.device)[y] * 9999).argmax(1, keepdim=True))
	loss = -logit_org + logit_target
	loss = torch.sum(loss)
	return loss

def fgsm(image, epsilon, target, model):

	X, y = Variable(image, requires_grad = True), Variable(target)

	# output of model
	out = model(X)

	#print(out.data[0])
	#print(F.softmax(out.data[0]))

	loss = margin_loss(out, y)

	model.zero_grad()

	loss.backward()

	data_grad = X.grad.data

	sign_data_grad = torch.sign(data_grad)

	perturbed_image = image + epsilon*sign_data_grad

	perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)

	return perturbed_image

def natural(model, test_loader):

	model.eval()

	normalize = T.Normalize(mean, std)

	wrong = 0
	confid_level = []
	pred_ = []

	for data, target in test_loader:

		data, target = data.to(device), target.to(device)

		data = normalize(data)

		X, y = Variable(data, requires_grad = True), Variable(target)

		# output of model
		out = model(X)

		confid_ = F.softmax(out.data)
		confid_level.extend(confid_.cpu().numpy())

		pred = out.data.max(1)[1].cpu()
		pred_.extend(pred)

		wrong += (out.data.max(1)[1] != Variable(target).data).float().sum()

	confid_level = np.array(confid_level)
	pred_ = np.array(pred_)

	path = '../data/' + args.model + '/000'

	if not os.path.exists(path):
		os.makedirs(path)

	np.save(path + '/confid_level.npy', confid_level)
	np.save(path + '/Y_hat.npy', pred_)

	f = open(path + '/error.pckl', 'wb')
	pickle.dump(wrong, f)
	f.close()

	print("Natural Error:" + str(wrong))

def attack(model, test_loader):

	model.eval()

	tensor_ = T.ToTensor()
	normalize = T.Normalize(mean, std)

	inv_normalize = T.Normalize(inv_mean, inv_std)

	wrong = 0
	adv_examples = []
	confid_level = []
	noise = []
	pred_ = []

	for data, target in test_loader:

		data, target = data.to(device), target.to(device)

		#display image
		#plt.imshow(data[0].cpu().numpy().transpose(1 , 2 , 0))
		#plt.show()

		perturbed_data = fgsm(data, epsilon, target, model)

		perturbed_data = normalize(perturbed_data)

		X_ = Variable(perturbed_data)
		out = model(X_)

		confid_ = F.softmax(out.data)

		confid_level.extend(confid_.cpu().numpy())

		pred = out.data.max(1)[1].cpu()
		pred_.extend(pred)

		wrong += (out.data.max(1)[1] != Variable(target).data).float().sum()

		#undo transformation
		perturbed_data = inv_normalize(perturbed_data)
		
		#display image
		#plt.imshow(perturbed_data[0].cpu().numpy().transpose(1 , 2 , 0))
		#plt.show()

		#obtain the noise applied and save it as a matrix
		og_image = data
		og_image = og_image.cpu().numpy()

		#og_image = np.array(np.expand_dims(og_image, axis=0), dtype=np.float32)
		diff = og_image - perturbed_data.cpu().numpy()
		diff = np.sign(diff)
		diff = (diff+1)/2

		noise.extend(diff)

		adv_examples.extend(perturbed_data.cpu().numpy().transpose(0, 2 , 3 , 1))
  
	adv_examples = np.array(adv_examples)
	confid_level = np.array(confid_level)
	pred_ = np.array(pred_)
	noise = np.array(noise)

	path = '../data/' + args.model + '/' + ''.join(str(args.epsilon).split('.'))

	if not os.path.exists(path):
		os.makedirs(path)

	np.save(path + '/adv_X.npy', adv_examples)
	np.save(path + '/Y_hat.npy', pred_)
	np.save(path + '/confid_level.npy', confid_level)
	np.save(path + '/noise.npy', noise)

	f = open(path + '/error.pckl', 'wb')
	pickle.dump(wrong, f)
	f.close()

	print("Robust Error: " + str(wrong))


def main():

	if args.model == "vgg16":
		model = vgg16_bn(pretrained=True).to(device)
	elif args.model == "vgg19":
		model = vgg19_bn(pretrained=True).to(device)
	if args.model == "resnet":
		model = resnet34(pretrained=True).to(device)
	elif args.model == "TRADES":
		model = ResNet34().to(device)
		model.load_state_dict(torch.load("./resnet/model-advres-epoch200.pt", map_location=torch.device('cpu')))

	if args.natural:
		natural(model, test_loader)
	else:
		attack(model, test_loader)

if __name__ == "__main__":
	main()
