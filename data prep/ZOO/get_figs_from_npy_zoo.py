from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import shutil

label_str = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def generate_data_and_noise_png(data_file_name="X.npy", parent="../data/ZOO/", noise=True):

	data = np.load(os.path.join(parent, data_file_name))
	data[data > 1] = 1
	data[data < 0] = 0

	if noise:
		global noise_data
		noise_data = np.load(os.path.join(parent, "noise.npy"))#.transpose(0,2,3,1)
		min_val = np.min(noise_data)
		max_val = np.max(noise_data)

		# Subtract min_val from all elements and divide by the range of values
		noise_data = (noise_data - min_val) / (max_val - min_val)
		noise_data[noise_data > 1] = 1
		noise_data[noise_data < 0] = 0

	parent = os.path.join(parent, "img_data/")
	shutil.rmtree(parent, ignore_errors=True)
	os.mkdir(parent)

	for i in range(data.shape[0]):
		cur_dir = os.path.join(parent, str(i))
		shutil.rmtree(cur_dir, ignore_errors=True)
		os.mkdir(cur_dir)
		plt.imsave(os.path.join(cur_dir, "img.png"), data[i], format="png")
		if noise:
			plt.imsave(os.path.join(cur_dir, "noise.png"), noise_data[i], format="png")

def generate_label_txt(label_file_name="Y.npy", parent="../data/ZOO/"):
	labels = np.load(os.path.join(parent, label_file_name))
	parent = os.path.join(parent, "img_data/")
	os.makedirs(parent, exist_ok=True)

	for i in range(labels.size):
		cur_dir = os.path.join(parent, str(i))
		os.makedirs(cur_dir, exist_ok=True)
		f = open(os.path.join(cur_dir, "label.txt"), "w")
		f.write(label_str[labels[i]])
		f.close()

adv_data = "adv_X.npy"
adv_label = "Y_hat.npy"

generate_data_and_noise_png(data_file_name = adv_data, parent = "./data/vgg16/01/")
generate_label_txt(label_file_name = adv_label, parent = "./data/vgg16/01/")

