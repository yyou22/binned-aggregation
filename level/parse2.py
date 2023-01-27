from __future__ import print_function
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='binned aggregation')

args = parser.parse_args()

def lvl(datas):

	idx1 = np.load('./idx.npy')
	idx2 = np.load('./idx2.npy')
	idx3 = np.load('./idx3.npy')
	idx4 = np.load('./idx4.npy')

	lvl = []
	in_lvl = []

	for i in idx4:

		lvl.append(datas[i])

		if i in idx1:
			in_lvl.append(1)
		elif i in idx2:
			in_lvl.append(2)
		elif i in idx3:
			in_lvl.append(3)
		else:
			in_lvl.append(4)

	lvl = np.array(lvl)

	idx = np.array(idx4)
	idx = idx.reshape((idx.shape[0], 1))

	in_lvl = np.array(in_lvl)
	in_lvl = in_lvl.reshape((in_lvl.shape[0], 1))

	type_ = ['%d'] * 2 + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, in_lvl, lvl), axis=1)
	np.savetxt("./vgg19_3_lvl4.csv", result, header="ogi,vis,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def main():

	datas = np.genfromtxt('./data_all_vgg19_3.csv', dtype=float, delimiter=',', skip_header=1)

	lvl(datas)

if __name__ == "__main__":
	main()