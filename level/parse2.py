from __future__ import print_function
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='binned aggregation')
parser.add_argument('--level', default=1, type=int)

args = parser.parse_args()

def lvl1(datas):

	idx = np.load('./idx.npy')

	lvl1 = []

	for i in idx:

		lvl1.append(datas[i])

	lvl1 = np.array(lvl1)
	idx = np.array(idx)
	idx = idx.reshape((idx.shape[0], 1))

	type_ = ['%d'] + ['%.5f'] * 14 + ['%d'] * 2

	result = np.concatenate((idx, lvl1), axis=1)
	np.savetxt("./vgg16_1_lvl1.csv", result, header="ogi,xpost,ypost,xposp,yposp,0,1,2,3,4,5,6,7,8,9,pred,target", comments='', delimiter=',', fmt=type_)

def main():

	datas = np.genfromtxt('./data_all_vgg16_1.csv', dtype=float, delimiter=',', skip_header=1)

	lvl1(datas)

if __name__ == "__main__":
	main()