from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import shutil

idx = 2

def main():

	og_data = np.load("./data/X.npy")
	adv_data1 = np.load("./data/resnet/01/adv_X.npy")
	adv_data3 = np.load("./data/resnet/03/adv_X.npy")
	adv_data5 = np.load("./data/resnet/05/adv_X.npy")

	#FIXME
	noise1 = np.load("./data/resnet/01/noise.npy")
	noise3 = np.load("./data/resnet/03/noise.npy")
	noise5 = np.load("./data/resnet/05/noise.npy")

	noise1_ = (noise1[idx] - np.min(noise1[idx])) / (np.max(noise1[idx]) - np.min(noise1[idx]))
	noise3_ = (noise3[idx] - np.min(noise3[idx])) / (np.max(noise3[idx]) - np.min(noise3[idx]))
	noise5_ = (noise5[idx] - np.min(noise5[idx])) / (np.max(noise5[idx]) - np.min(noise5[idx]))

	# Normalize the differences
	diff1 = og_data[idx] - adv_data1[idx]
	diff1 = (diff1 - np.min(diff1)) / (np.max(diff1) - np.min(diff1))

	diff3 = og_data[idx] - adv_data3[idx]
	diff3 = (diff3 - np.min(diff3)) / (np.max(diff3) - np.min(diff3))

	diff5 = og_data[idx] - adv_data5[idx]
	diff5 = (diff5 - np.min(diff5)) / (np.max(diff5) - np.min(diff5))

	print(np.max(og_data[idx] - adv_data5[idx]))
	print(np.min(og_data[idx] - adv_data5[idx]))

	# Create a new figure
	plt.figure(figsize=(12, 8))

	# Subplot for original data
	plt.subplot(2, 4, 1)
	plt.title("Original")
	plt.imshow(og_data[idx])  # Assuming the images are grayscale. If they are color, remove cmap='gray'

	# Subplot for adversarial data1
	plt.subplot(2, 4, 2)
	plt.title("Adversarial Data 1")
	plt.imshow(diff1)

	# Subplot for adversarial data3
	plt.subplot(2, 4, 3)
	plt.title("Adversarial Data 3")
	plt.imshow(diff3)

	# Subplot for adversarial data5
	plt.subplot(2, 4, 4)
	plt.title("Adversarial Data 5")
	plt.imshow(diff5)

	#print(np.min(diff1))
	#print(np.min(diff5))

	#print(np.min(noise1_))
	#print(np.min(noise3_))

	plt.subplot(2, 4, 5)
	plt.title("Adversarial Data 1 from noise.npy")
	plt.imshow(noise1_)

	plt.subplot(2, 4, 6)
	plt.title("Adversarial Data 3 from noise.npy")
	plt.imshow(noise3_)

	plt.subplot(2, 4, 7)
	plt.title("Adversarial Data 5 from noise.npy")
	plt.imshow(noise5_)

	# Show the plot
	plt.show()

	# Show the plot
	plt.show()

if __name__ == "__main__":
	main()