import glob
from scipy import misc
import os
import sys
import numpy as np
import ipdb


folder1 = '/media/IMLmatuser/DATA/Results/YouTube/Compilation/tinynet_dense'
folder2 = '/media/IMLmatuser/DATA/Results/YouTube/Compilation/FCN_dense'
folder3 = '/media/IMLmatuser/DATA/Results/YouTube/Compilation/FCN_ensemble'
folder4 = '/media/IMLmatuser/DATA/Results/YouTube/Compilation/tnet_tksmpwce_bnormFULLT'

outfolder = '/media/IMLmatuser/DATA/Results/YouTube/Compilation/comp_all'


if(not os.path.exists(outfolder)):
	os.makedirs(outfolder)

files = glob.glob(folder1 + '/casablanca*.png')
for i in range(0, len(files)):
	bname = os.path.basename(files[i])

	im1 =  misc.imread(folder1 + '/' + bname)
	im2 =  misc.imread(folder2 + '/' + bname)
	im3 =  misc.imread(folder3 + '/' + bname)
	im4 =  misc.imread(folder4 + '/' + bname)

	(H, W, CH) = im2.shape
	im2roi = im2[:,W/2:, :]

	(H, W, CH) = im3.shape
	im3roi = im3[:,W/2:, :]

	(H, W, CH) = im4.shape
	im4roi = im4[:,W/2:, :]

	imf = np.concatenate((im1, im2roi, im3roi, im4roi), axis=1)
	misc.imsave(outfolder + '/' + bname, imf)
