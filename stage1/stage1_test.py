import sys
sys.path.append('./')
from models import model as stage1_model
import params as p
import os
os.environ['CUDA_VISIBLE_DEVICES']=p.gpu_id	
from PIL import Image
import numpy as np
from data import dataloader as generate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size',default='1',help='size of batch to predict',type=int )
args = parser.parse_args()
bs = args.batch_size




pathm='./checkpoints/stage1/final/'

model,model_im  = stage1_model.get_model_gmm()

paths = '%s'%(p.stage1_res_folder)
filename=p.test_filename


# -----------------------------------------------
if(not os.path.exists(paths)):
	os.makedirs(paths)
paths = '%s/warps/'%(paths)

import shutil
if(os.path.exists(paths)):
	shutil.rmtree(paths)
os.makedirs(paths)

paths2 = paths.replace('/warps/','/warp-mask/')
if(not os.path.exists(paths2)):	
	os.makedirs(paths2)
# -----------------------------------------------


model.load_weights('%smodel_wt_latest.h5'%(pathm))

# -----------------------------------------------


file=open('./test_train_pairs/%s'%(filename))
dc=len(file.readlines())


obj_test = generate.generate_data(batch_size=bs,stage=1,dataset='mpv',filename=filename,shuffle=False)


def f(a):
	return (a*255).astype('uint8')

from skimage.filters import threshold_otsu, threshold_adaptive

import cv2
for i in range(0,dc,bs):
	
	[l_iuv1,l_iuv2],[l_i1,_],[l_cl,l_name],[l_src_im,l_trgt_im,l_cl_mask] = next(obj_test)
	ip=[l_iuv1,l_iuv2]
	op=[l_i1]

	ip1=[l_iuv1,l_iuv2,l_cl]
	op2 = model_im.predict(ip1)
	ip1=[l_iuv1,l_iuv2,l_cl_mask]
	op2_mask = model_im.predict(ip1)

	for j in range(len(op2_mask)):
		temp_mask = (op2_mask[j]>0).astype('float32')
		kernel = np.ones((5, 5), np.uint8)
		temp_mask = cv2.erode(temp_mask, kernel) 

		Image.fromarray(f(op2[j])).save('%s%s'%(paths,l_name[j]))
		Image.fromarray(f(temp_mask)).save('%s%s'%(paths2,l_name[j].replace('.jpg','_mask.png')))
