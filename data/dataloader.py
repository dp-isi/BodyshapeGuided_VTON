import params as p
import numpy as np
from PIL import Image
import random
import pickle
import os 
import cv2
import matplotlib.image as mpimg
import json
import cv2
import matplotlib.image as mpimg
import glob
	
def rng11(data):
	# data = (data*2)-1
	return data

def resize_img_forMPV(im):
	type=1
	
	if(len(im.shape)>2):
		dim=(256,256,im.shape[-1])
		
	else:
		dim=(256,256)
		type=0
	
	mult,dtyp=(255,'uint8') if(im.max()>1) else (1,'float32')
	temp = np.ones(shape=dim,dtype=dtyp)*mult
	if(type==1):
		temp[:,32:-32,:] = im
	else:
		temp[:,32:-32] = im				
	return temp


def resize_256_192(im):
	type=1
	
	if(len(im.shape)>2):
		dim=(256,256,im.shape[-1])
		
	else:
		dim=(256,256)
		type=0
	
	mult,dtyp=(255,'uint8') if(im.max()>1) else (1,'float32')
	temp = np.ones(shape=dim,dtype=dtyp)*mult
	if(type==1):
		temp[:,32:-32,:] = im
	else:
		temp[:,32:-32] = im				
	return temp

def resize_256_256(im,bkg='black'):
	type=1
	
	if(len(im.shape)>2):
		dim=(256,256,im.shape[-1])
		
	else:
		dim=(256,256)
		type=0
	
	mult,dtyp=(255,'uint8') if(im.max()>1) else (1,'float32')
	
	if(bkg=='white'):
		temp = np.ones(shape=dim,dtype=dtyp)*mult
	else:
		temp = np.zeros(shape=dim,dtype=dtyp)

	if(type==1):
		temp[:,32:-32,:] = im
	else:
		temp[:,32:-32] = im				
	return temp

def find_cloth_name(n):
	# print(n)
	n = n.split('@')[0]+'@*=cloth_*front_mask.jpg'
	# print(n)
	path=p.data_path+'all/'
	c=path+n
	if(len(glob.glob(c))==0):
		return ''
	tmp=glob.glob(c)[0]
	tmp=  tmp.replace('_mask.jpg','.jpg')
	tmp = tmp.replace(path,'')
	# print(tmp)
	return tmp


def generate_data(batch_size = 5, shuffle=True, filename='train_pairs.txt',dataset='df',work='home',stage=1):


	f = open('test_train_pairs/%s'%(filename))
	lines=f.readlines()
	f.close()

	index=-1

	while(1):		
		l_iuv1,l_iuv2,l_i2,l_i1 = [],[],[],[]
		l_seg,l_cl=[],[]
		l_src_im=[]
		l_trgt_im=[]
		l_warp=[]
		
		l_name=[]
		l_z=[]
		l_trgt_im_masked=[]

		

		for i in range(batch_size):
			# print(i)
			index=(index+1)%len(lines)
			
			if(shuffle==True):
			
				if(index==0 or index==len(lines)-1):

					np.random.shuffle(lines)

		
			base_name1,base_name2 = lines[index].strip().split(' ')
		
			base_name3 = find_cloth_name(base_name1)
			
			src_line,trgt_line = base_name1,base_name2

		
			result_name = base_name1.replace('/','+').replace('.jpg','')+ '_TO_' + base_name2.replace('/','+')
		
			l_name.append(result_name)
			

			src_line=p.data_path +'all/'+ src_line
			trgt_line=p.data_path + 'all/' + trgt_line


			def f_iuv(trgt_line,type=1):
				iuv_name=trgt_line.replace('.jpg','_IUV.png')
				if(not os.path.exists(iuv_name)):			   
					return -1,-1 			   
				
				iuv = cv2.imread(iuv_name)
				iuv = iuv.astype('float32')
				
				if(type==1):
					temp=np.zeros(shape=(256,192,26))
					temp[:,:,24:] = iuv[:,:,1:]/255.0
					for i in range(24):
						temp[:,:,i] = (iuv[:,:,0]==i).astype('uint8')
				
					return temp,1

				else:

					return iuv,1

			def get_segment(sname,tile=1):
				seg = np.array(Image.open(sname.replace('all/','all_parsing/').replace('.jpg','.png')))
				cl_mask = (seg==5) + (seg==7) + (seg==6)
				cl_mask = cl_mask[:,:,np.newaxis].astype('float32')
				if(tile==1):
					cl_mask = np.tile(cl_mask,(1,1,3))
				return cl_mask



			cl_mask = get_segment(src_line)
			im=np.array(Image.open(src_line))/255.0
			l_src_im.append(rng11(im))
			
			l_cl.append(rng11(cl_mask*im))

			trgt_im=np.array(Image.open(trgt_line))/255.0
			l_trgt_im.append(rng11(trgt_im))
			


			if(stage==1):

				iuv1,available1 = f_iuv(src_line,type=1)
				iuv2,available2 = f_iuv(trgt_line,type=1)
				if(available1==-1 or available2==-1):
					continue	

				iuv1[:,:,:24] = iuv1[:,:,:24]
				iuv2[:,:,:24] = iuv2[:,:,:24]

				l_iuv1.append(iuv1)
				l_iuv2.append(iuv2)

				l_i2.append(iuv2)
				l_i1.append(iuv1)
				
				l_seg.append(cl_mask)

		
		if(stage==1):
			yield ([l_iuv1,l_iuv2],[l_i2,l_i2],[l_cl,l_name],[l_src_im,l_trgt_im,l_seg])
		
		





