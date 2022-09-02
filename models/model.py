import sys
sys.path.append("./")
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model,Model
from keras.applications.vgg19 import VGG19
from keras.layers import *
from keras.layers import Layer

from models.spline_transformer_tf import *
import keras.losses
from keras import optimizers
import keras.backend as K
from models.spline_transformer_tf import *

class FeatureL2Norm(Layer):
	def __init__(self):
		super(FeatureL2Norm, self).__init__()

	def call(self, feature):
		epsilon = 1e-6
		norm = tf.math.pow(tf.math.reduce_sum(tf.math.pow(feature,2),1)+epsilon,0.5)
		norm = K.repeat_elements(K.expand_dims(norm,1),feature.shape[1],axis=1)
		return tf.math.divide(feature,norm)


class FeatureCorrelation_2D(Layer):
	def __init__(self):
		# self.od=16
		super(FeatureCorrelation_2D, self).__init__()

	def call(self,feature):

		[feature_A,feature_B] = feature
		b,h,w,c = feature_A.get_shape().as_list()
		
		b=-1
		feature_A = K.reshape(feature_A,(b,h*w,c))
		feature_B = K.reshape(feature_B,(b,h*w,c))
		feature_B = Lambda(lambda l: tf.transpose(l,perm=(0,2,1)))(feature_B)
		
		feature_mul = K.batch_dot(feature_A,feature_B)		
		correlation_tensor = K.reshape(feature_mul,(b,h,w,h*w))
		
		return correlation_tensor


	def get_config(self):
		config = super(FeatureCorrelation_2D, self).get_config()
		return config

	def compute_output_shape(self, input_shape):
		return (input_shape[0][0],input_shape[0][1],input_shape[0][2],input_shape[0][1]*input_shape[0][2])


from keras_contrib.losses import DSSIMObjective


def get_model_gmm():
	
	iuv1 = Input(shape=(256,192,26))	
	iuv2 = Input(shape=(256,192,26))	
	i1 = Lambda(lambda l: l[:,:,:,:24])(iuv1)

	def FeatureExtraction(fa):
		n_layers=3
		ngf=64
		x = Conv2D(ngf, kernel_size=(4,4),strides=(2,2),padding='same')(fa)
		x = ReLU()(x)
		for i in range(n_layers):
			out_ngf = 2**(i+1) * ngf if 2**i * ngf < 512 else 512
			x = Conv2D(ngf, kernel_size=(4,4),strides=(2,2),padding='same')(x)
			x = ReLU()(x)			
		x = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same')(x)
		x = ReLU()(x)		
		x = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same')(x)
		x = ReLU()(x)
		x = FeatureL2Norm()(x)
		return x

	feature_i1 = FeatureExtraction(iuv1)
	feature_i2 = FeatureExtraction(iuv2)
	
	correlation_iuv = FeatureCorrelation_2D()([feature_i1, feature_i2])
	correlation = correlation_iuv

	
	x = Conv2D(512, (4,4), strides = (2,2),padding='same')(correlation)	
	x = ReLU()(x)
	x = Conv2D(256, (4,4), strides = (2,2), padding='same')(x)	
	x = ReLU()(x)
	x = Conv2D(128, (3,3), strides = (1,1),padding='same')(x)	
	x = ReLU()(x)
	x = Flatten()(x)
	x = Dense(64, activation='relu')(x)
	
	
	num_points = 9
	theta = Dense(num_points*2,activation = 'tanh',kernel_initializer = initializers.RandomNormal(stddev=0.01))(x)
	num_points1=16
	theta_44 = Dense(num_points1*2,activation = 'tanh',kernel_initializer = initializers.RandomNormal(stddev=0.01))(x)
	num_points2=25
	theta_55 = Dense(num_points1*2,activation = 'tanh',kernel_initializer = initializers.RandomNormal(stddev=0.01))(x)
		
	iuv_warp = TPSTransformerLayer(control_points=num_points,input_shape=(256,192,24))([i1,theta])	
	iuv_warp_44 = TPSTransformerLayer(control_points=num_points1,input_shape=(256,192,24))([i1,theta_44])	
	iuv_warp_55 = TPSTransformerLayer(control_points=num_points1,input_shape=(256,192,24))([i1,theta_55])
	
	
	model_gmm = Model([iuv1,iuv2],[iuv_warp,iuv_warp_44,iuv_warp_55])
	model_gmm.compile(loss = [DSSIMObjective(kernel_size = 23),DSSIMObjective(kernel_size = 23),\
		DSSIMObjective(kernel_size = 23)],optimizer=optimizers.Adam(0.00002, beta_1=0.5, beta_2=0.999))	
	
		
	im=Input(shape=(256,192,3))
	im_warp = TPSTransformerLayer(control_points=num_points1,input_shape=(256,192,3))([im,theta_55])
	model_gmm_cloth = Model([iuv1,iuv2,im],[im_warp])

	return model_gmm,model_gmm_cloth

