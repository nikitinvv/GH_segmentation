import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras import losses
from tensorflow import Tensor

from keras.utils import multi_gpu_model

smooth = 1.
dropout_rate = 0.5

##############################################################################################################################

# define the custom loss functions 

def jaccard_distance_loss(y_true, y_pred, smooth=100):

    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


##############################################################################################################################

# Define the Deep learning models using functional Keras API: 
# 1. Unet 
# 2. NestNet(UNet++)
# 3. DenseNet67(not work)

def unet(pretrained_weights = None,input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def unet_multiGPU(pretrained_weights = None,input_size = (512,512,1),GPU_list = [0, 1]):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    print(GPU_list)
    parallel_model = multi_gpu_model(model, gpus=GPU_list)
    parallel_model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return parallel_model

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    act = 'relu'

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def Nest_Net(img_rows, img_cols, color_type=1, num_class=2, deep_supervision=False):

    nb_filter = [32,64,128,256,512]
    act = 'relu'

    # Handle Dimension Ordering for different backends

    global bn_axis
    if K.common.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
    return model


def wU_Net(img_rows, img_cols, color_type=1, num_class=1):

    nb_filter = [32,64,128,256,512]
    act = 'relu'

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.common.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(input=img_input, output=unet_output)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = jaccard_distance_loss, metrics = ['accuracy'])
    
    model.summary()

    return model


def BN_act_conv(x,nb_filter, dropout_rate=0.2):

    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), kernel_initializer="VarianceScaling", padding="same")(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)

    return x

def transitionDown(x, nb_filter, dropout_rate=0.2):

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1), kernel_initializer="VarianceScaling", padding="same")(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
 
    return x

def transitionUp(x, nb_filter, dropout_rate=0.2):

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (1, 1), kernel_initializer="VarianceScaling", padding="same")(x)
    if dropout_rate: x = Dropout(dropout_rate)(x)
    x = Conv2DTranspose(n_filters, kernel_size=(3,3), strides=(2,2), padding='same', activation='linear', 
                        kernel_initializer='VarianceScaling')(x)
 
    return x



def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    """

    l = BatchNormalization(axis=3)(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, (3, 3), kernel_initializer="he_uniform", padding="same")(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2, 2), strides=(2, 2))(l)
    
    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = concatenate(block_to_upsample, axis = 3)
    l = Conv2DTranspose(n_filters_keep, kernel_size=(3,3), strides=(2,2), padding='same', activation='linear', 
                        kernel_initializer="he_uniform")(l)
    # Concatenate with skip connection
    l = concatenate([l, skip_connection], axis=3)

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution

def DenseNet (input_shape=(512,512,1),
             n_classes=1,
             n_filters_first_conv=48,
             n_pool=4,
             growth_rate=16,
             n_layers_per_block=5,
             dropout_p=0.2):

    """
    This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
    The network consist of a downsampling path, where dense blocks and transition down are applied, followed
    by an upsampling path where transition up and dense blocks are applied.
    Skip connections are used between the downsampling path and the upsampling path
    Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.
    :param input_shape: shape of the input batch. Only the first dimension (n_channels) is needed
    :param n_classes: number of classes
    :param n_filters_first_conv: number of filters for the first convolution applied
    :param n_pool: number of pooling layers = number of transition down = number of transition up
    :param growth_rate: number of new feature maps created by each layer in a dense block
    :param n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
    :param dropout_p: dropout rate applied after each convolution (0. for not using)
    """

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    inputs = Input(input_shape)


    #####################
    # First Convolution #
    #####################
    
    # We perform a first convolution. All the features maps will be stored in the tensor called stack
    stack = Conv2D(48, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D")(inputs)
    
    # The number of feature maps in the stack is stored in the variable n_filters
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################

    skip_connection_list = []

    for i in range(n_pool):
        # Dense Block
        for j in range(n_layers_per_block[i]):
            # Compute new feature maps
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            # And stack it : the Tiramisu is growing
            stack = concatenate([stack, l], axis=3)
            n_filters += growth_rate
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p)

    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store now the output of the next dense block in a list. We will only upsample these new feature maps
    block_to_upsample = []

    # Dense Block
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack, l], axis=3)

    #######################
    #   Upsampling path   #
    #######################

    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        # Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l], axis=3)

    #####################
    #      Softmax      #
    #####################
    
    last_conv = Conv2D(n_classes, (1, 1), kernel_initializer="he_uniform", padding="same", activation='softmax', name="last_conv2D")(stack)
    
    model = Model(input=inputs, output=last_conv)
    model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def DenseNet_multiGPU (input_shape=(512,512,1),
             GPU_numb=8,
             n_classes=1,
             n_filters_first_conv=48,
             n_pool=4,
             growth_rate=16,
             n_layers_per_block=5,
             dropout_p=0.2):

    """
    This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
    The network consist of a downsampling path, where dense blocks and transition down are applied, followed
    by an upsampling path where transition up and dense blocks are applied.
    Skip connections are used between the downsampling path and the upsampling path
    Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.
    :param input_shape: shape of the input batch. Only the first dimension (n_channels) is needed
    :param n_classes: number of classes
    :param n_filters_first_conv: number of filters for the first convolution applied
    :param n_pool: number of pooling layers = number of transition down = number of transition up
    :param growth_rate: number of new feature maps created by each layer in a dense block
    :param n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
    :param dropout_p: dropout rate applied after each convolution (0. for not using)
    """

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    inputs = Input(input_shape)


    #####################
    # First Convolution #
    #####################
    
    # We perform a first convolution. All the features maps will be stored in the tensor called stack
    stack = Conv2D(48, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D")(inputs)
    
    # The number of feature maps in the stack is stored in the variable n_filters
    n_filters = n_filters_first_conv

    #####################
    # Downsampling path #
    #####################

    skip_connection_list = []

    for i in range(n_pool):
        # Dense Block
        for j in range(n_layers_per_block[i]):
            # Compute new feature maps
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            # And stack it : the Tiramisu is growing
            stack = concatenate([stack, l], axis=3)
            n_filters += growth_rate
        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(stack)

        # Transition Down
        stack = TransitionDown(stack, n_filters, dropout_p)

    skip_connection_list = skip_connection_list[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store now the output of the next dense block in a list. We will only upsample these new feature maps
    block_to_upsample = []

    # Dense Block
    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack, l], axis=3)

    #######################
    #   Upsampling path   #
    #######################

    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        # Dense Block
        block_to_upsample = []
        for j in range(n_layers_per_block[n_pool + i + 1]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l], axis=3)

    #####################
    #      Softmax      #
    #####################
    
    last_conv = Conv2D(n_classes, (1, 1), kernel_initializer="he_uniform", padding="same", activation='softmax', name="last_conv2D")(stack)
    
    model = Model(input=inputs, output=last_conv)
    parallel_model = multi_gpu_model(model, GPU_numb)
    parallel_model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model