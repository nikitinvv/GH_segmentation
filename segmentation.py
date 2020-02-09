from model import *
from data import *
import cv2 as cv
import h5py

test_path = 'data/HDF5/exp2_time_19p6_101_102_107to110_113to185.h5'

dataset_list = ['185']

f_all = h5py.File('data/segmentation/All.h5', 'w')
f_grains = h5py.File('data/segmentation/grains.h5', 'w')
f_wh = h5py.File('data/segmentation/water_hydrate.h5', 'w')

for i in range (len(dataset_list)):

    testGene = testGenerator_h5(test_path, dataset_name = dataset_list[i], target_size = (1024,1024))
    model = unet(input_size = (1024,1024,1))
    model.load_weights("unet_pretrain1.hdf5")
    grains = model.predict_generator(testGene,512,verbose=1)

    All = threshold_data(test_path, dataset_name = dataset_list[i], treshold_range = (80,255), target_size = (1024,1024))
    
    print('start to make hdf5 files')
    f_all.create_dataset(str(i), data = np.uint8(All*255))
    f_grains.create_dataset(str(i), data = np.uint8(grains[:,:,:,0]*255))
    f_wh.create_dataset(str(i),data =  np.uint8(All*255-grains[:,:,:,0]*255))

f_all.close()
f_grains.close()
f_wh.close()