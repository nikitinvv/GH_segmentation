from model import *
from data import *
import matplotlib.pyplot as plt

# number of training images
N_train = 800
# number of validation images
N_valid = 400
# define list of GPUs for multimode calculation
multimode = True
GPUs = [0,1]
# batch size is the number of images that are sent to the model in one step. 
batch_size = 16 
# steps_per_epoch = number of training images / batch_size
steps_per_epoch = N_train/batch_size
# steps_per_epoch = number of validation images / batch_size
validation_steps = N_valid/batch_size

# initialize data generator for training and validaton data. 
data_gen_args = dict()
myGene = trainGenerator(batch_size,'data/pretrain','image','label',data_gen_args,target_size = (256,256),save_to_dir = None)
myGene_val = trainGenerator(batch_size,'data/pretrainVal','image_val','label_val',data_gen_args,target_size = (256,256),save_to_dir = None) 

# initialize DeepLearning model (U-net)
if (multimode): model = unet_multiGPU(input_size=(256,256,1),GPU_list=GPUs)
else: model = unet(input_size=(256,256,1))
model.summary() 
model_checkpoint = ModelCheckpoint('unet_pretrain1.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=steps_per_epoch,epochs=200,callbacks=[model_checkpoint], validation_data=myGene_val,validation_steps=validation_steps)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()