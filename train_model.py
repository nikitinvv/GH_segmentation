from model import *
from data import *
import matplotlib.pyplot as plt
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="6"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[:], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# number of training images
N_train = 800
# number of validation images
N_valid = 400
# define list of GPUs for multimode calculation
multimode = True
GPUs = [0,1]
# batch size is the number of images that are sent to the model in one step. 
batch_size = 4*len(gpus)
# steps_per_epoch = number of training images / batch_size
steps_per_epoch = N_train/batch_size
# steps_per_epoch = number of validation images / batch_size
validation_steps = N_valid/batch_size

# initialize data generator for training and validaton data. 
data_gen_args = dict()
myGene = trainGenerator(batch_size,'data/pretrain','image','label',data_gen_args,target_size = (256,256),save_to_dir = None)
myGene_val = trainGenerator(batch_size,'data/pretrainVal','image_val','label_val',data_gen_args,target_size = (256,256),save_to_dir = None) 

# initialize DeepLearning model (U-net)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = unet(input_size=(256,256,1))

model.summary() 
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model_checkpoint = ModelCheckpoint('unet_pretrain.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit(myGene,steps_per_epoch=steps_per_epoch,epochs=100,callbacks=[model_checkpoint], validation_data=myGene_val,validation_steps=validation_steps)

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
