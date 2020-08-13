import FunctionLibrary as lib


import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,LeakyReLU
from tensorflow.keras.optimizers import Adam
import glob
import os





################################################################################
################ parameters governing the training of the gan ##################
################################################################################
#imageParameters
image_Size = 128 #controls the number of pixels in the images used for training the GAN
NoiseLength = 150 # the number of elements in the noise vector used by the generator network

#Training parameters
NumberOfEpochs = 200 # the number of iterations of taining over the intire training dataset
#a four times higher number is needed compared to the thesis,
# the data load routine has been rewriten to keep the data set the same size and choose a random angle each time a batch is sampled
BatchSize = 30 #The number of images used to calculate the gradients used to update the networks iteratively, should be shosen as large as possible give the memory constraints
PlotEpochs = 5 #Epoch interval after which examples of generated images are stored
Use1sidedLabelSmooth = True #whether or not one side label smoothning is applied during training
saveEpochs = [] #epochs(integers) at which to save the network
OverTrainDiscr = 1 # the amount the discriminator is trained more than the generator in each epoch (if 2 the discriminator will be trained for twice the total dataset in an epoch)

################################################################################
############################### directories ####################################
################################################################################
data_Dir = os.path.expandvars('${VSC_DATA}/CNN/grid_large_disks_unifCosi/[*/Image*NOSTAR.fits') #Direcories at which the images(fits format) are located
save_dir = os.path.join(os.getcwd(), 'saved_models') #directory where the trained networks are stored
#model_name = 'version1' #name of the stored keras model, a .h5 file extension is used for the stored keras model,
                        # the component networks are stored by adding the component name in front of this string


################################################################################
############################### networks ####################################
################################################################################


def create_generator():
    generator=Sequential()

    generator.add(layers.Dense(int((image_Size/4)*(image_Size/4)*128), use_bias=False, input_shape=(NoiseLength,)))
    generator.add(Activation('relu'))

    # when adding an extra layer with stride the layers above, the image size has to be devided by two an extra time and those below  once less
    generator.add(layers.Reshape((int(image_Size/4),int(image_Size /4), 128)))
    assert generator.output_shape == (None, int(image_Size/4), int(image_Size/4), 128) # Note: None is the batch size


    generator.add(layers.Conv2DTranspose(64, (4,4), strides=(1, 1), padding='same', use_bias=False,kernel_initializer='glorot_normal'))
    assert generator.output_shape == (None, int(image_Size/4), int(image_Size/4), 64)
    generator.add(layers.BatchNormalization())
    generator.add(Activation('relu'))


    generator.add(layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', use_bias=False,
                            kernel_initializer='glorot_normal'))
    assert generator.output_shape == (None, int(image_Size/2), int(image_Size/2), 32)
    generator.add(layers.BatchNormalization())
    generator.add(Activation('relu'))


    generator.add(layers.Conv2DTranspose(1, (4,4), strides=(2, 2), padding='same', use_bias=False, activation='linear',kernel_initializer='glorot_normal'))
    assert generator.output_shape == (None, image_Size, image_Size, 1)
    generator.add(layers.BatchNormalization())
    generator.add(Activation('tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def create_discriminator():
    disc = Sequential()
    disc.add(layers.Conv2D(32, (4,4), strides=(2, 2), padding='same',input_shape=[image_Size, image_Size, 1]))
    disc.add(layers.LeakyReLU(0.2))
    disc.add(layers.Dropout(0.3))
    assert disc.output_shape == (None, int(image_Size/2), int(image_Size/2), 32)


    disc.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    disc.add(layers.LeakyReLU(0.2))
    disc.add(layers.Dropout(0.3))

    disc.add(layers.Conv2D(128, (4, 4), strides=(1, 1), padding='same'))
    disc.add(layers.LeakyReLU(0.2))
    disc.add(layers.Dropout(0.3))

    disc.add(layers.Flatten())
    disc.add(layers.Dense(1,activation='sigmoid'))

    disc.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return disc
optimizer = Adam(lr=0.0002, beta_1=0.5)

################################################################################
###################### preform  training #######################################
################################################################################
lib.classicalGANtraining(create_generator(),create_discriminator(), #A compiled generator network # A compiled discriminator network
        optimizer,
        data_Dir,
        image_Size,
        NoiseLength,
        NumberOfEpochs,
        BatchSize,
        OverTrainDiscr,
        save_dir,
        PlotEpochs,
        Use1sidedLabelSmooth)
