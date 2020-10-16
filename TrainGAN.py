import FunctionLibrary as lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
import glob
import os





################################################################################
################ parameters governing the training of the gan ##################
################################################################################
#imageParameters
image_Size = 128 #controls the number of pixels in the images used for training the GAN
NoiseLength = 100 # the number of elements in the noise vector used by the generator network

#Training parameters
NumberOfEpochs = 2000 # the number of iterations of taining over the intire training dataset
#a four times higher number is needed compared to the thesis,
# the data load routine has been rewriten to keep the data set the same size and choose a random angle each time a batch is sampled
BatchSize = 25 #The number of images used to calculate the gradients used to update the networks iteratively, should be shosen as large as possible give the memory constraints
PlotEpochs = 20 #Epoch interval after which examples of generated images are stored
Use1sidedLabelSmooth = True #whether or not one side label smoothning is applied during training
saveEpochs = [500,1000,1500] #epochs(integers) at which to save the network
OverTrainDiscr = 1 # the amount the discriminator is trained more than the generator in each epoch (if 2 the discriminator will be trained for twice the total dataset in an epoch)

################################################################################
############################### directories ####################################
################################################################################
#UNIFORM INCLINATIONS!!
#data_Dir = os.path.expandvars('${VSC_DATA}/CNN/grid_Large_disks/[*/Image*NOSTAR.fits') #Direcories at which the images(fits format) are located
data_Dir = '/data/leuven/333/vsc33398/processedcube03.fits'
save_dir = os.path.join(os.getcwd(), 'saved_models') #directory where the trained networks are stored
#model_name = 'version1' #name of the stored keras model, a .h5 file extension is used for the stored keras model,
                        # the component networks are stored by adding the component name in front of this string
loadFromCube = True #if true loads data from a single Fits cube, if false searches a given directory

################################################################################
############################### networks ####################################
################################################################################


def create_generator():
    generator=Sequential()

    generator.add(layers.Dense(int((image_Size/8)*(image_Size/8)*256), use_bias=False, input_shape=(NoiseLength,)))
    generator.add(layers.LeakyReLU(0.25))

    # when adding an extra layer with stride the layers above, the image size has to be devided by two an extra time and those below  once less
    generator.add(layers.Reshape((int(image_Size/8),int(image_Size /8), 256)))
    assert generator.output_shape == (None, int(image_Size/8), int(image_Size/8), 256) # Note: None is the batch size


    generator.add(layers.Conv2DTranspose(128, (4,4), strides=(2, 2), padding='same', use_bias=False))
    assert generator.output_shape == (None, int(image_Size/4), int(image_Size/4), 128)
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.25))


    generator.add(layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', use_bias=False))
    assert generator.output_shape == (None, int(image_Size/2), int(image_Size/2), 64)
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.25))

    generator.add(layers.Conv2DTranspose(48, (4,4), strides=(1, 1), padding='same', use_bias=False))
    assert generator.output_shape == (None, int(image_Size/2), int(image_Size/2), 48)
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.25))

    generator.add(layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', use_bias=False))
    assert generator.output_shape == (None, int(image_Size), int(image_Size), 32)
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU(0.25))


    generator.add(layers.Conv2D(1, (5,5), strides=(1, 1), padding='same', use_bias=False, activation='linear',kernel_initializer='glorot_normal'))
    assert generator.output_shape == (None, image_Size, image_Size, 1)
    #generator.add(layers.BatchNormalization())
    generator.add(Activation('tanh'))



    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator


def create_discriminator():
    disc = Sequential()


    disc.add(layers.Conv2D(32, (4,4), strides=(2, 2), padding='same',input_shape=[image_Size, image_Size, 1]))
    disc.add(layers.LeakyReLU(0.25))
    disc.add(layers.SpatialDropout2D(0.4))
    assert disc.output_shape == (None, int(image_Size/2), int(image_Size/2), 32)

    disc.add(layers.Conv2D(48, (4, 4), strides=(2, 2), padding='same'))
    disc.add(layers.LeakyReLU(0.25))
    disc.add(layers.SpatialDropout2D(0.4))

    disc.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    disc.add(layers.LeakyReLU(0.25))
    disc.add(layers.SpatialDropout2D(0.4))

    disc.add(layers.Conv2D(128, (4,4), strides=(2, 2), padding='same'))
    disc.add(layers.LeakyReLU(0.25))
    disc.add(layers.SpatialDropout2D(0.4))

    disc.add(layers.Flatten())
    disc.add(layers.Dense(1,activation='sigmoid'))

    disc.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return disc
optimizer = Adam(lr=0.0002, beta_1=0.5)


#ImageDataGenerator: Keras object used to preprocces the input images
#see: https://keras.io/api/preprocessing/image/ for more information
dataGen =  ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0,
        shear_range=0.,  # set range for random shear
        zoom_range=[1.5,1.7],  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=-1.,  # value used for fill_mode = "constant"
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


gen = create_generator()
print(gen.summary())
dis = create_discriminator()
print(dis.summary())


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
        Use1sidedLabelSmooth,
        saveEpochs,
        loadFromCube,
        dataGen
        )
