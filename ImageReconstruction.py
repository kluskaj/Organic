
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import FunctionLibrary as lib
################################################################################
################ parameters for the reconstruction #############################
################################################################################
discPath = os.path.expandvars('${VSC_DATA}/CNN/unifCosi_basicGAN_multi4/saved_model_test/discriminatorGAN_CNN_test.h5')
discriminator = load_model(discPath)


genPath = os.path.expandvars('${VSC_DATA}/CNN/unifCosi_basicGAN_multi4/saved_model_test/generatorGAN_CNN_test.h5')
Generator= load_model(genPath)
hyperParam = 2
Begin_avraging =9000
image_Size = 256
NoiseLength = 150
RandomWalkStepSize = 0.5
alterationInterval = 500
plotinterval = 3000
epochs = 30000
optimizer=Adam(learning_rate=0.0001,amsgrad=False)
#in order to load artificial datasets in a numpy format
dirV2 =os.path.expandvars('${VSC_DATA}/CNN/TrainingDoubleLossGen/V2ModelImage6_gausianNoise_IRAS08544baselines.npy')
simV2 = np.load(dirV2)
dirCP =os.path.expandvars('${VSC_DATA}/CNN/TrainingDoubleLossGen/CPModelImage6_gausianNoise_IRAS085442baselines.npy')
simCP = np.load(dirCP)
#directoryand name of the OIfits file in case of real data
DataDir = os.path.expandvars('${VSC_DATA}/CNN/OIfits/')
filename = 'IRAS08544-4431_PIONIER_alloidata.fits'



################################################################################
################ choose a data likelilhood loss term ###########################
################################################################################

#options are:
    #no sparco
    #fixed sparco parameters
    #TODO: sparco fitting(requires a second neural net to be passed to the reconstruction)
    #TODO: fourier space GAN/LOSS/reconstruction

#sparco Parameters
x = -0.44   #the right-ascention of a point source star, to be removed using sparco
y = -0.68   #the declination of a point source star, to be removed using sparco
UDflux =59.7 #the flux contribution of the a central resolved star, represented as a uniform disk (set to 0 for an unresloved point source)
PointFlux = 3.9 # The flux contribution of a point source star
denv = 0.42 # the spectral index of the environment
dsec = -2, #  the spectral index of the point source star (the uniform disk source has a default index of 4)
UDdiameter = 0.5 # the diameter of the resolved source
pixelSize = 0.2734375 #pixel size in mas (currently same as thesis, yields an fov for an imagesize of 256)

dataLikelihood = lib.dataLikeloss_FixedSparco(DataDir,filename,image_Size,
x,
y,
UDflux,
PointFlux,
denv,
dsec,
UDdiameteraryDiameter,
pixelSize,
V2Artificial = None,CPArtificial = None)

################################################################################
################## run  the image reconstrution ################################
################################################################################

mean, Image, diskyLoss, fitLoss = lib.reconsruction(Generator, discriminator,optimizer,dataLikelihood , epochs ,image_Size ,hyperParam,NoiseLength,Begin_avraging ,RandomWalkStepSize,alterationInterval,plotinterval,saveDir  = '')


# store the mean and variance image
np.save('meanImage',mean)
np.save('varianceImage',Image)
#store the numpy arrays for further use
np.save('diskyLoss',diskyLoss)
np.save('fitLoss',fitLoss)


#print the data likelihood terms for the reconstructed images
# first set up the dataLikelihood to return all of its components
dataLikelihood = lib.dataLikeloss_FixedSparco(DataDir,filename,image_Size,
    x,
    y,
    UDflux,
    PointFlux,
    denv,
    dsec,
    UDdiameteraryDiameter,
    pixelSize,
    forTraining = False
)
#alter the output range  convert the mean to a tensorflow objectand, give the correct shape
img = (mean*2)-1 
img = tf.constant(img)
img = tf.reshape(img,[1,image_Size,image_Size,1])
print(dataLikelihood(None,img).numpy())
