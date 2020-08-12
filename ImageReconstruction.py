
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
epochs = 21000
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
x = -0.44    #coordinates of the secondary in radian
y = -0.68
primFlux =59.7/100, #percentage of total flux contributed by primary
secFlux = 0,#3.9/100 #percentage of total flux contributed by secondary
denv = 0.42,
dsec = -2,
primaryDiameter = 0.5,
spacialFreqPerPixel = ,

dataLikelihood = lib.dataLikeloss_FixedSparco(DataDir,filename,image_Size,
x = -0.44   *np.pi*0.001/(3600*180),  #coordinates of the secondary in radian
y = -0.68   *np.pi*0.001/(3600*180),
primFlux =59.7/100, #percentage of total flux contributed by primary
secFlux = 0,#3.9/100 #percentage of total flux contributed by secondary
denv = 0.42,
dsec = -2,
primaryDiameter = 0.5 * np.pi*0.001/(3600*180),
spacialFreqPerPixel = (3600/0.07)*(180/np.pi),
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
#!!!!!!!!!!!!!!!!!!!!!!! todo keep batchsize at one for the reconsruction!!!!!!!!!!!!!!
