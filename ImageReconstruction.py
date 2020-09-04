
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import FunctionLibrary as lib
import tensorflow as tf

################################################################################
################ parameters for the reconstruction #############################
################################################################################
discPath = os.path.expandvars('${VSC_DATA}/summerjobTests/theGAN/saved_models/discriminatorfinalModel.h5')
discriminator = load_model(discPath)


genPath = os.path.expandvars('${VSC_DATA}/summerjobTests/theGAN/saved_models/generatorfinalModel.h5')
Generator= load_model(genPath)
hyperParam = 1
Begin_avraging = 251
image_Size = 128
NoiseLength = 100
RandomWalkStepSize = 0.5
alterationInterval = 1
plotinterval = 300 #set to a multiple of the alteration interval to see the effect of noise vector change
epochs = 251
numberOfRestarts =100
optimizer=Adam(learning_rate=0.0001,beta_1=0.91,beta_2=0.999,amsgrad=False)

#load artificial datasets in a numpy format
dirV2 =os.path.expandvars('${VSC_DATA}/summerjobTests/ArtificialDatasets/V2Model_gausianNoise_IRAS_baselines.npy')
simV2 =None#np.load(dirV2)
dirCP =os.path.expandvars('${VSC_DATA}/summerjobTests/ArtificialDatasets/CPModel_gausianNoise_IRAS_baselines.npy')
simCP =None#np.load(dirCP)
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


#sparco Parameters
x = -0.44   #the right-ascention of a point source star, to be removed using sparco
y = -0.68   #the declination of a point source star, to be removed using sparco
UDflux = 59.7 #the flux contribution of the a central resolved star, represented as a uniform disk (set to 0 for an unresloved point source)
PointFlux = 3.9 # The flux contribution of a point source star
denv = 0.42 # the spectral index of the environment
dsec = -2, #  the spectral index of the point source star (the uniform disk source has a default index of 4)
UDdiameter = 0.5 # the diameter of the resolved source
pixelSize = 0.546875 #pixel size in mas
dataLikelihood = lib.dataLikeloss_FixedSparco(DataDir,filename,image_Size,
                                                x,
                                                y,
                                                UDflux,
                                                PointFlux,
                                                denv,
                                                dsec,
                                                UDdiameter,
                                                pixelSize,
                                                V2Artificial = simV2,CPArtificial = simCP)


#dataLikelihood for stellar surfaces
#dataLikelihood = lib.dataLikeloss_NoSparco(DataDir,filename,image_Size,pixelSize)


################################################################################
################## run  the image reconstrution ################################
################################################################################

#mean, varianceImage, diskyLoss, fitLoss = lib.reconsruction(Generator, discriminator,optimizer,dataLikelihood ,pixelSize, epochs ,image_Size ,hyperParam,NoiseLength,Begin_avraging ,RandomWalkStepSize,alterationInterval,plotinterval,saveDir  = '')
mean, varianceImage = lib.restartingImageReconstruction(numberOfRestarts,Generator, discriminator,optimizer,dataLikelihood ,pixelSize, epochs ,image_Size ,hyperParam,NoiseLength,Begin_avraging ,RandomWalkStepSize,alterationInterval,plotinterval)

# store the mean and variance image
np.save('meanImage',mean)
np.save('varianceImage',varianceImage)
#store the numpy arrays for further use
#np.save('diskyLoss',diskyLoss)
#np.save('fitLoss',fitLoss)


#print the data likelihood terms for the reconstructed images
# first set up the dataLikelihood to return all of its components
dataLikelihood = lib.dataLikeloss_FixedSparco(DataDir,filename,image_Size,
    x,
    y,
    UDflux,
    PointFlux,
    denv,
    dsec,
    UDdiameter,
    pixelSize,
    forTraining = False
    ,V2Artificial = simV2,CPArtificial = simCP
)
#alter the output range  convert the mean to a tensorflow objectand, give the correct shape
imgNP = (mean*2)-1
img = tf.constant(imgNP)
img = tf.reshape(img,[1,image_Size,image_Size,1])
lossValue, V2loss , CPloss = dataLikelihood(None,img)
print('the total reduced chi squared')
print(lossValue.numpy())
print('the squared visibilities reduced chi squared')
print(V2loss.numpy())
print('the closure phases reduced chi squared')
print(CPloss.numpy())

#print the regularization value of the mean image
score = discriminator.predict(imgNP.reshape(1,image_Size,image_Size,1))
print('The f prior value is')
print(-np.log(score))
#Create a fits file for the final image
lib.toFits(mean,image_Size,pixelSize,os.path.basename(os.getcwd()),comment= "")
