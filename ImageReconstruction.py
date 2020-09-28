
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
import FunctionLibrary as lib
import tensorflow as tf
from FunctionLibrary import framework
################################################################################
################ parameters for the reconstruction #############################
################################################################################
#load the discriminator obtaned during phase 1
discPath = os.path.expandvars('${VSC_DATA}/summerjobTests/theGAN/saved_models/discriminatorfinalModel.h5')
discriminator = load_model(discPath)

#load the Generator obtaned during phase 1
genPath = os.path.expandvars('${VSC_DATA}/summerjobTests/theGAN/saved_models/generatorfinalModel.h5')
Generator= load_model(genPath)


hyperParam = 1 # the hyperParam tuning the strength of the regularization
image_Size = 128 #pixel size of the image which are created (needs to be the same as used during GAN training)
NoiseLength = 100 #length of the inputvector suplied to the generator
epochs = 250 #the number of iterations
numberOfRestarts = 100 #the number of times to restart the image computation for a different noise vector
BoothstrappinIter = 100 #the number of times to alter the dataset during Boothstrapping

#define the optimizer used to finetune the generator
optimizer=Adam(learning_rate=0.0001,beta_1=0.91,beta_2=0.999,amsgrad=False)

#lib.plot_generated_images(1, Generator,NoiseLength,image_Size, examples=16, dim=(4, 4), figsize=(7, 7))

#quit()
#load artificial datasets in a numpy format
dirV2 =os.path.expandvars('${VSC_DATA}/summerjobTests/ArtificialDatasets/V2Model_gausianNoise_HD_baselines.npy')
simV2 =None #np.load(dirV2)
dirCP =os.path.expandvars('${VSC_DATA}/summerjobTests/ArtificialDatasets/CPModel_gausianNoise_HD_baselines.npy')
simCP =None #np.load(dirCP)
#directoryand name of the OIfits file in case of real data
DataDir = os.path.expandvars('${VSC_DATA}/CNN/OIfits/')
filename = 'IRAS08544-4431_PIONIER_alloidata.fits'#'HD45677all.fits'#




################################################################################
################ choose a data likelilhood loss term ###########################
################################################################################

#options are:
    #no sparco
    #fixed sparco parameters
    #TODO: sparco fitting(requires a second neural net to be passed to the reconstruction)


#sparco Parameters
x =-0.44# 0#   #the right-ascention of a point source star, to be removed using sparco
y = -0.68#0#   #the declination of a point source star, to be removed using sparco
UDflux = 59.7#0# #the flux (percent) contribution of the a central resolved star, represented as a uniform disk (set to 0 for an unresloved point source)
PointFlux = 3.9#43.5# # The flux contribution of a point source star, in percent
denv = 0.42#1.73# # the spectral index of the environment
dsec =-2#-4 # #  the spectral index of the point source star (the uniform disk source has a default index of 4)
UDdiameter = 0.5# # the diameter of the resolved source
pixelSize = 0.6#0.546875 #0.5#0.6pixel size in mas
#dataLikelihood = lib.dataLikeloss_FixedSparco(DataDir,filename,image_Size,
#                                                x,
#                                                y,
#                                                UDflux,
#                                                PointFlux,
#                                                denv,
#                                                dsec,
#                                                UDdiameter,
#                                                pixelSize,
#                                                V2Artificial = simV2,CPArtificial = simCP)


#dataLikelihood for stellar surfaces
#dataLikelihood = lib.dataLikeloss_NoSparco(DataDir,filename,image_Size,pixelSize)


################################################################################
################## run  the image reconstrution ################################
################################################################################


reconstr = framework(DataDir,filename,image_Size,pixelSize,Generator,discriminator,optimizer,NoiseLength)
reconstr.setSparco(x,y,UDflux,PointFlux,denv,dsec,UDdiameter)

#use this to use artificial V2 and CP from numpy arrays
#reconstr.useArtificialDataNP(simV2,simCP)

#Run this line for a single image reconstruction
#mean = reconstr.AveragingImageReconstruction(numberOfRestarts,epochs,hyperParam)

#Run this line for Boothstrapping
#mean, varianceImage = reconstr.bootstrappingReconstr(BoothstrappinIter,numberOfRestarts,epochs,hyperParam)

#Run this line to create a grid

reconstr.runGrid(nrRestarts =[numberOfRestarts],epochs = [epochs], mus = [1,4],pixelSize=[0.6,0.5],dsec = [-2,0])

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
    ,V2Artificial = simV2#None
    ,CPArtificial = simCP#None
)

quit()
#alter the output range  convert the mean to a tensorflow objectand, give the correct shape, these are also added in the header of the .fits files
imgNP = (mean*2)-1
img = tf.constant(imgNP)
img = tf.reshape(img,[1,image_Size,image_Size,1])

lossValue, V2loss , CPloss = dataLikelihood(None,img)
print('the total reduced chi squared')
print(lossValue.numpy())
print('the squared visibility reduced chi squared')
print(V2loss.numpy())
print('the closure phases reduced chi squared')
print(CPloss.numpy())

#print the regularization value of the mean image
score = discriminator.predict(imgNP.reshape(1,image_Size,image_Size,1))
print('The f prior value is')
print(-np.log(score))
#Create a fits file for the final image
