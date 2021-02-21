
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
import FunctionLibrary as lib
import tensorflow as tf
from FunctionLibrary import framework
################################################################################
############# load pretrained neural networks #####################
################################################################################

discPath = os.path.expandvars('${VSC_DATA}/summerjobTests/theGANextended2/saved_models/discriminatorfinalModel.h5')
discriminator = load_model(discPath)

genPath = os.path.expandvars('${VSC_DATA}/summerjobTests/theGANextended2/saved_models/generatorfinalModel.h5')
Generator= load_model(genPath)

################################################################################
################ parameters for the reconstruction #############################
################################################################################
hyperParam = 3 # the hyperParam tuning the strength of the regularization
image_Size = 128 #pixel size of the image which are created (needs to be the same as used during GAN training)
NoiseLength = 100 #length of the inputvector suplied to the generator
epochs = 250 #the number of iterations
numberOfRestarts = 200 #the number of times to restart the image computation for a different noise vector
BoothstrapIter = 100 #the number of times to alter the dataset during Boothstrapping
pixelSize = 0.6#0.1#0.1#0.546875#0.0625#0.6# #0.5#pixel size in mas
resetOpt = False # If true the optimizer is reset when the generator is reset, if false the state is carried over.
#define the optimizer used to finetune the generator
optimizer = {'name': 'Adam', 'learning_rate': 0.0002,  'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07}
########### Parameters for Photocenter-centering ##############""
ShiftPhotoCenter = True #If true ORGANIC centers the photocenter of the images, this is also the default value, so it does not expresly need to be set.
#this option is outomaticly set to false when using Fixed sparco parameters
UseRoll=True #if True the integer photocenter offset is removed by "rolling" the image. If false this is not done and the pixels shifted into the image have value of 0 rel. flux
Interp='BILINEAR' #Difines the type of interpolation to be used for the photocenter centering. Options are: 'BILINEAR' and 'NEAREST'



#used to plot generator output for random input noisevectors
#lib.plot_generated_images(1, Generator,NoiseLength,image_Size, examples=16, dim=(4, 4), figsize=(7, 7))
#quit()

################################################################################
################ load the data used in the image reconstruction ################
################################################################################
dirV2 =os.path.expandvars('${VSC_DATA}/summerjobTests/ArtificialDataSets/V2Model_CLLac_clip_baselines.npy')
simV2 =np.load(dirV2)
dirCP =os.path.expandvars('${VSC_DATA}/summerjobTests/ArtificialDataSets/CPModel_CLLac_clip_baselines.npy')
simCP =np.load(dirCP)
#directoryand name of the OIfits file in case of real data
#DataDir = os.path.expandvars('${VSC_DATA}/CNN/OIfits/')
DataDir = os.path.expandvars('${VSC_DATA}/summerjobTests/CLLac_data/')
filename = 'HD45677all.fits'#'*.fits'#'HD45677all.fits'#'IRAS08544-4431_PIONIER_alloidata.fits'#




################################################################################
################ set the sparco parameters if used #############################
################################################################################

#sparco Parameters
x =-0.44# 0#   #the right-ascention of a point source star, to be removed using sparco
y = -0.68#0#   #the declination of a point source star, to be removed using sparco
UDflux = 0#59.7#0# #the flux (percent) contribution of the a central resolved star, represented as a uniform disk (set to 0 for an unresloved point source)
PointFlux = 43.5#0#3.9#43.5# # The flux contribution of a point source star, in percent
denv = 1.73#0.42#1.73# # the spectral index of the environment
dsec = -4 #-2# #  the spectral index of the point source star (the uniform disk source has a default index of -4)
UDdiameter = 0.5# # the diameter of the resolved source
wavel0 = 1.65e-6  # The reference wavelength for sparco





################################################################################
################## run  the image reconstrution ################################
################################################################################


reconstr = framework(DataDir,filename,image_Size,pixelSize,Generator,discriminator,optimizer,NoiseLength,resetOpt,useRoll=UseRoll,interp=Interp)#,shiftPhotoCenter=ShiftPhotoCenter

#set the parameters for the
reconstr.setSparco(x,y,UDflux,PointFlux,denv,dsec,UDdiameter,wavel0,useLowCPapprox = True)

#use this to use artificial V2 and CP from numpy arrays
#reconstr.useArtificialDataNP(simV2,simCP)

#Run this line for a single image reconstruction
#median = reconstr.ImageReconstruction(numberOfRestarts,epochs,hyperParam=hyperParam,plotAtEpoch = [1],loud=True)

#Run this line for Boothstrapping
#mean, varianceImage = reconstr.bootstrappingReconstr(BoothstrapIter,numberOfRestarts,epochs,hyperParam,plotAtEpoch = [])

#Run this line to create a grid for the chosen parameters
reconstr.runGrid(nrRestarts =[numberOfRestarts],epochs = [epochs], mus = [10,1,0.01,0.000001,0.001,0.1,2,5,100],pixelSize=[pixelSize,0.5])
