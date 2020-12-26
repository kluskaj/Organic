Object Reconstruction with Generative Adversarial Networks from InterferometriC data

In order to reconstruct images a generative adversarial neural network (GAN) must be trained. 
This is phase 1 of training.
This training can be done by running the file:
  
  **TrainGAN.py**
  
running this script on a HPC can be done using the

  **TrainGan.pbs**

shell script.
In this file the used resourse are set in the header, an email address can also be added here to notify you if your job starts running. 
On line 13 the path to miniconda is exported. The environment containing all the neccesary python packages is activated on line 14. 
This shell script also creates a directory where the results of this pretraining phase will be stored, this directories need to be changed on line 15 and 16.
Lines 17 and 18 copy TrainGAN.py and the function library to the created directory allowing it to be ran from that location and to parameters used for a run to be reviewed afterwards.

The following sections are present in the **TrainGAN.py**:

* Network parameters:
   these parameters are used to determine the neural network architectures in the functions **create_generator()** and **create_discriminator()**
  * **image_Size** : is used to scale the networks to the amount of pixels in the images, aswell as set the pixel size to which the training data images are rescaled using biliniear interpolation.
  * **NoiseLength** : set the length of the input noise vector to te generator.

* Training parameters:
these parameters determine how the GAN is trained
  * **NumberOfEpochs** : the number of iterations of taining over the intire training dataset
  * **BatchSize** : The number of images used to calculate the gradients used to update the networks iteratively, should be shosen as large as possible give the memory constraints
  * **PlotEpochs** :Epoch interval after which examples of generated images are stored
  * **Use1sidedLabelSmooth** : whether or not one-sided label smoothning is applied during training, for an explanation on onesided label smoothening see:
  * **saveEpochs** : epochs at which to save the networks
  * **OverTrainDiscr** : the amount the discriminator is trained more than the generator in each epoch (if 2 the discriminator will be trained for twice the total dataset in an epoch)
 
* Directories:
  * **data_Dir** :  `< os.path.expandvars('${VSC_DATA}/CNN/grid_large_disks_unifCosi/[*/Image*NOSTAR.fits') Direcories at which the images(fits format) are located >`
  * **save_dir** :  #directory where the trained networks are stored `<os.path.join(os.getcwd(), 'saved_models')>` is used to store the networks in a folder named ##saved_models##
  * **loadFromCube** : Bollean, if true loads data from a single Fits cube, if false searches a given directory

* Networks
  * In this section the architecture of the generator and discriminator networks are defined. The example networks given in the file can be addapted for different image and inputnoise sizes using the parameters listed under section ##Network parameters##. In order to understand the networks defined here, or to implement your own neural network architectures, see the [Keras documentation](https://keras.io/api/).
  * summarys of the used networks are printed in the output (.out) file on lines 162 to 165.
  
* Perform training
  * Here the training routine is called from the function library for al of the parameters set in the previous sections of the file. Leave this as is for ease of use.
  
After a GAN is trained the image reconstruction can be preformed. This can be preformed by running the
  ImageReconstruction.py
script. running this script on a HPC can be done using the 
  reconstructImages.pbs 
shell script. 
In this file the used resourse are set in the header, an email address can also be added here to notify you if your job starts running.
On line 13 the path to miniconda is exported. The environment containing all the neccesary python packages is activated on line 14. 
This shell script also creates a directory where the results of this pretraining phase will be stored, this directories need to be changed on line 15 and 16.
Lines 17 and 18 copy TrainGAN.py and the function library to the created directory allowing for it to be ran from that location and to parameters used for a run to be reviewed afterwards.

The following sections are present in **ImageReconstruction.py**

* Directories of the pretrained neural networks:
  * Here the directories of the pretrained generator and discriminator need to be altered.
  * **discPath** is used to load the discriminator and **genPath** to load the generator.
  * The networks are loaded on the lines following the directories. Leave this as is for ease of use.
* Parameters for the reconstruction
  * **hyperParam** : The hyperparameter tuning the strength of the regularization
  * **image_Size** : Pixel size of the image which are created (needs to be the same as used during GAN training).
  * **NoiseLength** : Length of the inputvector suplied to the generator (needs to be the same as used during GAN training).
  * **epochs** : The number of iterations within a single optimazation of the generator.
  * **numberOfRestarts** : The number of times to restart the image computation for a different noise vector.
  * **BoothstrapIter** : The number of times to alter the dataset during Boothstrapping. Only used when bootsrapping 
  * **pixelSize** : Angular size of a pixel size in milli arc seconds (mas).
  * **optimizer** : The optimizer to be used for fine tuning the the generator. This optimizer is given as a dictionary and is reset for each reset of the generator training.
* Load the data used in the image reconstruction.
  * **dirV2** : Directory of a numpy array containing artificial V2 values, if one wishes to use these. The corresponding baselines and errorvalues are addopted from the used real dataset.
  * **dirCP** : Directory of a numpy array containing artificial V2 values, if one wishes to use these. The corresponding baselines and errorvalues are addopted from the used real dataset.
  * simCP =np.load(dirCP)
  * **DataDir** = os.path.expandvars('${VSC_DATA}/summerjobTests/CLLac_data/') directory of the OIfits file in case of real data
  * **filename** : '*.fits' directory and name of the OIfits file in case of real data
  






