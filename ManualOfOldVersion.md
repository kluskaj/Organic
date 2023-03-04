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
  
Image preprocessing:
  * **dataGen**  an ImageDataGenerator used to preprocess the images from the training dataset. This preprocessing is applied after the images are rescaled using Bilinear interpolation. The documentation for this type of object can be found [here]( https://keras.io/api/preprocessing/image/#imagedatagenerator-class ).
  * one of the options in the ImageDataGenerator is to set a costum **preprocessing_function**. Two such options are already profided in the FunctionLibrary.py file. these are:
    * **preprocesfunc5** : This adds a 1 in 5 chance to add a uniform backgroud randomly selected from a uniform distribution between 0 and 0.1 relative flux to an image uppon sampling. The Image is renormalized to have a maximum flux equal to 1 after this is is done.
    * **preprocesfuncMax1** sets th maximum of the image to 1 after the rotation/zoom preformed in the ImageDataGenerator.
  
  
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
  * **resetOpt** : Boolean. If true the optimizer is reset when the generator is reset, if false the state is carried over.
*
  * **ShiftPhotoCenter** If true ORGANIC centers the photocenter of the images, this is also the default value, so it does not expresly need to be set. This option is automaticly set to false when using Fixed sparco parameters.
  * UseRoll if True the integer photocenter offset is removed by "rolling" the image. If false this is not done and the pixels shifted into the image have value of 0 rel. flux
  * Interp Defines the type of interpolation to be used for the photocenter centering. Options are: 'BILINEAR' and 'NEAREST'
* Load the data used in the image reconstruction.
  * **dirV2** : Directory of a numpy array containing artificial squared visibility values, if one wishes to use these. The corresponding baselines and errorvalues are addopted from the used real dataset.
  * **dirCP** : Directory of a numpy array containing artificial closure phase values, if one wishes to use these. The corresponding baselines and errorvalues are addopted from the used real dataset.
  * The artificial data values are loaded on the lines following the directories. Leave this as is for ease of use.
  * **DataDir** Directory of the OIfits file(s) containing the data for the image reconstructions.
  * **filename** :  name of the OIfits file(s) in case of real data. In order to use data from multiple files use an expression like: `<'*.fits'>` 
  
* set the SPARCO Parameters
  * **x** : the right-ascention of a point source, to be modeled using SPARCO, in MAS.
  * **y** :  the declination of a point source, to be modeled using SPARCO, in MAS.
  * **UDflux** : The flux contribution of the a resolved central star, represented as a uniform disk (set to 0 for an unresloved point source), in percent
  * **PointFlux** : The flux contribution of a point source star, in percent
  * **denv** : the spectral index of the environment
  * **dsec** :  the spectral index of the point source (the uniform disk source has a default index of -4).
  * **UDdiameter** : The diameter of the resolved source in mas
  * **wavel0** The reference wavelength for sparco, default is 1.65e-6 $\mu m$
  
* run the image reconstrution
  * on line 78  a **framework** object is created using the parameters set using the previously set options.
  * on line 81 the SPARCO parameters of the reconstruction are set. If you do not wish to use SPARCO, do not run this line.
  * on line 84 the artificial squared visibility and closure phases are set. If you do not wish to use artificial data from numpy arrays, do not run this line.
  * The following three run options are present as an atribute of the framework object: 
    * **ImageReconstruction** : runs a single image reconstruction for the chosen parameters, by computing the average of a number of restarts
    * **bootstrappingReconstr** : Runs a boothstrapping of the image reconstruction. In this boothstrapping the closure phases and squared visibilities are treated independently.
    * For ease of use these first two run options can be uncommented, when this is done they use the parameters set in the 
    * **runGrid** : Runs ImageReconstruction for all combinations of the given parameters. The parameters are that can be set are:
        * nrRestarts: list with the amounts of restrarts for which to run the grid.
        * epochs: list with the number of epochs to use in differetn runs of the grid
        * mus: list with the values of hyperparameter for which to run the grid
        * kwargs: lists of other parameters to alter during the runs of the grid.
                  These can be the sparco parameters and pixelsize.
           
* **tips for Using ORGANIC**
  * When using fixed sparco parameters and PIONIER data for Circumstellar disks work best with the default parameters given in the ImageReconstruction.py.
  * MATISSE data of IRAS08 gave bad results, alterations will likely need to be made to accomodate this data. Altough experimenting with the learning rate and number of epochs may also help.
  * For reconstructions without SPARCO where the photocenter shift needs to be applied a lower learning rate, higher number of epochs and use of bilinear interpolation are beneficial. The use of bilinear interpolation greatly reduces the amount of checkerboard artifacts. 
  * For stellar surfaces the GANs display a horizontal/vertical waffle-like pattern on the stellar surface. a better GAN likely needs to be trained, doing so may require a larger number of models, or a stronger discriminator. It is however also possible that the seen structure is a consequence of the convolutional architecture used. An alternative method may be found by fitting a variational autoencoder. 
  * A lower value of **mu** does not nececerely provide a better fit to the data. A likely explanation for this is that the noise introduced by the dropout active in the discriminator helps with the optimizaton procedure. Adding droput layers to the generator (turned of during the GAN training and on during the image reconstruction) may be worth trying for future experiments.
  * ORGANIC produces different results for different pre-trained GANs, parameters will likely need to be re-adjusted when using a different GAN
  * ORGANIC lacks theoretical considerations, use at your own risk
  






