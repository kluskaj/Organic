Object Reconstruction with Generative Adversarial Networks from InterferometriC data

In order to reconstruct images a generative adversarial neural network must be trained. 
This is phase 1 of training.
This training can be done by running the file:
  
  TrainGAN.py
  
running this script on a HPC can be done using the

  TrainGan.pbs

shell script.
In this file the used resourse are set in the header, an email address can also be added here to notify you if your job starts running.
This shell script also creates a directory where the results of this pretraining phase will be stored, this directories need to be changed on line 16 and 17.
Lines 17 and 18 copy TrainGAN.py and the function library to the created directory allowing to parameters used for a run to be reviewed afterwards.

the following sections are present in the TrainGAN.py:

* Network parameters:
   these parameters are used to determine the neural network architectures in the functions **create_generator()** and **create_discriminator()**
  * **image_Size** is used to scale the networks to the amount of pixels in the images, aswell as set the pixel size to which the training data images are rescaled using biliniear interpolation.
  * **NoiseLength** set the length of the input noise vector to te generator.

* Training parameters:
  * **NumberOfEpochs** the number of iterations of taining over the intire training dataset
  * **BatchSize** The number of images used to calculate the gradients used to update the networks iteratively, should be shosen as large as possible give the memory constraints
  * **PlotEpochs** = 25 Epoch interval after which examples of generated images are stored
  * **Use1sidedLabelSmooth** whether or not one-sided label smoothning is applied during training, for an explanation on onesided label smoothening see:
  * **saveEpochs** epochs at which to save the networks
  * **OverTrainDiscr** the amount the discriminator is trained more than the generator in each epoch (if 2 the discriminator will be trained for twice the total dataset in an epoch)
  






