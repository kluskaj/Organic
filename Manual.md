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

-Network parameters:
  --these parameters are used to determine the neural network architectures, and are used in the functions **create_generator()** and **create_discriminator**
  --**image_Size
  






