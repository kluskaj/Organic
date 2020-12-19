Object Reconstruction with Generative Adversarial Networks from InterferometriC data

In order to reconstruct images a generative adversarial neural network must be trained. 
This is phase 1 of training.
This training can be done by running the file:
  
  TrainGAN.py
  
running this script on a HPC can be done using the TrainGan.pbs shell script.
In this file the used resourse are set in the header, an email address can also be added here to notify you if your job starts running.
This shell script also creates a directory where the results of this pretraining phase will be stored, this directories need to be changed on line 16 and 17.
The final two lines first copy TrainGAN.py to the 




