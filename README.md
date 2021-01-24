# Organic
Object Reconstruction with Generative Adversarial Networks from InterferometriC data

In order to use the image reconstruction method two scripts need to be ran.


TrainGAN.py

First a GAN network must be trained. 
This can be done by running the TrainGAN.py
In order to do so a training dataset, consisting of fits files is needed.
Both the parameters governing the training routine and the network architectures can be altered in this file.  
The directory where the trained networks are stored can also be set. 




ImageReconstruction.py 

After a GAN is trained the image reconstruction can be preformed.
This can be preformed by running the ImageReconstruction.py script.

Here the various parameters governing the reconstruction can be altered.
The direcories ofthe relevant trained neural networks must also be suplied. 

A Datalikelihood loss function must be initialized beforehand and passed to the image reconstruction function.
three datalikelihood- cost functions can be chosen 
these differ by having 
- fixed sparco paramaters
- no sparco contributions
- variable sparco parameters, which are fitted during training (yet to be implemented!!!!), this datalikelihood only works with the ImagereconstructioAndFit function.



The environment used to run the scripts is included as organic.yml 
To use the right environment type:
```
conda env create --file organic.yml
```
A number of pre-trained GAN are available.
- **theGANextended2** MCMax model circumstellar disks with extended component /data/leuven/334/vsc33405/summerjobTests/GANspirals/saved_models
- **GANspirals** geometrical pinwheel nebula models /data/leuven/334/vsc33405/summerjobTests/GANspirals/saved_models
- **GANstellarSurf** Stellar surface models /data/leuven/334/vsc33405/summerjobTests/GANstellarSurf/saved_models
- **GANstellarSurf2** Stellar surface models /data/leuven/334/vsc33405/summerjobTests/GANstellarSurf2/saved_models
