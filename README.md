# Organic
**O**bject **R**econstruction with **G**enerative **A**dversarial **N**etworks from **I**nterferometri**C** data
You can find the associated paper [here](https://ui.adsabs.harvard.edu/abs/2020SPIE11446E..1UC/abstract)

## Purpose

The goal of this image reconstruction algorithm is to use an astropbysical Bayeisan prior to reconstruct an image from optical (infrared and visible) interferometric data.
Neural networks can learn the main characteristics of the image from the models and guide the image reconstruction to a similar images. The data itself can induce more complex features if they are needed to reproduce the data.

## Installation

To install Organic you should type this in your terminal:
```
conda create -n organic python=3.9
conda activate
pip install organicoi
```

It is strongly advised to also install postOrganic to analyse the outputs from [Organic](https://github.com/kluskaj/postORGANIC).
You can find it [here](https://github.com/kluskaj/postORGANIC)


## Image reconstruction

To perform image reconstruction you can use the example file `ImgRecExample.py` file in the examples folder.
The steps are summarised below:

1.  Loading the Neural Network
Set the paths and name of the neural network discriminator (`dis`) and generator (`gen`).
Load the Neural Network:
```
thegan = org.GAN(dis=dis, gen=gen, Adam_lr=0.0001, Adam_beta_1=0.91, amsgrad=True)
```
`Adam_lr` is the learning rate.
`Adam_beta_1` is the exponential decay of the first moment estimates.
`amsgrad` is whether to apply AMSGrad variant of Adam.
More informatino on the Adam optimizer can be found [here](https://keras.io/api/optimizers/adam/)

2. Set the SPARCO parameters
SPARCO is an approach allowing to model the star(s) as a geometrical model and image the environment only.
This improves the image of the stellar environment and takes into account the spectral index difference between the stars and the environment.
```
sparco = org.sparco(fstar=0.61, fstar=0.6, dstar=-4.0, denv=0.0, UDstar=0.01, fsec=0.0,
                        dsec=-4, xsec = 0.0, ysec = 0.0, wave0=1.65e-6,)
```
with:
`fstar` being the stellar-to-total flux ratio at `wave0`
`dstar` being the spectral index of the secondary (if the star is assumed to be Rayleigh-Jeans then lambda^-4^ and `dstar` should be set to -4)
`denv` being the spectral index of the environment 
`UDstar` uniform disk diameter of the primary (in mas)
`fsec` is the secondary-to-total flux ratio
`dsec` is the secondary star spectral index
`xsec` is the ra position of the secondary relative to the primary (in mas)
`ysec` is the dec position of the secondary relative to the primary (in mas)

For more information about SPARCO read [this paper (original paper)](https://ui.adsabs.harvard.edu/abs/2014A%26A...564A..80K/abstract) or [this one (application to a circumbinary disk)](https://ui.adsabs.harvard.edu/abs/2016A%26A...588L...1H/abstract).

The SPARCO parameters can be obtained either by making a grid on them with an image reconstructio algorithm like ORGANIC or using geometrical model fitting like [PMOIRED](https://github.com/amerand/PMOIRED).

3. Perform the image reconstruction

```
thegan.ImageReconstruction(datafiles, sparco, data_dir=data_dir, mu=1, ps=0.6, diagnostics=False, epochs=50, nrestar=50, name='output1', boot=False, nboot=100, )
```
with:
`datafiles` being the name of the file or files, like `*.fits` for example will select all the files ending with.fits in the `data_dir`
`data_dir` is the path to the data files
`sparco` is the sparco object defined in point 2.
`mu` is the hyperparameter giving more or less weight to the Bayesian prior term (usually 1 works well)
`ps` pixels size in the image (in mas)
`diagnostics` if True will plot image and convergence criterium for each restart
`epochs` number of optimizing steps for a givern restart
`nrestart` number of restarts. starting from a different point in the latent space.
`name` the name of the output directory that is created
`boot` if True it will perform a bootstrapping loop where the data will be altered by randomnly drawing new datasets from existant measurements.
`nboot` number of new datasets to be drawn.

4. Perform a grid on image recosntruction parameters.

ORGANIC is optimized for easy parameter exploration throught grid.
It is simple: just make a list of values of any given parameter in `thegan.ImageReconstruction` or `sparco`.
It will automatically make image reconstructions corresponding to each parameter combination.
It will create folders for each combination of parameters with the value of the parameters in the name of the folder.


## Training the neural network (to be fully developped)

To train a neural network you need to generate many (~1000) images. Then you can ues them to train the neural network. You need to put them in a fits format as cube of images the third dimenstion being the number of images.
Then you can train your neural network such as indicated in the files `TrainGanExample.py`

1. Initialise the GAN

First you need to initialise an empty GAN:
```
mygan = org.GAN()
```

2. Load your images

The images should be **128 by 128 pixels** (unless you change the shape of the neural network internally...).
ORGANIC will take care of augmenting your data with random rotations, shifts and rescaling. 
You can load a cube of images using this command:
```
imgs = org.inputImages(dir, file, width_shift_range=0, height_shift_range=0,zoom_range=[0.8,1.3],)
```
where:
`dir` is the directory with the image cube
`file` is the name of the fits file with the image cube
`width_shift_range` will shift your image in the horizontal axis by this number of pixels
`height_shift_range` will shift your image in the vertical axis by this number of pixels
`zoom_range=[1.8,2.3]` will apply a random zoom factor in this ranges (e.g., smaller than 1 makes your image smaller)

3. Train your GAN



## Pre-trained GANs
- **theGANextended2** MCMax model circumstellar disks with extended component /data/leuven/334/vsc33405/summerjobTests/GANspirals/saved_models
- **GANspirals** geometrical pinwheel nebula models /data/leuven/334/vsc33405/summerjobTests/GANspirals/saved_models