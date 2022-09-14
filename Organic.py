# Organic FunctionLibrary
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import os
from astropy.io import fits
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import ReadOIFITS as oi
import tensorflow.keras.backend as K
mpl.use('Agg')# %matplotlib inline


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Some fancy message writing...
def header(msg):
    print(bcolors.HEADER + msg + bcolors.ENDC)


def bold(msg):
    print(bcolors.BOLD + msg + bcolors.ENDC)


def underline(msg):
    print(bcolors.UNDERLINE + msg + bcolors.ENDC)


def inform(msg):
    print(bcolors.OKBLUE + msg + bcolors.ENDC)


def inform2(msg):
    print(bcolors.OKGREEN + msg + bcolors.ENDC)


def warn(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)


def fail(msg):
    print(bcolors.FAIL + msg + bcolors.ENDC)


def log(msg, dir):
    f = open(dir+"log.txt", "a")
    f.write(msg+"\n")
    f.close()

def CrossEntropy(y_true,y_pred):
    return K.binary_crossentropy(y_true, y_pred, from_logits=False)

class GAN:
    """
    The GAN class to train and use it
    The GAN is made from a generator and a discriminator
    """

    def __init__(self, gen='', dis='', npix=128, train_disc=False, noiselength=100, Adam_lr=0.0002, Adam_beta_1=0.5, resetOpt=False):
        self.resetOpt = resetOpt
        self.Adam_lr = Adam_lr
        self.Adam_beta_1 = Adam_beta_1
        self.opt = self.getOptimizer(self.Adam_lr, self.Adam_beta_1)
        self.train_disc = train_disc
        self.noiselength = noiselength
        if gen != '' and dis != '':
            self.dispath = dis
            self.genpath = gen
            self.read()
        else:
            self.npix = npix
            self.gen = self.create_generator()
            self.dis = self.create_discriminator()

        self.gan = self.create_gan(train_disc = self.train_disc)


    @staticmethod
    def getOptimizer(lr, beta1, beta2=0.999, epsilon = 1e-7):
        return Adam(learning_rate = lr, beta_1 = beta1, beta_2 = beta2, epsilon = epsilon)

    def read(self):
        '''
        Loading the dictionnary from the generator and discriminator pathes
        '''
        inform(f'Loading the generator from {self.genpath}')
        gen = load_model(self.genpath)
        gen.summary()
        inform(f'Loading the discriminator from {self.dispath}')
        dis = load_model(self.dispath)
        dis.summary()
        self.gen = gen
        self.dis = dis
        self.npix = gen.output.shape[1]


    def create_generator(self, ReLU=0.25):
        inform('Creating the generator')
        npix = self.npix
        generator = keras.Sequential(
        [
            layers.Dense(int((npix/8)*(npix/8)*256), use_bias=False, kernel_initializer='he_normal',input_shape=(self.noiselength,)),
            layers.LeakyReLU(alpha = ReLU),
            layers.Reshape((int(npix/8),int(npix /8), 256)),
            layers.Conv2DTranspose(npix, (4,4), strides=(2, 2), padding='same', use_bias=False,kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha = ReLU),
            layers.Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', use_bias=False,kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha = ReLU),
            layers.Conv2DTranspose(32, (4,4), strides=(2, 2), padding='same', use_bias=False,kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha = ReLU),
            layers.Conv2D(1, (2,2), strides=(1, 1), padding='same', use_bias=False, activation='tanh',kernel_initializer='glorot_normal')
        ],
        name = 'generator'
        )
        generator.summary()

        generator.compile(loss='binary_crossentropy', optimizer=self.getOptimizer(self.Adam_lr, self.Adam_beta_1))
        return generator


    def create_discriminator(self, ReLU=0.25, dropout=0.5):
        inform('Creating the discriminator')
        npix = self.npix
        discriminator = keras.Sequential(
        [
            layers.Conv2D(npix/4, (3,3), strides=(2, 2), padding='same', input_shape=[npix, npix, 1], kernel_initializer='he_normal'),
            layers.LeakyReLU(ReLU),
            layers.SpatialDropout2D(dropout),
            layers.Conv2D(npix/2, (3,3), strides=(2, 2), padding='same',kernel_initializer='he_normal'),
            layers.LeakyReLU(ReLU),
            layers.SpatialDropout2D(dropout),
            layers.Conv2D(npix, (3,3), strides=(2, 2), padding='same',kernel_initializer='he_normal'),
            layers.LeakyReLU(ReLU),
            layers.SpatialDropout2D(dropout),
            layers.Flatten(),
            layers.Dense(1,activation='sigmoid', use_bias=False,kernel_initializer='glorot_normal')
        ],
        name = 'discriminator'
        )
        discriminator.summary()
        discriminator.compile(loss='binary_crossentropy', optimizer=self.getOptimizer(self.Adam_lr, self.Adam_beta_1), metrics=["accuracy"])

        return discriminator


    def create_gan(self, train_disc = False):
        """
        create_gan

        parameters:
            discriminator: a keras sequential model (network) with outputsize = 1 and inputsize = imagesize*imagesize*1
            generator: a keras sequential model (network) with outputsize = imagesize*imagesize*1 and inputsize = NoiseLength
        returns:
            gan: a compiled keras model where the generator is followed by the discriminator and the discriminator is not trainable

        """
        self.dis.trainable = train_disc
        gan_input = layers.Input(shape = (self.noiselength,))
        x = self.gen(gan_input)
        gan_output= self.dis(x)
        gan= Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer = self.opt, metrics=["accuracy"])
        return gan

    def train(
        self,
        images,
        save_dir='./saved_models/',
        nepochs = 2000,
        nbatch = 50,
        OverTrainDiscr = 1,
        plotEpochs = 25,
        Use1sidedLabelSmooth = False,
        saveEpochs = []
        ):
        '''
        epochs: the number of iterations over a set of image with its size eaqual to the training dataset
        batch_size: mini-batch size used for training the gan
        saveDir: directory where the trained networks will be stored
        PlotEpochs: the epoch interval at which examples of generated images will be created
        Use1sidedLabelSmooth: whether ornot to use onsided label smoothing (best true when using binary binary_crossentropy, best false when using MSE (LSGAN))
    effect:
        Trains the GAN
        Saves plots with examples of generated images and the loss evolution of the gans components
        Saves the trained netorks at the requested and final epochs of training
        '''
        self.save_dir = save_dir

        self.nbatch = nbatch
        self.nepochs = nepochs
        # this is the GAN
        generator = self.gen
        discriminator = self.dis
        gan = self.gan
        # these are the images
        X_train = images.images
        datagen = images.dataGen
        batch_count = int(np.ceil(X_train.shape[0] / nbatch)) # defining the batches

        # define the labels
        y_real = 1
        batches = datagen.flow(X_train,y=None, batch_size = nbatch)
        if Use1sidedLabelSmooth:
            y_real = 0.9
        y_false= np.zeros(nbatch)
        y_true = np.ones(nbatch)*y_real

        disFakeLoss, disRealLoss, disFakeAccuracy, genAccuracy, genLoss = [], [], [], [], []

        inform('Starting GAN training')
        for epoch in np.arange(nepochs):
            inform2(f'Epoch {epoch+1} of {nepochs}')
            disFakeLossEp, disRealLossEp, disFakeAccuracyEp = 0, 0 ,0
            genLossEp, genAccuracyEp = 0, 0
            for _ in range(nbatch): #batch_size in version from jacques
                #generate  random noise as an input  to  initialize the  generator
                noise = np.random.normal(0,1, [nbatch, self.noiselength])
                # Generate ORGANIC images from noised input
                generated_images = generator.predict(noise)
                # train the discriminator more than the generator if requested
                for i in range(OverTrainDiscr):
                    # Get a random set of  real images
                    image_batch = batches.next()
                    # if the batch created by the generator is too small, resample
                    if image_batch.shape[0] != nbatch:
                        batches = datagen.flow(X_train, y=None, batch_size = nbatch)
                        image_batch = batches.next()
                    image_batch = image_batch.reshape(nbatch, self.npix, self.npix, 1)
                    #Construct different batches of  real and fake data
                    X = np.concatenate([image_batch, generated_images])
                    # Labels for generated and real data
                    y_pred = np.concatenate([y_true,y_false])
                    #Pre train discriminator on  fake and real data  before starting the gan.
                    discriminator.trainable=True
                    discriminator.train_on_batch(X, y_pred)
                disRealEval = discriminator.evaluate(image_batch, y_pred[:nbatch],verbose=0)
                disFakeEval = discriminator.evaluate(generated_images, y_pred[nbatch:],verbose=0)
                #evaluations for the cost evolution of the discriminator
                disFakeLossEp += disFakeEval[0]/batch_count
                disRealLossEp += disRealEval[0]/batch_count
                disFakeAccuracyEp += disFakeEval[1]/batch_count

                #Tricking the noised input of the Generator as real data
                noise= np.random.normal(0,1, [nbatch, self.noiselength])
                y_gen = np.ones(nbatch)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                discriminator.trainable = False

                # training  the GAN by alternating the training of the Discriminator
                gan.train_on_batch(noise, y_gen)
                #evaluation of generator
                genEval = gan.evaluate(noise, y_gen, verbose =0)
                genLossEp += genEval[0]/batch_count
                genAccuracyEp += genEval[1]/batch_count

            # Saving all the metrics per epoch
            genAccuracy.append(genAccuracyEp)
            genLoss.append(genLossEp)
            disFakeLoss.append(disFakeLossEp)
            disRealLoss.append(disRealLossEp)
            disFakeAccuracy.append(disFakeAccuracyEp)
            #saveing current state of networks
            self.gan = gan
            self.dis = discriminator
            self.gen = generator

            # plot examples of generated images
            if epoch == 1 or epoch % plotEpochs == 0:
                self.plot_generated_images(epoch)
            if epoch in saveEpochs:
                self.saveModel(str(e)+'thEpoch.h5')

        self.saveModel('finalModel.h5')
        self.plotGanEvolution(disFakeLoss,disRealLoss,genLoss,disFakeAccuracy,genAccuracy)

        inform(f'Training succesfully finished.\nResults saved at {self.save_dir}')


    def plotGanEvolution(self, disFakeLoss, disRealLoss,
    genLoss, disFakeAccuracy, genAccuracy):
        """
        plotGanEvolution


        parameters:
            epoch: array containing the epochs to be plotted on the x-newaxis
            discrFakeLoss: cost values for the discriminators response to fake(generated) image data
            discrRealLoss: cost values for the discriminators response to real(model) image data
            genLoss: cost values for the generator
            discrFakeAccuracy: accuracy values for the discriminators response to fake(generated) image data
            discrRealAccuracy: accuracy values for the discriminators response to real(model) image data
            genAccuracy: accuracy values for the generator

        effect:
            Plots the cost and accuracy terms as a function of epoch and stores the resulting plots

        """
        dir = self.save_dir

        fig, ax = plt.subplots()

        color = iter(plt.cm.rainbow(np.linspace(0,1,5)))

        c = next(color)
        plt.plot(disFakeLoss,label = 'discriminator fake data loss', c = c)
        c=next(color)
        plt.plot(disRealLoss,label = 'discriminator real data loss', c = c)
        c=next(color)
        plt.plot(genLoss, label = 'generator loss', c = c)
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(dir+'LossEvolution.pdf')
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(disFakeAccuracy,label = 'discriminator data accuracy',c = c)
        c=next(color)
        plt.plot(genAccuracy,label = 'generator data accuracy',  c = c)
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig(dir+'AccuracyEvolution.pdf')
        plt.close()


    def saveModel(self, model_name):
        """
        saveModel

        parameters:
            Modelname: name to be used for storing the networks of this run
        effect:
            saves the keras models (neural networks) in their curren state

        """
        #test if the path exists, if not, creates it
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        model_path_GAN = os.path.join(self.save_dir, 'GANfull'+ model_name)
        self.gan.save(model_path_GAN)
        plot_model(self.gan, to_file = 'full.png', show_shapes=True)

        model_path_generator = os.path.join(self.save_dir, 'generator' + model_name)
        self.gen.save(model_path_generator)
        plot_model(self.gen, to_file='generator.png',show_shapes=True)

        model_path_discriminator = os.path.join(self.save_dir, 'discriminator' + model_name)
        self.dis.save(model_path_discriminator)
        plot_model(self.dis, to_file='discriminator.png',show_shapes=True)
        print(f'Saved trained model at {model_path_GAN}')


    def plot_generated_images(self, epoch, examples=36, dim=(6, 6), figsize=(15, 9)):
        """
        plot_generated_images

        parameters:
            epoch: the epoch at which th plots are made, used for naming the image
            generator: the generator neural network during the given epochs
            examples: the number of examples to be displayed in the plot
        effect:
            saves images contain a number of random example images created by the generator

        """
        generator = self.gen
        noise = np.random.normal(loc=0, scale=1, size=[examples,self.noiselength])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(examples, self.npix, self.npix)
        fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize, sharex=True, sharey=True)
        i = -1
        for axv in axs:
            for ax in axv:
                i += 1
                ax.imshow(generated_images[i], origin = 'lower', interpolation=None, cmap='hot', vmin=-1, vmax=1)
                #ax.invert_xaxis()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'cgan_generated_image_ep{epoch}.png')
        plt.close()


    def ImageReconstruction(self, data_files, sparco, data_dir='./', mu=1, epochs=200, nrestart=200, boot=False, nboot=100,
                            ps=0.6, shiftPhotoCenter = True, UseRoll=True,
                            interp = 'BILINEAR', useLowCPapprox = False, grid = False
                            ):
        self.mu = mu
        self.epochs = epochs
        self.ps = ps
        self.nboot = nboot
        self.boot = boot
        self.nrestart = nrestart
        self.useLowCPapprox = useLowCPapprox
        self.sparco = sparco
        self.shiftPhotoCenter = shiftPhotoCenter
        self.use_roll = UseRoll
        self.interp = interp
        self.grid = grid
        self.data = Data(data_dir, data_files)

        # Creating dictionary with image recsontruction parameters
        self.params = {
            'mu' : self.mu,
            'ps' : self.ps,
            'epochs' : self.epochs,
            'nrestart' : self.nrestart,
            'useLowCPapprox' : self.useLowCPapprox,
            'fstar' : sparco.fstar,
            'dstar' : sparco.dstar,
            'denv' : sparco.denv,
            'UDstar' : sparco.UDstar,
            'fsec' : sparco.fsec,
            'dsec' : sparco.dsec,
            'xsec' : sparco.xsec,
            'ysec' : sparco.ysec,
        }

        # checking if grid of reconstructions needed
        ngrid, niters = 0, 1
        gridpars, gridvals = [], []
        for x, v in self.__dict__.items():
            if isinstance(v, list):
                self.grid = True
                ngrid += 1
                gridpars.append(x)
                gridvals.append(v)
                niters *= len(v)
        # same for SPARCO
        for x, v in sparco.__dict__.items():
            if isinstance(v, list):
                self.grid = True
                ngrid += 1
                gridpars.append(x)
                gridvals.append(v)
                niters *= len(v)


                #        print(self.__dict__[gridpars[0]])
        # Run a single image recosntruction or a grid
        if self.grid:
            self.niters = niters
            self.ngrid = ngrid
            self.gridpars = gridpars
            self.gridvals = gridvals
            print(gridvals)
            self.iterable = itertools.product(*self.gridvals)
            inform(f'Making an image reconstruction grid ({niters} reconstructions) on {ngrid} parameter(s): {gridpars}')
            self.runGrid()
        else:
            inform('Running a single image reconstruction')
            self.dir = 'ImageRec'
            self.SingleImgRec()

    def runGrid(self):
        for i, k in zip(self.iterable, np.arange(self.niters)):
            state = ''
            dir = 'ImageRec'
            for pars, p in zip(self.gridpars, np.arange(self.ngrid)):
                self.params[f'{pars}'] = i[p]
                state += f' {pars}={i[p]}'
                dir += f'_{pars}={i[p]}'
            inform2(f'Image reconstruction with{state}')
            self.dir = dir
            self.ImgRec()

    def SingleImgRec(self):
        print('prout')

    def ImgRec(self ):
        params = self.params
        #Create data loss
        self.data_loss = self.dataloss
        # Create the GAN
        # GAN = self.createGAN() NO NEED

        # Update GAN
        GeneratorCopy = tf.keras.models.clone_model(self.gen)
        GeneratorCopy.set_weights(self.gen.get_weights())
        outdir = os.path.join(os.getcwd(),self.dir)
        for r in range(params['nrestart']):
            #GeneratorCopy.set_weights(Generator.get_weights())
            if self.gan != None:
                self.gan.get_layer(index=1).set_weights(self.gen.get_weights())
                if self.resetOpt == True:
                    opt = 'optimizers.'+self.opt._name
                    opt = eval(opt)
                    opt = opt.from_config(self.opt.get_config())
                    self.gan.compile(loss=[CrossEntropy ,dataLikelihood],optimizer= opt,loss_weights=[params['mu'],1])

            # generating the noise vector for this restart
            noisevector = np.array([np.random.normal(0, 1, 100)])
            y_gen = [np.ones(1),np.ones(1)]

            # the loop on epochs with one noise vector
            discloss = []
            chi2 = []
            for e in range(1, params['epochs']+1 ):
                print(e)
                #generate  random noise as an input  to  initialize the  generator
                hist= self.gan.train_on_batch(noisevector, y_gen)
                discloss.append( hyperParam*hist[1] )
                chi2.append( hist[2] )
                #plot the image at the specified epochs
                if e in plotAtEpoch:
                    plot_generated_images2(e, self.fullNet.get_layer(index=1), noisevector, image_Size, pixelSize, saveDir)
                    plt.close()

    def createGAN(self):
        # Loading networks
        dis = self.dis
        gen = self.gen
        dis.trainable = False
        noise_input = layers.Input(shape = (100,))
        x = gen(noise_input)
        gan_output= dis(x)
        gan = Model(inputs=noise_input, outputs=[gan_output,x])
        losses = [CrossEntropy , self.data_loss]
        mu = self.params['mu']
        gan.compile(loss=losses, optimizer= opt, loss_weights=[mu,1])


    def dataloss(self):
        data = self.data
        if self.boot:
            V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_bootstrap()
        else:
            V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_data()

        params = self.params
        fstar = params['fstar']
        fsec = params['fsec']
        UD = params['UDstar']
        denv = params['denv']
        dsec = params['dsec']
        xsec = params['xsec']
        ysec = params['ysec']
        ps = params['ps']
        wave0 = params['wave0']
        useLowCPapprox = params['useLowCPapprox']
        nV2 = len(V2)
        nCP = len(CP)
        npix = 128 ## to CHANGE TO GRAB FROM THE GAN
        spacialFreqPerPixel = (3600/(0.001*npix*ps))*(180/np.pi)

        def offcenterPointFT(x, y, u, v):
            u = tf.constant(u,dtype = tf.complex128)
            v = tf.constant(v,dtype = tf.complex128)
            return tf.math.exp(-2*np.pi*1j*(x*u+y*v))

        #preforms a binlinear interpolation on grid at continious pixel coordinates ufunc,vfunc
        def bilinearInterp(grid, ufunc, vfunc):
            ubelow = np.floor(ufunc).astype(int)
            vbelow = np.floor(vfunc).astype(int)
            uabove = ubelow +1
            vabove = vbelow +1
            coords = tf.constant([[[0,ubelow[i],vbelow[i]] for i in range(len(ufunc))]])
            interpValues =  tf.gather_nd(grid,coords)*(uabove-ufunc)*(vabove-vfunc)
            coords1 =tf.constant([[[0,uabove[i],vabove[i]] for i in range(len(ufunc))]])
            interpValues += tf.gather_nd(grid,coords1)*(ufunc-ubelow)*(vfunc-vbelow)
            coords2 = tf.constant([[[0,uabove[i],vbelow[i]] for i in range(len(ufunc))]])
            interpValues +=  tf.gather_nd(grid,coords2)*(ufunc-ubelow)*(vabove-vfunc)
            coords3 = tf.constant([[[0,ubelow[i],vabove[i]] for i in range(len(ufunc))]])
            interpValues += tf.gather_nd(grid,coords3)*(uabove-ufunc)*(vfunc-vbelow)
            return interpValues

        #plots a comperison between observations and observables of the reconstruction,aswell as the uv coverage
        def plotObservablesComparison(V2generated, V2observed, V2err, CPgenerated, CPobserved, CPerr):

            #v2 with residual comparison, no colors indicating wavelength
            plt.figure(figsize=(3.5, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[6, 3])
            ax1=plt.subplot(gs[0]) # sharex=True)
            absB = (np.sqrt(u**2+v**2)/(10**6))
            plt.scatter(absB, V2generated[0].numpy(), marker='.',s=40, label = 'image',c = 'b',alpha=0.4,edgecolors ='k',linewidth=0.15)
            plt.scatter(absB,V2observed,marker='*',s=40,label = 'observed',c = 'r',alpha=0.4,edgecolors ='k',linewidth=0.15)
            plt.errorbar(absB,V2observed,V2err,elinewidth=0.2,ls='none',c ='r')
            plt.ylim(0,1)
            plt.ylabel(r'$V^2$')
            plt.legend()

            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.subplot(gs[1], sharex=ax1)
            plt.scatter(absB,((V2observed-V2generated[0].numpy())/(V2err)),s=30,marker='.',c = 'b',alpha=0.6,edgecolors ='k',linewidth=0.1)
            plt.ylabel(r'residuals',fontsize =12)
            plt.xlabel(r'$\mid B\mid (M\lambda)$')
            plt.tight_layout()
            if bootstrapDir == None:
                plt.savefig(os.path.join(os.getcwd(),'V2comparisonNoColor.png'))
            else:
                plt.savefig(os.path.join(os.getcwd(),bootstrapDir+ '/V2comparisonNoColor.png'))

            #plots the uv coverage
            plt.figure()
            plt.scatter(u/(10**6),v/(10**6),marker='.',c=np.real(wavelV2),cmap ='rainbow',alpha=0.9,edgecolors ='k',linewidth=0.1)
            plt.scatter(-u/(10**6),-v/(10**6),marker='.',c=np.real(wavelV2),cmap ='rainbow',alpha=0.9,edgecolors ='k',linewidth=0.1)
            plt.xlabel(r'$ u (M\lambda)$')
            plt.ylabel(r'$ v (M\lambda)$')
            plt.gca().set_aspect('equal', adjustable='box')
            if bootstrapDir == None:
                plt.savefig(os.path.join(os.getcwd(), 'uvCoverage.png'))
            else:
                plt.savefig(os.path.join(os.getcwd(),bootstrapDir+ '/uvCoverage.png'))


            #cp with residual comparison without color indicating wavelength
            plt.figure(figsize=(3.5, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[6, 3])
            ax1=plt.subplot(gs[0]) # sharex=True)
            maxB = (np.maximum(np.maximum(np.sqrt(u1**2 +v1**2),np.sqrt(u2**2 +v2**2)),np.sqrt(u3**2 +v3**2))/(10**6))
            plt.scatter(maxB,CPgenerated[0].numpy(),s=30,marker='.',c = 'b',label = 'image',cmap ='rainbow',alpha=0.4,edgecolors =colors.to_rgba('k', 0.1), linewidth=0.3)
            plt.scatter(maxB,CPobserved,s=30,marker='*',label = 'observed',c = 'r',alpha=0.4,edgecolors =colors.to_rgba('k', 0.1), linewidth=0.3)
            plt.errorbar(maxB,CPobserved,CPerr,ls='none',elinewidth=0.2,c ='r')
            plt.legend()
            plt.ylabel(r'closure phase(radian)',fontsize =12)

            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.subplot(gs[1], sharex=ax1)
            plt.scatter(maxB,((CPobserved-CPgenerated[0].numpy())/(CPerr)),s=30,marker='.',c = 'b',cmap ='rainbow',alpha=0.6,edgecolors =colors.to_rgba('k', 0.1), linewidth=0.3)
            #color = colors.to_rgba(np.real(wavelCP.numpy())[], alpha=None) #color = clb.to_rgba(waveV2[])
            #c[0].set_color(color)

            plt.xlabel(r'max($\mid B\mid)(M\lambda)$',fontsize =12)
            plt.ylabel(r'residuals',fontsize =12)
            plt.tight_layout()
            if bootstrapDir == None:
                plt.savefig(os.path.join(os.getcwd(),'cpComparisonNoColor.png'))
            else:
                plt.savefig(os.path.join(os.getcwd(),bootstrapDir+ '/cpComparisonNoColor.png'))
            plt.close()



        def compTotalCompVis(ftImages, ufunc, vfunc, wavelfunc):
            #to compute the radial coordinate in the uv plane so compute the ft of the primary, which is a uniform disk, so the ft is....
            radii = np.pi*UD*np.sqrt(ufunc**2 + vfunc**2)
            ftPrimary = tf.constant(2*sp.jv(1,radii)/(radii),dtype = tf.complex128)
            #see extra function
            ftSecondary = offcenterPointFT( xsec, ysec, ufunc, vfunc)
            # get the total visibility amplitudes
            VcomplDisk = bilinearInterp(ftImages,(vfunc/spacialFreqPerPixel)+int(ImageSize/2),(ufunc/spacialFreqPerPixel)+int(ImageSize/2))
            #equation 4 in sparco paper:
            VcomplTotal = fstar * ftPrimary* K.pow(wavelfunc/wave0, dstar)
            VcomplTotal += fsec * ftSecondary* K.pow(wavelfunc/wave0, dsec)
            VcomplTotal += (1-fstar-fsec) *VcomplDisk* K.pow(wavelfunc/wave0, denv)
            VcomplTotal = VcomplTotal/((fstar*K.pow(wavelfunc/wave0, dstar))+(fsec * K.pow(wavelfunc/wave0,dsec)) +((1-fstar-fsec)*K.pow(wavelfunc/wave0,denv)))
            return VcomplTotal


        def data_loss(y_true, y_pred):
            y_pred = tf.squeeze(y_pred,axis = 3)
            y_pred = (y_pred+1)/2
            y_pred = tf.cast((y_pred),tf.complex128)
            y_pred = tf.signal.ifftshift(y_pred,axes = (1,2))
            ftImages = tf.signal.fft2d(y_pred)#is complex!!
            ftImages = tf.signal.fftshift(ftImages,axes=(1,2))
            coordsMax = [[[[0,int(npix/2),int(npix/2)]]]]
            ftImages =ftImages/tf.cast(tf.math.abs(tf.gather_nd(ftImages,coordsMax)),tf.complex128)
            VcomplForV2 = compTotalCompVis(ftImages, u, v, waveV2)
            V2image = tf.math.abs(VcomplForV2)**2# computes squared vis for the generated images

            V2Chi2Terms = K.pow(V2 - V2image,2)/(K.pow(V2err,2)*nV2)# individual terms of chi**2 for V**2
            #V2Chi2Terms = V2Chi2Terms
            V2loss = K.sum(V2Chi2Terms, axis=1) #the squared visibility contribution to the loss

            CPimage  = tf.math.angle(compTotalCompVis(ftImages, u1, v1, waveCP))
            CPimage += tf.math.angle(compTotalCompVis(ftImages, u2, v2, waveCP))
            CPimage -= tf.math.angle(compTotalCompVis(ftImages, u3, v3, waveCP))
            CPchi2Terms = 2*(1-tf.math.cos(CP-CPimage))/(K.pow(CPerr,2)*nCP)
            if useLowCPapprox:
                CPchi2Terms=K.pow(CP-CPimage, 2)/(K.pow(CPerr,2)*nCP)

            CPloss = K.sum(CPchi2Terms, axis=1)

            lossValue  = (K.mean(V2loss)*nV2 + K.mean(CPloss)*nCP)/(nV2+nCP)
            if forTraining == True:
                print(V2loss, CPloss, lossValue)
                return  tf.cast(lossValue, tf.float32)

            else:
                plotObservablesComparison(V2image, V2, V2err, CPimage, CP, CPerr)
                print('no training')
                print(V2loss,CPloss,lossValue)
                return lossValue, V2loss , CPloss

        self.data_loss = data_loss



class ImgRec:
    def __init__(self, data, gen, dis, sparco, mu=1, epochs=100, ps=0.5,
                            boot=0, nrestart = 200,
                            shiftPhotoCenter = True, UseRoll=True,
                            interp = 'BILINEAR', useLowCPapprox = False):
        self.mu = mu
        self.epochs = epochs
        self.ps = ps
        self.boot = boot
        self.nrestart = nrestart
        self.useLowCPapprox = useLowCPapprox
        self.datafiles = datafiles
        self.sparco = sparco
        self.data_dir = dataDir
        self.shiftPhotoCenter = shiftPhotoCenter
        self.use_roll = UseRoll
        self.interp = interp
        self.gen = gen
        self.dis = dis
        self.sparco = sparco
        self.dataset = data
        self.imgrec()

    def imgrec(self):

        # First data loading and selection
        data = self.dataset
        if self.boot >= 1:
            V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_bootstrap()
        else:
            V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_data()

        gan = self.createGan()

        a = self.data_loss(  u, u1, u2, u3, v, v1, v2, v3)


    def data_loss(self, y_pred, u, u1, u2, u3, v, v1, v2, v3):
        #compute the fourier transform of the images
        y_pred = tf.squeeze(y_pred, axis = 3)
        y_pred = (y_pred+1)/2
        y_pred = tf.cast( (y_pred), tf.complex128)
        y_pred = tf.signal.ifftshift(y_pred, axes = (1,2))
        ftImages = tf.signal.fft2d(y_pred)#is complex!!
        ftImages = tf.signal.fftshift(ftImages, axes=(1,2))
        coordsMax = [[[[0,int(self.npix/2),int(self.npix/2)]]]]
        ftImages =ftImages/tf.cast(tf.math.abs(tf.gather_nd(ftImages,coordsMax)),tf.complex128)
        VcomplForV2 = compTotalCompVis(ftImages,u,v,wavelV2)
        V2generated = tf.math.abs(VcomplForV2)**2# computes squared vis for the generated images

        V2Chi2Terms = K.pow(V2observed - V2generated,2)/(K.pow(V2err,2)*nV2)# individual terms of chi**2 for V**2
        #V2Chi2Terms = V2Chi2Terms
        V2loss = K.sum(V2Chi2Terms,axis=1) #the squared visibility contribution to the loss

        CPgenerated  = tf.math.angle(compTotalCompVis(ftImages,u1,v1,wavelCP))
        CPgenerated += tf.math.angle(compTotalCompVis(ftImages,u2,v2,wavelCP))
        CPgenerated -= tf.math.angle(compTotalCompVis(ftImages,u3,v3,wavelCP))
        CPchi2Terms = 2*(1-tf.math.cos(CPobserved-CPgenerated))/(K.pow(CPerr,2)*nCP)
        if useLowCPapprox == True:
            CPchi2Terms=K.pow(CPobserved-CPgenerated,2)/(K.pow(CPerr,2)*nCP)

        CPloss = K.sum(CPchi2Terms,axis=1)

        lossValue  = (K.mean(V2loss)*nV2 + K.mean(CPloss)*nCP)/(nV2+nCP)
        if forTraining == True:
            print(V2loss,CPloss,lossValue)
            return  tf.cast(lossValue,tf.float32)

        else:
            plotObservablesComparison(V2generated,V2observed,V2err,CPgenerated,CPobserved,CPerr)
            print('no training')
            print(V2loss,CPloss,lossValue)
            return lossValue, V2loss , CPloss


    @staticmethod
    def offcenterPointFT(x,y,u,v):
        u = tf.constant(u,dtype = tf.complex128)
        v = tf.constant(v,dtype = tf.complex128)
        return tf.math.exp(-2*np.pi*1j*(x*u+y*v))


    @staticmethod
    def bilinearInterp(grid,ufunc,vfunc):
    #preforms a binlinear interpolation on grid at continious pixel coordinates ufunc,vfunc
        ubelow = np.floor(ufunc).astype(int)
        vbelow = np.floor(vfunc).astype(int)
        uabove = ubelow +1
        vabove = vbelow +1
        coords = tf.constant([[[0,ubelow[i],vbelow[i]] for i in range(len(ufunc))]])
        interpValues =  tf.gather_nd(grid,coords)*(uabove-ufunc)*(vabove-vfunc)
        coords1 =tf.constant([[[0,uabove[i],vabove[i]] for i in range(len(ufunc))]])
        interpValues += tf.gather_nd(grid,coords1)*(ufunc-ubelow)*(vfunc-vbelow)
        coords2 = tf.constant([[[0,uabove[i],vbelow[i]] for i in range(len(ufunc))]])
        interpValues +=  tf.gather_nd(grid,coords2)*(ufunc-ubelow)*(vabove-vfunc)
        coords3 = tf.constant([[[0,ubelow[i],vabove[i]] for i in range(len(ufunc))]])
        interpValues += tf.gather_nd(grid,coords3)*(uabove-ufunc)*(vfunc-vbelow)
        return interpValues

    def dataloss(self):
        data = oi.read()
        # SPARCO HERE!!!!
        x = sparco.xsec *np.pi*0.001/(3600*180)
        y = sparco.ysec *np.pi*0.001/(3600*180)
        fstar = sparco.fstar # UDflux
        fsec = sparco.fsec # Pointflux
        UDstar = sparco.UDstar # UDdiameter


class Data:
    def __init__(self, dir, file):
        self.dir = dir
        self.file = file
        self.read_data()

    def read_data(self):
        data = oi.read(self.dir, self.file)
        dataObj = data.givedataJK()

        V2observed, V2err = dataObj['v2']
        nV2 = len(V2err)

        CPobserved, CPerr = dataObj['cp']
        nCP = len(CPerr)

        u, u1, u2, u3 = dataObj['u']
        v, v1, v2, v3 = dataObj['v']

        waveV2 = dataObj['wave'][0]
        waveCP = dataObj['wave'][1]

        V2 = tf.constant(V2observed)#conversion to tensor
        V2err = tf.constant(V2err)#conversion to tensor
        CP = tf.constant(CPobserved)*np.pi/180 #conversion to radian & cast to tensor
        CPerr = tf.constant(CPerr)*np.pi/180 #conversion to radian & cast to tensor
        waveV2 = tf.constant(waveV2,dtype = tf.complex128) #conversion to tensor
        waveCP = tf.constant(waveCP,dtype = tf.complex128) #conversion to tensor

        self.nV2 = nV2
        self.nCP = nCP
        self.V2 = V2
        self.V2err = V2err
        self.CP = CP
        self.CPerr = CPerr
        self.waveV2 = waveV2
        self.waveCP = waveCP
        self.u = u
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.v = v
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3


    def get_data(self):
        return self.V2, self.V2err, self.CP, self.CPerr, self.waveV2, self.waveCP,self.u, self.u1, self.u2, self.u3, self.v, self.v1, self.v2, self.v3


    def get_bootstrap(self):
        V2selection = np.random.randint(0,self.nV2,self.nV2)
        newV2, newV2err = self.V2[V2selection], self.V2err[V2selection]
        CPselection = np.random.randint(0,self.nCP,self.nCP)
        newCP, newCPerr = self.CP[CPselection], self.CPerr[CPselection]
        newu, newu1, newu2, newu3 = self.u[V2selection], self.u1[CPselection], self.u2[CPselection], self.u3[CPselection]
        newv, newv1, newv2, newv3 = self.v[V2selection], self.v1[CPselection], self.v2[CPselection], self.v3[CPselection]
        newwavelV2 = self.waveV2[V2selection]
        newwavelCP = self.waveCP[CPselection]
        return newV2, newV2err, newCP, newCPerr, newwaveV2, newwaveCP, newu, newu1, newu2, newu3, newv, newv1, newv2, newv3


class SPARCO:
    def __init__(self, wavel0=1.65e-6, fstar=0.0, dstar=-4.0, denv=0.0, UDstar=0.01, fsec=0.0,
                        dsec=-4, xsec = 0.0, ysec = 0.0):
        self.wavel0 = wavel0
        self.fstar = fstar
        self.dstar = dstar
        self.denv = denv
        self.UDstar = UDstar
        self.fsec = fsec
        self.dsec = dsec
        self.xsec = xsec
        self.ysec = ysec


class inputImages:
    '''
    a class to get format the images the way keras can augment them
    '''
    def __init__(self,
    dir, file,
    imagesize=128,
    loadfromCube = True,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0,
    shear_range=0.,  # set range for random shear
    zoom_range=[1.8,2.3],  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=-1.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0):
        self.npix = imagesize
        self.dir = os.path.join(os.path.expandvars(dir), file)
        self.loadfromCube = loadfromCube
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.data_format = data_format
        self.validation_split = validation_split

        self.load()

    def load(self):
        self.dataGen = ImageDataGenerator(
            featurewise_center = self.featurewise_center,
            samplewise_center = self.samplewise_center,
            featurewise_std_normalization = self.featurewise_std_normalization,
            samplewise_std_normalization = self.samplewise_std_normalization,
            zca_whitening = self.zca_whitening,
            zca_epsilon = self.zca_epsilon,
            rotation_range = self.rotation_range,
            width_shift_range = self.width_shift_range,
            height_shift_range = self.height_shift_range,
            shear_range = self.shear_range,
            zoom_range = self.zoom_range,
            channel_shift_range = self.channel_shift_range,
            fill_mode = self.fill_mode,
            cval = self.cval,
            horizontal_flip = self.horizontal_flip,
            vertical_flip = self.vertical_flip,
            rescale = self.rescale,
            preprocessing_function = self.preprocessing_function,
            data_format = self.data_format,
            validation_split = self.validation_split,
            )
        if self.loadfromCube == True:
            self.images = self.load_data_fromCube()
        else:
            self.images = self.load_data()
        inform(f'Input images loaded successfully with shape {self.images.shape}')

    def load_data_fromCube(self):
        """
        load_data_fromCube

        parameters:
            dir: a directory where the training images can be found (must contain * to expand and find multiple images)
            imagesize: the size to which the obtained images will be rescaled (imagesize*imagesize pixels)
        returns:
            images: a numpy array containing the image information and dimensions (number of images *imagesize*imagesize*1 )
        """
        cube = fits.getdata(self.dir, ext=0)

        images = []
        for i in np.arange(len(cube)):

            img0 = Image.fromarray(cube[i])
            img = img0.resize((self.npix,self.npix),Image.BILINEAR )
            img /= np.max(img)
            #img = img[:,:,np.newaxis]
            images.append(img)

        newcube = np.array(images)
        newcube = (newcube[:,:,:,np.newaxis]-0.5)*2

        return newcube

    def load_data(self):
        dirs = glob.glob(self.dir)
        images = []
        for i in np.arange(len(dirs)):
            image = fits.getdata(dirs[i], ext=0)
            img = Image.fromarray(image)
            img = img.resize((self.npix,self.npix),Image.BILINEAR )
            img /= np.max(img)
            # img = img[:,:,np.newaxis]
            images.append(img)
        newcube = np.array(images)
        newcube = (newcube[:,:,:,np.newaxis]-0.5)*2

        return newcube




if "__main__" == __name__:
    test = GAN()
    dir = 'caca/'
    file = 'pipi'
    imgs = inputImages(dir, file)
