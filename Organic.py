# Organic FunctionLibrary
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
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



class GAN:
    """
    The GAN class to train and use it
    The GAN is made from a generator and a discriminator
    """

    def __init__(self, gen='', dis='', imagesize=128, train_disc=False, noiselength=100, Adam_lr=0.0002, Adam_beta_1=0.5):
        self.Adam_lr = Adam_lr
        self.Adam_beta_1 = Adam_beta_1
        self.opt = self.getOptimizer(self.Adam_lr, self.Adam_beta_1)
        self.train_disc = train_disc
        if gen != '' and dis != '':
            self.dispath = dis
            self.genpath = gen
            self.read()
        else:
            self.npix = imagesize
            self.noiselength = noiselength
            self.gen = self.create_generator()
            self.dis = self.create_discriminator()

        self.gan = self.create_gan(train_disc = self.train_disc)


    @staticmethod
    def getOptimizer(lr, beta):
        return Adam(learning_rate = lr, beta_1 = beta)

    def read(self):
        inform(f'Loading the generator from {self.genpath}')
        gen = load_model(self.genpath)
        inform(f'Loading the discriminator from {self.dispath}')
        dis = load_model(self.dispath)
        self.gen = gen
        self.dis = dis


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
