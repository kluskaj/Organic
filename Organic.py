# Organic FunctionLibrary
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
import os
from astropy.io import fits
from PIL import Image


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

    def __init__(self, gen='', dis='', imagesize=128, noiselength=100, Adam_lr=0.0002, Adam_beta_1=0.5):
        if gen != '' and dis != '':
            self.dispath = dis
            self.genpath = gen
            self.read()
        else:
            self.npix = imagesize
            self.noiselength = noiselength
            self.Adam_lr = Adam_lr
            self.Adam_beta_1 = Adam_beta_1
            self.generator = self.create_generator()
            self.discriminator = self.create_discriminator()

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

    def GANtraining(
        self,
        images,
        saveDir='./saved_models/',
        nepochs = 2000,
        nbatch = 50,
        OverTrainDiscr = 1,
        PlotEpochs = 25,
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
        # this is the GAN
        generator = self.gen
        discriminator = self.dis
        # these are the images
        X_train = images.images

        batch_count = int(np.ceil(X_train.shape[0] / batch_size))
        # Creating GAN
        gan = create_gan(discriminator, generator,NoiseLength,optim)
        # arrays for plotting the loss evolution
        epochArray = np.linspace(1,epochs,epochs,endpoint=True)
        length = len(epochArray)
        discrFakeLoss = np.zeros(length)
        discrRealLoss = np.zeros(length)
        discrFakeAccuracy = np.zeros(length)
        discrRealAccuracy = np.zeros(length)
        genLoss = np.zeros(length)
        genAccuracy = np.zeros(length)
        bepochArray = np.linspace(1,int(epochs*batch_count),int(epochs*batch_count),endpoint=True)
        length = len(bepochArray)
        bdiscrFakeLoss = np.zeros(length)
        bdiscrRealLoss = np.zeros(length)
        bdiscrFakeAccuracy = np.zeros(length)
        bdiscrRealAccuracy = np.zeros(length)
        bgenLoss = np.zeros(length)
        bgenAccuracy = np.zeros(length)
        y_real = 1
        batches = datagen.flow(X_train,y=None, batch_size = batch_size)
        if Use1sidedLabelSmooth:
            y_real = 0.9
        y_false= np.zeros(batch_size)
        y_true = np.ones(batch_size)*y_real
        b = 0
        for e in range(1, epochs+1 ):
            for _ in range(batch_count): #batch_size in version from jacques
                #generate  random noise as an input  to  initialize the  generator
                noise= np.random.normal(0,1, [batch_size, NoiseLength])
                # Generate fake MNIST images from noised input
                generated_images = generator.predict(noise)
                # train the discriminator more than the generator if requested
                for i in range(OverTrainDiscr):
                    # Get a random set of  real images
                    image_batch = batches.next()
                    #plt.figure()
                    #img = centerPhotocenter(tf.expand_dims(image_batch[1], axis=0), 128)
                    #mapable=plt.imshow((np.squeeze(img)+1)/2,interpolation=None,cmap='hot',vmin = 0,vmax = 1)
                    #plt.colorbar(mapable)
                    #plt.tight_layout()
                    #plt.savefig('example loadedImage %d.png'%e)
                    #plt.close()
                    #quit()
                    # if the batch created by the generator is too small, resample
                    if image_batch.shape[0] != batch_size:
                        batches = datagen.flow(X_train,y=None, batch_size = batch_size)
                        image_batch = batches.next()
                    image_batch = image_batch.reshape(batch_size, image_size,image_size,1)
                    #Construct different batches of  real and fake data
                    X = np.concatenate([image_batch, generated_images])
                    # Labels for generated and real data
                    y_pred = np.concatenate([y_true,y_false])
                    #Pre train discriminator on  fake and real data  before starting the gan.
                    discriminator.trainable=True
                    discriminator.train_on_batch(X, y_pred)
                discrimRealEval = discriminator.evaluate(image_batch, y_pred[:batch_size],verbose=0)
                discrimFakeEval = discriminator.evaluate(generated_images, y_pred[batch_size:],verbose=0)
                #evaluations for the cost evolution of the discriminator
                discrFakeLoss[e-1] += discrimFakeEval[0]/batch_count
                discrRealLoss[e-1] += discrimRealEval[0]/batch_count
                discrFakeAccuracy[e-1] += discrimFakeEval[1]/batch_count

                bdiscrFakeLoss[b] = discrimFakeEval[0]
                bdiscrRealLoss[b] = discrimRealEval[0]
                bdiscrFakeAccuracy[b] = discrimFakeEval[1]

                #Tricking the noised input of the Generator as real data
                noise= np.random.normal(0,1, [batch_size, NoiseLength])
                y_gen = np.ones(batch_size)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                discriminator.trainable = False

                # training  the GAN by alternating the training of the Discriminator
                gan.train_on_batch(noise, y_gen)
                #evaluation of generator
                genEval = gan.evaluate(noise,y_gen,verbose =0)
                genLoss[e-1] += genEval[0]/batch_count
                genAccuracy[e-1] += genEval[1]/batch_count
                bgenLoss[b] += genEval[0]
                bgenAccuracy[b] += genEval[1]
                b = b+1
            # plot examples of generated images
            if e == 1 or e % PlotEpochs == 0:
                plot_generated_images(e, generator,NoiseLength,image_size)
            if e in saveEpochs:
                saveModel(saveDir,str(e)+'thEpoch.h5',gan,generator,discriminator)
                plotGanEvolution(epochArray,discrFakeLoss,discrRealLoss,genLoss,discrFakeAccuracy,genAccuracy)
        saveModel(saveDir,'finalModel.h5',gan,generator,discriminator)
        plotGanEvolution(epochArray,discrFakeLoss,discrRealLoss,genLoss,discrFakeAccuracy,genAccuracy)
        plotGanEvolution(bepochArray,bdiscrFakeLoss,bdiscrRealLoss,bgenLoss,bdiscrFakeAccuracy,bgenAccuracy,plus = 'NonAveraged')






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
        if self.loadFromCube == True:
            X_train = self.load_data_fromCube()
        else:
            X_train = self.load_data()
        inform('Input image class initialised')

    def load_data_fromCube():
        """
        load_data_fromCube

        parameters:
            dir: a directory where the training images can be found (must contain * to expand and find multiple images)
            imagesize: the size to which the obtained images will be rescaled (imagesize*imagesize pixels)
        returns:
            images: a numpy array containing the image information and dimensions (number of images *imagesize*imagesize*1 )
            TO TEST DOING THINGS TWICE????
        """
        cube = fits.getdata(self.dir, ext=0)
        img = Image.fromarray(cube[0])
        img = img.resize((self.npix,self.npix),Image.BILINEAR )
        #img=img.transpose(Image.FLIP_LEFT_RIGHT)
        images= np.array([np.array(img)[:, :, np.newaxis]])
        images = images/np.max(images)
        #images=(images-np.min(images))/(np.max(images)-np.min(images))
        for i in range(1,len(cube)):
            image = cube[i]
            img = Image.fromarray(image)
            img = img.resize((self.npix,self.npix),Image.BILINEAR )
            #img=img.transpose(Image.FLIP_LEFT_RIGHT)
            image=np.array([np.array(img)[:, :, np.newaxis]])
            #image=(image-np.min(image))/(np.max(image)-np.min(image))
            image = image/np.max(image)
            images = np.concatenate([images, image]) #add the rescaled image to the array
        # normalize to [-1,+1]
        images = (images-0.5)*2
        return images





if "__main__" == __name__:
    test = GAN()
    dir = 'caca/'
    file = 'pipi'
    imgs = inputImages(dir, file)
