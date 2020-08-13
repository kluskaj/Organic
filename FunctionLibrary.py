import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')# %matplotlib inline
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Dropout, Input,Activation,LeakyReLU,Lambda
#from keras.models import Model,Sequential
#from tensorflow.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from PIL import Image
import glob
import os
from astropy.io import fits
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import cm
import tensorflow as tf
import datafuncRik
import scipy.special as sp
################################################################################
############################### for GAN training ####################################
################################################################################


#ImageDataGenerator: Keras object used to preprocces the input images
#see: https://keras.io/api/preprocessing/image/ for more information
def createDataGen():
    return ImageDataGenerator(
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
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)



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
def plotGanEvolution(epoch,discrFakeLoss=[],discrRealLoss=[],
genLoss=[],discrFakeAccuracy=[],discrRealAccuracy= [],genAccuracy= []):
    fig1 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,5)))
    c=next(color)
    plt.plot(epoch,discrFakeLoss,label = 'discriminator fake data loss',c = c)
    c=next(color)
    plt.plot(epoch,discrRealLoss,label = 'discriminator real data loss',c = c)
    c=next(color)
    plt.plot(epoch,genLoss,label = 'generator loss',c = c)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('Loss evolution%d.png')
    fig2 = plt.figure()
    plt.plot(epoch,discrFakeAccuracy,label = 'discriminator fake data accuracy',c = c)
    c=next(color)
    plt.plot(epoch,discrRealAccuracy,label = 'discriminator real data accuracy',c = c)
    c=next(color)
    plt.plot(epoch,genAccuracy,label = 'generator data accuracy',c = c)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('Accuracy evolution.png')
    plt.close()


"""
create_gan

parameters:
    discriminator: a keras sequential model (network) with outputsize = 1 and inputsize = imagesize*imagesize*1
    generator: a keras sequential model (network) with outputsize = imagesize*imagesize*1 and inputsize = NoiseLength
returns:
    gan: a compiled keras model where the generator is followed by the discriminator and the discriminator is not trainable

"""
def create_gan(discriminator, generator, NoiseLength,opt):
    discriminator.trainable = False
    gan_input = Input(shape = (NoiseLength,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=opt,metrics=["accuracy"])
    return gan

"""
load_data

parameters:
    dir: a directory where the training images can be found (must contain * to expand and find multiple images)
    imagesize: the size to which the obtained images will be rescaled (imagesize*imagesize pixels)
returns:
    images: a numpy array containing the image information and dimensions (number of images *imagesize*imagesize*1 )

"""
def load_data2(dir,imagesize):
    directories = glob.glob(dir)
    image = fits.getdata(directories[0], ext=0)
    normmax = np.amax(image)
    #normalize to [0,255]
    image *=255/(normmax)
    img = Image.fromarray(image)
    img = img.resize((imagesize,imagesize),Image.BILINEAR )
    images= np.array([np.array(img)[:, :, np.newaxis]])
    for i in range(1,len(directories)):
        image = fits.getdata(directories[i], ext=0)
        normmax = np.amax(image)
        #normalize to [0,255]
        image*=255/(normmax)
        img = Image.fromarray(image)
        img = img.resize((imagesize,imagesize),Image.BILINEAR )
        image=np.array([np.array(img)[:, :, np.newaxis]])
        images = np.concatenate([images, image]) #add the rescaled image to the array normalize to [-1,+1]
    images = (images-127.5)/127.5
    return images

def load_data(dir,imagesize,dataMultiplier):
    directories = glob.glob(dir)[:100]
    image = fits.getdata(directories[0], ext=0)
    normmax = np.amax(image)
    #normalize to [0,255]
    image *=255/(normmax)
    img = Image.fromarray(image)
    img = img.resize((imagesize,imagesize),Image.LANCZOS)
    images= np.array([np.array(img)[:, :, np.newaxis]])
    for i in range(1,len(directories)):
        image = fits.getdata(directories[i], ext=0)
        normmax = np.amax(image)
        #normalize to [0,255]
        image*=255/(normmax)
        img = Image.fromarray(image)
        img = img.resize((imagesize,imagesize),Image.LANCZOS)
        image=np.array([np.array(img)[:, :, np.newaxis]])
        images = np.concatenate([images, image])
    xtrain= datagen.flow(images,shuffle=False,batch_size=len(directories) )[0]
    for j in range(1,dataMultiplier):
        addedXtrain = datagen.flow(images,shuffle=False,batch_size=len(directories) )[0]
        xtrain = np.concatenate([xtrain, addedXtrain])
    #normalize to [-1,+1]
    return (xtrain-127.5)/127.5
"""
plot_generated_images

parameters:
    epoch: the epoch at which th plots are made, used for naming the image
    generator: the generator neural network during the given epochs
    examples: the number of examples to be displayed in the plot
effect:
    saves images contain a number of random example images created by the generator

"""
def plot_generated_images(epoch, generator,NoiseLength,image_Size, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, NoiseLength])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,image_Size,image_Size)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i],interpolation=None)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cgan_generated_image %d.png' %epoch)
    plt.close()


"""
saveModel

parameters:
    Savedir: directory to save the generated models
    Modelname: name to be used for storing the networks of this run
    GAN: the trained combined generator and discriminator network, as created by create_gan, to be stored
    generator: the trained generator to be stored
    discr: the trained discriminator to be stored
effect:
    saves the keras models (neural networks) in their curren state

"""
def saveModel(save_dir,model_name,GAN,generator,discr):
    #test if the path exists, if not, creates it
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path_GAN = os.path.join(save_dir, 'GANfull'+ model_name)
    GAN.save(model_path_GAN)
    plot_model(GAN, to_file= 'full.png',show_shapes=True)

    model_path_generator = os.path.join(save_dir, 'generator'+model_name)
    generator.save(model_path_generator)
    plot_model(generator, to_file='generator.png',show_shapes=True)

    model_path_discriminator = os.path.join(save_dir, 'discriminator'+model_name)
    discr.save(model_path_discriminator)
    plot_model(discr, to_file='discriminator.png',show_shapes=True)
    print('Saved trained model at %s ' % model_path_GAN)





#discriminator = create_discriminator()
def modelCompiler(layers,optim,metr = None):
    model =  Sequential()
    for i in layers:
        model.add(i)
    model.compile(loss ='binary_crossentropy',optimizer = optim,metrics = metr)
    return model



"""
classicalGANtraining


parameters:
    generator: the generator netork to be used
    discriminator: the discriminator network to be used
    image_size: the size of the images
    NoiseLength: the number of elements in the noise vector used by the generator network
    epochs: the number of iterations over a set of image with its size eaqual to the training dataset
    batch_size: mini-batch size used for training the gan
    saveDir: directory where the trained networks will be stored
    PlotEpochs: the epoch interval at which examples of generated images will be created
    Use1sidedLabelSmooth: whether ornot to use onsided label smoothing (best true when using binary binary_crossentropy, best false when using MSE (LSGAN))
effect:
    Trains the GAN
    Saves plots with examples of generated images and the loss evolution of the gans components
    Saves the trained netorks at the requested and final epochs of training

"""

def classicalGANtraining(gen,discr,optim,dir,image_size,NoiseLength,epochs=1, batch_size=128,OverTrainDiscr=1,saveDir=None, PlotEpochs = 5,Use1sidedLabelSmooth = True, saveEpochs = []):
    #Loading the data
    #global generator
    generator = gen#modelCompiler(gen,optim)
    #global discriminator
    discriminator = discr#modelCompiler(discr,optim,metr=["accuracy"])
    #discriminator.compile(loss='binary_crossentropy', optimizer=optim,metrics=["accuracy"])
    X_train = load_data2(dir,image_size)
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
    y_real = 1
    datagen = createDataGen()
    batches = datagen.flow(X_train,y=None, batch_size = batch_size)
    if Use1sidedLabelSmooth:
        y_real = 0.9
    y_dis= np.zeros(2*batch_size)
    y_dis[:batch_size]= y_real
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
                # if the batch created by the generator is too small, resample
                if image_batch.shape[0] != batch_size:
                    batches = datagen.flow(X_train,y=None, batch_size = batch_size)
                    image_batch = batches.next()

                image_batch = image_batch.reshape(batch_size, image_size,image_size,1)
                #Construct different batches of  real and fake data
                X = np.concatenate([image_batch, generated_images])
                # Labels for generated and real data

                #Pre train discriminator on  fake and real data  before starting the gan.
                discriminator.trainable=True
                discriminator.train_on_batch(X, y_dis)

            #evaluations for the cost evolution of the discriminator
            discrimFakeEval = discriminator.evaluate(X[batch_size:],y_dis[batch_size:],verbose=0)
            discrimRealEval = discriminator.evaluate(X[:batch_size],y_dis[:batch_size],verbose=0)
            discrFakeLoss[e-1] += discrimFakeEval[0]/batch_count
            discrRealLoss[e-1] += discrimRealEval[0]/batch_count
            discrFakeAccuracy[e-1] += discrimFakeEval[1]/batch_count
            discrRealAccuracy[e-1] += discrimFakeEval[1]/batch_count

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
        # plot examples of generated images
        if e == 1 or e % PlotEpochs == 0:
            plot_generated_images(e, generator,NoiseLength,image_size)
        if e in saveEpochs:
            saveModel(saveDir,str(e)+'thEpoch.h5',gan,generator,discriminator)
    saveModel(saveDir,'finalModel.h5',gan,generator,discriminator)
    plotGanEvolution(epochArray,discrFakeLoss,discrRealLoss,genLoss,discrFakeAccuracy,discrRealAccuracy,genAccuracy)





################################################################################
############################### for image reconstruction########################
################################################################################


"""
dataLikeloss_FixedSparco

parameters:
    DataDir: directory containing the oifits-file to be used. Only the baselines and errors are used in the case artificial datasets are used
    filename: name of the OIfits file to be used
    V2Artificial: numpy array holding the squared visibilities when using artificial dataset
    CPArtificial: numpy array holding the closure phases when using artificial dataset
returns:
    data likelihood cost function for the provided data

"""
def dataLikeloss_FixedSparco(DataDir,filename,ImageSize,
x,
y,
primFlux,
secFlux,
denv,
dsec,
primaryDiameter,
pixelSize,forTraining = True
V2Artificial = None,CPArtificial = None):
    dataObj = datafuncRik.ReadFilesPionier(DataDir,filename)
    V2observed, V2err = dataObj['v2']
    nV2 = len(V2err)
    print(nV2)
    V2observed = tf.constant(V2observed)#conversion to tensor
    V2err = tf.constant(V2err)#conversion to tensor
    CPobserved, CPerr = dataObj['cp']
    nCP = len(CPerr)
    print(nCP)
    CPobserved = tf.constant(CPobserved)*np.pi/180 #conversion to degrees & cast to tensor
    CPerr = tf.constant(CPerr)*np.pi/180 #conversion to degrees & cast to tensor
    if V2Artificial is not None and CPArtificial is not None:
        V2observed = tf.constant(V2Artificial)
        CPobserved = tf.constant(CPArtificial)
    u, u1, u2, u3 = dataObj['u']
    v, v1, v2, v3 = dataObj['v']
    wavel0 = 1.65 *10**(-6) #TODO ask jaqcues what he used maybe, needs to be cast to tensor!!!
    wavelV2 = dataObj['wave'][0]
    wavelV2 = tf.constant(wavelV2,dtype = tf.complex128) #conversion to tensor
    wavelCP = dataObj['wave'][1]
    wavelCP = tf.constant(wavelCP,dtype = tf.complex128) #conversion to tensor
     #divide u,v by this number to get the pixelcoordinate
    x = x*np.pi*0.001/(3600*180)
    y = y*np.pi*0.001/(3600*180)
    primFlux = primFlux/100
    secFlux = secFlux/100
    primaryDiameter = primaryDiameter * np.pi*0.001/(3600*180)
    spacialFreqPerPixel = (3600/(0.001*ImageSize*pixelSize))*(180/np.pi)
    def offcenterPointFT(x,y,u,v):
        u = tf.constant(u,dtype = tf.complex128)
        v = tf.constant(v,dtype = tf.complex128)
        return tf.math.exp(-2*np.pi*1j*(x*u+y*v))

    #u,v must be provided in pixel number!!!
    def bilinearInterp(grid,ufunc,vfunc):
        #assert(max(ufunc)>255)),"u too big"
        #assert(ufunc<0),"u too small"
        #assert(vfunc>255),"v too big"
        #assert(vfunc<0), "v too small"
        ubelow = np.floor(ufunc).astype(int)
        vbelow = np.floor(vfunc).astype(int)
        uabove = ubelow +1
        vabove = vbelow +1
        coords = [[[0,ubelow[i],vbelow[i]] for i in range(len(ufunc))]]
        interpValues =  tf.gather_nd(grid,coords)*(uabove-ufunc)*(vabove-vfunc)
        coords1 =tf.constant([[[0,uabove[i],vabove[i]] for i in range(len(ufunc))]])
        interpValues += tf.gather_nd(grid,coords1)*(ufunc-ubelow)*(vfunc-vbelow)
        coords2 = tf.constant([[[0,uabove[i],vbelow[i]] for i in range(len(ufunc))]])
        interpValues +=  tf.gather_nd(grid,coords2)*(ufunc-ubelow)*(vabove-vfunc)
        coords3 = tf.constant([[[0,ubelow[i],vabove[i]] for i in range(len(ufunc))]])
        interpValues += tf.gather_nd(grid,coords3)*(uabove-ufunc)*(vfunc-vbelow)
        #print((uabove-ufunc)*(vabove-vfunc)+(ufunc-ubelow)*(vfunc-vbelow)+(ufunc-ubelow)*(vabove-vfunc) +(uabove-ufunc)*(vfunc-vbelow))
        return interpValues


    def compTotalCompVis(ftImages,ufunc,vfunc, wavelfunc):
        #to compute the radial coordinate in the uv plane so compute the ft of the primary, which is a uniform disk, so the ft is....
        radii = np.pi*primaryDiameter*np.sqrt(ufunc**2 + vfunc**2)# tf.constant(np.pi*primaryDiameter*np.sqrt(ufunc**2 + vfunc**2))
        ftPrimary = tf.constant(2*sp.jv(1,radii)/(radii),dtype = tf.complex128)
        #see extra function
        ftSecondary = offcenterPointFT(x,y,ufunc,vfunc)
        # get the total visibility amplitudes
        VcomplDisk = bilinearInterp(ftImages,(vfunc/spacialFreqPerPixel)+int(ImageSize/2),(ufunc/spacialFreqPerPixel)+int(ImageSize/2))
        #equation 4 in sparco paper:
        VcomplTotal = primFlux * ftPrimary* K.pow(wavelfunc/wavel0,-4)
        VcomplTotal += secFlux * ftSecondary* K.pow(wavelfunc/wavel0,dsec)
        VcomplTotal += (1-primFlux-secFlux) *VcomplDisk* K.pow(wavelfunc/wavel0, denv)
        VcomplTotal = VcomplTotal/((primFlux*K.pow(wavelfunc/wavel0,-4))+(secFlux*K.pow(wavelfunc/wavel0,dsec)) +((1-primFlux-secFlux)*K.pow(wavelfunc/wavel0,denv)))
        return VcomplTotal

    def internalloss(y_true,y_pred):
        #compute the fourier transform of the images
        y_pred = (tf.squeeze(y_pred,axis = 3) +1)/2   #/K.sum(K.sum(y_pred,axis =2),axis =1)
        y_pred = tf.cast((y_pred),tf.complex128)
        y_pred = tf.signal.ifftshift(y_pred,axes = (1,2))
        ftImages = tf.signal.fft2d(y_pred)#is complex!!
        ftImages = tf.signal.fftshift(ftImages,axes=(1,2))
        coordsMax = [[[[0,int(ImageSize/2),int(ImageSize/2)]]]]
        ftImages =ftImages/tf.cast(tf.math.abs(tf.gather_nd(ftImages,coordsMax)),tf.complex128)
        VcomplForV2 = compTotalCompVis(ftImages,u,v,wavelV2)
        V2generated = tf.math.abs(VcomplForV2)**2# computes squared vis for the generated images!!!!!!!!! use pow again!!!!!!!

        V2Chi2Terms = K.pow(V2observed - V2generated,2)/(K.pow(V2err,2)*nV2)# individual terms of chi**2 for V**2
        #V2Chi2Terms = V2Chi2Terms
        V2loss = K.sum(V2Chi2Terms,axis=1) #the squared visibility contribution to the loss

        CPgenerated  = tf.math.angle(compTotalCompVis(ftImages,u1,v1,wavelCP))
        CPgenerated += tf.math.angle(compTotalCompVis(ftImages,u2,v2,wavelCP))
        CPgenerated -= tf.math.angle(compTotalCompVis(ftImages,u3,v3,wavelCP))
        CPchi2Terms=K.pow(CPobserved-CPgenerated,2)/(K.pow(CPerr,2)*nCP)
        CPloss = K.sum(CPchi2Terms,axis=1)

        lossValue  = (K.mean(V2loss)*nV2 + K.mean(CPloss)*nCP)/(nV2+nCP)
        if forTraining = True:
            return  tf.cast(lossValue,tf.float32)
        else: return lossValue, V2loss , CPloss

    return internalloss


"""
adjustedCrossEntropy

parameters:
    y_true: the labels for used for training
    y_pred: the predictions made by the neural network
returns:
    binary binary_crossentropy loss function  with an added quadratic term in order to prevent over optimization

"""
def adjustedCrossEntropy(y_true,y_pred):
    mask = K.cast(K.less(0.95,K.mean(y_pred)), K.floatx())
    return K.binary_crossentropy(y_true, y_pred, from_logits=False) + mask*(80*(K.mean(y_pred)-0.95))**2


"""
createNetwork

parameters:
    discriminator: a GAN trained generator network
    generator: the corresponding discriminator network
returns:
    Keras model made of the combination of the input networks. This network has two outputs: the image and the discriminators response to this image.
    these outputs are optimized using the dataLikelihood cost of choice and the adjusted binary_crossentropy as cost function.

"""
def createNetwork(discriminator, generator,dataLikelihood,hyperParam):
    noise_input = Input(shape = (150,))
    x = generator(noise_input)
    gan_output= discriminator(x)
    gan= Model(inputs=noise_input, outputs=[gan_output,x])
    #losses are reversed compared to paper!
    losses = [adjustedCrossEntropy ,dataLikelihood]
    gan.compile(loss=losses,optimizer= Adam(learning_rate=0.0002,beta_1=0.91,beta_2=0.999, amsgrad=False),loss_weights=[hyperParam,1])
    discriminator.trainable = False
    return gan

"""
createNetwork

parameters:
    epoch: the epoch at which the image is created by the generator
    generator: the generator at the given epoch
    theOneNoiseVector: the noise vector used during this epoch of training
    image_Size: the pixel size of the create image
effect:
    Creates and stores an the image generated at a stage during training

"""
def plot_generated_images2(epoch, generator,theOneNoiseVector,image_Size):
    noise= np.array([theOneNoiseVector for i in range(100)])
    generated_images = generator.predict(noise)
    #print(discriminator.predict(generated_images))
    #print(discriminator.evaluate(generated_images,np.ones(batch_size)))
    generated_images = generated_images.reshape(100,image_Size,image_Size)

    plt.figure()
    mapable = plt.imshow((generated_images[0]+1)/2,origin = 'lower',extent = [-35,35,-35,35],cmap='hot',vmin= np.min((generated_images[0]+1)/2),vmax =np.max((generated_images[0]+1)/2))
    np.save('cgan_generated_image %d.png' %epoch,(generated_images[0]+1)/2)
    ax = plt.gca()
    ax.invert_xaxis()
    print(np.max((generated_images[0]+1)/2))
    print(np.min((generated_images[0]+1)/2))
    plt.xlabel(r'$\Delta \alpha (mas)$')
    plt.ylabel(r'$\Delta \delta (mas)$')
    plt.colorbar(mapable)
    plt.savefig('cgan_generated_image %d.png' %epoch)
    plt.close()

"""
saveModel2

parameters:
    Savedir:directory to save the generator
    Modelname: name of the reconstruction
    generator: the generator network to be stored
effect:
    saves the generator at the end of training (is not neccecary)

"""
def saveModel2(Savedir,Modelname,generator):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path_generator = os.path.join(save_dir, 'generator'+Modelname)
    generator.save(model_path_generator)

"""
updateMeanAndVariance

parameters:
    I: the Number of contributions to the mean and variance that this update would make
    mean: the mean image before this update
    variance: the variance before for this update
    theOneNoiseVector: the noise vector used
    Generator: the generator in the state producing the image used in the update
returns:
    updated mean and varience to take into account the given mean and varience images which are the I'th

"""
def updateMeanAndVariance(I,mean,variance,theOneNoiseVector, Generator):
    generatedImage = Generator.predict(np.array([theOneNoiseVector for i in range(2)]))[0]
    generatedImage = np.squeeze((generatedImage+1)/2)
    generatedImage = generatedImage/np.sum(generatedImage)
    print(generatedImage.shape)
    newmean = mean + (generatedImage - mean)/I
    if I > 1:
        newvariance = ((I-1)*variance + (I-1)*(mean -newmean)**2 + (generatedImage - newmean)**2)/(I-1)
        variance = newvariance
    mean = newmean
    return mean, variance


"""
plotEvolution

parameters:
    epoch: array or list containing the epochs of training at which values are plotted
    diskyLoss: array or list holding the cost computed for the discriminators output at the given epochs
    fitLoss: array or list holding the cost computed using the data likelihood at the given epochs
returns:
    Makes various plots of both the components of the objective function and the total objective function itself

"""
def plotEvolution(epoch,hyperParam,diskyLoss=[],fitLoss=[]):
    # plots of both terms of the objective function and the total objective function itself as  a function of epoch
    fig1 = plt.figure()
    plt.plot(epoch,diskyLoss,label = r'$\mu f_{prior}$',c = 'b',alpha=0.5)
    plt.plot(epoch,fitLoss,label = r'$f_{data}$',c = 'g',alpha=0.5)
    plt.plot(epoch,fitLoss+diskyLoss,label = r'$f_{data}+\mu f_{prior}$',c = 'r',alpha=0.5)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('Loss_evolution.png')
    fig2 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)

    #plots of both terms of the objective function and the total objective function itself as  a function of epoch on log scale
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.plot(epoch,np.log10(diskyLoss),label = r'$\mu f_{prior}$',c = c,alpha=0.7)
    c=next(color)
    plt.plot(epoch,np.log10(fitLoss),label = r'$f_{data}$',c = c,alpha=0.7)
    c=next(color)
    plt.plot(epoch,np.log10(fitLoss+diskyLoss),label = r'$f_{data}+\mu f_{prior}$',c = c,alpha=0.7)
    plt.legend()
    plt.ylabel(r'$log_{10}$(loss)')
    plt.xlabel('epoch')
    plt.savefig('Log10_Loss_evolution.png')


    #plots the regularization cost term as a function of epoch on a log scale
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.plot(epoch,np.log10(diskyLoss),c = c,alpha=0.7)
    plt.legend()
    plt.ylabel(r'$\mu f_{prior}$')
    plt.xlabel('epoch')
    plt.savefig('Log10_Loss_discriminatorEvolution.png')

    #plots the data likelihood term as a function of epoch on a log scale
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.plot(epoch,np.log10(fitLoss),c = c,alpha=0.7)
    plt.legend()
    plt.ylabel(r'$f_{data}$')
    plt.xlabel('epoch')
    plt.savefig('Log10_Loss_fitevolution.png')


    #plots the total objective function as a function of epoch
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.plot(epoch,np.log10(fitLoss+diskyLoss),c = c,alpha=0.7)
    plt.legend()
    plt.ylabel(r'$f_{data}+\mu f_{prior}$')
    plt.xlabel('epoch')
    plt.savefig('Log10_Loss_TotalEvolution.png')

    #plots the dataLikelihood troughout training agains the regularization term
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.scatter(fitLoss,diskyLoss,c = c,alpha=0.5)
    plt.legend()
    plt.ylabel(r'$\mu f_{prior}$')
    plt.xlabel(r'$f_{data}$')
    plt.xlim(0,1000)
    plt.ylim(0,1000)
    plt.savefig('discrLossVSFitLoss1000range.png')


    #plots the dataLikelihood troughout training agains the regularization term, zoomed in on the origin
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.scatter(fitLoss,diskyLoss,c = c,alpha=0.5)
    plt.legend()
    plt.ylabel(r'$\mu f_{prior}$')
    plt.xlabel(r'$f_{data}$')
    plt.ylim(0,10)
    plt.xlim(0,10)
    plt.savefig('discrLossVSFitLoss100range.png' )

    ##plots the dataLikelihood troughout training agains the regularization term divided by the hyperParam, zoomed in on the origin
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.scatter(fitLoss,diskyLoss/hyperParam,c = c,alpha=0.5)
    plt.legend()
    plt.ylabel(r'$ f_{prior}$')
    plt.xlabel(r'$f_{data}$')
    plt.ylim(0,100)
    plt.xlim(0,100)
    plt.savefig('discrLossVSFitLoss100rangeNoMu.png')

    #plots the dataLikelihood troughout training agains the regularization term, on a log scale
    fig4 = plt.figure()
    color=iter(cm.rainbow(np.linspace(0,1,3)))
    c=next(color)
    plt.scatter(np.log10(fitLoss),np.log10(diskyLoss),c = c,alpha=0.5)
    plt.ylabel(r'$log_{10}(\mu f_{prior })$')
    plt.xlabel(r'$log_{10}(f_{data})$')
    plt.savefig('logdiscrLossVSFitLossNo.png')

"""
plotMeanAndSTD

parameters:
    mean: the mean image to be displayed
    variance: the vaiance image to be used
effect:
    stores plots of the mean image, plots are made with a 5, 3 and no sigma significance contour.
    Plots of the variance maps are also stored.
    !!!!! In the current implementation the significance is calculated based on the results obtained for a different input vector
     passed to the generator and is thus not a proper significance !!!!!!!


"""
def plotMeanAndSTD(mean,variance):
    # plots the mean image,
    plt.figure()
    mapable = plt.imshow(mean/np.max(mean),origin = 'lower',extent = [-35,35,-35,35],cmap='hot',vmin= 0.,vmax =1.)
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlabel(r'$\Delta \alpha (mas)$')
    plt.ylabel(r'$\Delta \delta (mas)$')
    plt.colorbar(mapable)
    plt.savefig('meanImage.png' )
    plt.close()

    #plots the variance image
    plt.figure()
    mapable = plt.imshow(np.sqrt(variance),origin = 'lower',extent = [-35,35,-35,35],cmap='hot')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlabel(r'$\Delta \alpha (mas)$')
    plt.ylabel(r'$\Delta \delta (mas)$')
    plt.savefig('rmsImage.png' )
    plt.close()

    #plot the mean image with a 5 sigma significance contour.
    plt.figure()
    mapable = plt.imshow(mean/np.max(mean),origin = 'lower',extent = [-35,35,-35,35],cmap='hot',vmin= 0.,vmax =1.)
    plt.contour(mean/np.sqrt(variance), [5.], colors='b', origin='lower', extent=[-35,35,-35,35])
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlabel(r'$\Delta \alpha (mas)$')
    plt.ylabel(r'$\Delta \delta (mas)$')
    plt.colorbar(mapable)
    plt.savefig('Sign5image.png')
    plt.close()

    #plot the mean image with a 3 sigma significance contour.
    plt.figure()
    mapable = plt.imshow(mean/np.max(mean),origin = 'lower',extent = [-35,35,-35,35],cmap='hot',vmin= 0.,vmax =1.)
    plt.contour(mean/np.sqrt(variance), [3.], colors='b', origin='lower', extent=[-35,35,-35,35])
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlabel(r'$\Delta \alpha (mas)$')
    plt.ylabel(r'$\Delta \delta (mas)$')
    plt.colorbar(mapable)
    plt.savefig('Sign3image.png')
    plt.close()

    #plots the corresponding mean over standard deviation map
    plt.figure()
    mapable = plt.imshow(mean/np.sqrt(variance),origin = 'lower',extent = [-35,35,-35,35],cmap='hot')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.xlabel(r'$\Delta \alpha (mas)$')
    plt.ylabel(r'$\Delta \delta (mas)$')
    plt.colorbar(mapable)
    plt.savefig('meanOverSignImage.png')
    plt.close()

"""
reconsruction

parameters:
    generator:
    discriminator:
    dataLikelihood:
    epochs:
    image_Size:
    hyperParam:
    NoiseLength:
    beginepoch: the epoch at which the first contribution to the mean image is made
    RandomWalkStepSize:
    alterationInterval:
    plotinterval: the epoch interval between changes to the Noisevector and contributions to the
    saveDir: directory to store the generator in its final state
Returns:
    the reconstructed image (mean over noisevectors)



"""
def reconsruction(Generator, discriminator,opt,dataLikelihood , epochs = 21000,image_Size = 64,hyperParam = 2,NoiseLength = 100,beginepoch =9000,RandomWalkStepSize =0.5,alterationInterval = 500,plotinterval = 3000,saveDir  = ''):
    #create the network with two cost terms
    discriminator.compile(loss=adjustedCrossEntropy, optimizer=opt)
    discriminator.trainable =False
    Generator.trainable = True
    Generator.add(Lambda(lambda x: 2*((x-K.min(x))/(K.max(x) -K.min(x)))-1 ))

    fullNet  = createNetwork(discriminator,Generator,dataLikelihood, hyperParam)
    # initialize empty arrays to store the cost evolution
    diskyLoss = np.zeros(epochs)
    fitLoss = np.zeros(epochs)
    # make an array with the epoch of the reconstruction
    epoch = np.linspace(1,epochs,epochs)
    #initialize the mean and variance images
    mean = np.zeros([image_Size,image_Size])
    variance = np.zeros([image_Size,image_Size])
    # start tracking the amount of contributions made to the mean and variance image
    i = 1
    # initialize a initial random noise vector
    theOneNoiseVector = np.random.normal(0,1,150)
    y_gen =[np.ones(1),np.ones(1)]
    for e in range(1, epochs+1 ):
        #generate  random noise as an input  to  initialize the  generator
        noise= np.array([theOneNoiseVector])
        hist= fullNet.train_on_batch(noise, y_gen)
        diskyLoss[e-1] += hyperParam*(hist[1])
        fitLoss[e-1] += (hist[2])
        # alter the noise vector each alterationInterval, if the first contribution to the mean and var has been made
        if e >= beginepoch and e%alterationInterval ==0:
            theOneNoiseVector = (theOneNoiseVector  + np.random.normal(0,RandomWalkStepSize,150))/(1.+RandomWalkStepSize)
        #update the mean and variance 1 iteration before the noisevector is altered
        if (e+1) >= beginepoch and (e+1)%alterationInterval ==0:
            mean, variance = updateMeanAndVariance(i,mean,variance,theOneNoiseVector,Generator)
            i += 1
        #plot the image after the noisevector is altered, this is the new start position for re-convergence
        if e % plotinterval== 0 or e ==1:
            plot_generated_images2(e, Generator,theOneNoiseVector,image_Size)
        #plot the image contributing to the mean and variance
        if (e+1) % plotinterval== 0:
            plot_generated_images2(e, Generator,theOneNoiseVector,image_Size)
    #plot the loss evolution
    plotEvolution(epoch,hyperParam,diskyLoss,fitLoss)
    #plot and store the mean and variance image
    plotMeanAndSTD(mean,variance)

    return mean, Image, diskyLoss, fitLoss
