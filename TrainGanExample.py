#from astropy.io import fits
#from PIL import Image
#import numpy as np
#import matplotlib.pyplot as plt
import organic.organic as org




if __name__ == "__main__":

    test = org.GAN()

    dir = '/Users/jacques/Work/Organic/'
    file = 'processedcube03.fits'
    imgs = org.inputImages(dir, file)

    test.train(imgs, nepochs=2, plotEpochs=2)
