import Organic as org
import os

if __name__ == "__main__":

    # Loading the generators and discriminators
    dis = os.path.expandvars('theGANextended2/saved_models/discriminatorfinalModel.h5')
    gen = os.path.expandvars('theGANextended2/saved_models/generatorfinalModel.h5')

    test = org.GAN(dis=dis, gen=gen)

    # Data folder and files
    datafolder = '/Users/jacques/Work/Organic/'
    datafiles = '*.fits'
    # setting SPARCO
    sparco = org.SPARCO()

    # Launching image reconstruction
    test.ImageReconstruction(datafiles, sparco, dataDir = datafolder)
    
