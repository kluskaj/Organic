import Organic as org
import os


def main():
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
    test.ImageReconstruction(datafiles, sparco, data_dir = datafolder, mu=[0.1, 1, 10], ps=[0.4, 0.6])


if __name__ == "__main__":
    main()
