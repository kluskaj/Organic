import organic.organic as org
import os
import matplotlib.pyplot as plt


def main():
    # Loading the generators and discriminators
    dis = os.path.expandvars('theGANextended2/saved_models/discriminatorfinalModel.h5')
    gen = os.path.expandvars('theGANextended2/saved_models/generatorfinalModel.h5')

    test = org.GAN(dis=dis, gen=gen)

    # Data folder and files
    datafolder = '/Users/jacques/Work/IRAS08Var/data/'
    datafiles = 'epoch1.fits'
    # setting SPARCO
    sparco = org.SPARCO(fstar = 0.6179, denv = 0.286, dsec = -2, xsec = -0.36, ysec = -1.33, fsec = 0.0175, UDstar = 0.5)

    # Launching image reconstruction
    test.ImageReconstruction(datafiles, sparco, data_dir = datafolder, mu=[0.1, 1, 10], ps=0.6, diagnostics=False, epochs=50, nrestart=50, name='test')


if __name__ == "__main__":
    main()
