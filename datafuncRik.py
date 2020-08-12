import numpy as np
import oifits
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import fnmatch
import math
import gc

def Bases(data):

    # gives you back the bases lengths for V2 and cpres
    u = data['u']
    v = data['v']
    wave = data['wave']
    base = np.sqrt(u[0]**2 + v[0]**2)
    B1 = np.sqrt(u[1]**2 + v[1]**2)
    B2 = np.sqrt(u[2]**2 + v[2]**2)
    B3 = np.sqrt(u[3]**2 + v[3]**2)
    Bmax = np.maximum(B1, B2, B3)

    return base, Bmax




def read_PIONIER(file):
    PIONIER = oifits.open(file)
    PIONIERv2 = PIONIER.allvis2
    PIONIERcp = PIONIER.allt3

    wave = np.array(PIONIERv2['eff_wave'])
    u = np.array(PIONIERv2['ucoord'])/wave
    v = np.array(PIONIERv2['vcoord'])/wave
    V2 = np.array(PIONIERv2['vis2data'])
    V2err = np.array(PIONIERv2['vis2err'])
    wavecp = np.array(PIONIERcp['eff_wave'])
    u1 = np.array(PIONIERcp['u1coord'])/wavecp
    v1 = np.array(PIONIERcp['v1coord'])/wavecp
    u2 = np.array(PIONIERcp['u2coord'])/wavecp
    v2 = np.array(PIONIERcp['v2coord'])/wavecp
    u3 = u1 + u2
    v3 = v1 + v2
    CP = np.array(PIONIERcp['t3phi'])
    CPerr = np.array(PIONIERcp['t3phierr'])

    # print('the keys:', PIONIERv2.mask)
    # TODO: Temporary fix for 0 errorbar
    CPerr = CPerr + (CPerr == 0)*1e12
    V2err = V2err + (V2err == 0)*1e12

    data = {}
    data['u'] = (u, u1, u2, u3)
    data['v'] = (v, v1, v2, v3)
    data['wave'] = (wave, wavecp)
    data['v2'] = (V2, V2err)
    data['cp'] = (CP, CPerr)

    return data


def ReadFilesPionier(dir, files):

    listOfFiles = os.listdir(dir)
    #print listOfFiles
    #print files
    pattern = files
    i = 0
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            i += 1
            inform ('Reading '+entry+'...')
            if i == 1:
                print(dir+entry)
                data = read_PIONIER(dir+entry)
            else:
                datatmp = read_PIONIER(dir+entry)
                # Appending all the stuff together
                # Starting with u coordinates
                ut, u1t, u2t, u3t = datatmp['u']
                u, u1, u2, u3 = data['u']
                u = np.append(u, ut)
                u1 = np.append(u1, u1t)
                u2 = np.append(u2, u2t)
                u3 = np.append(u3, u3t)
                data['u'] = (u, u1, u2, u3)
                # v coordinates
                vt, v1t, v2t, v3t = datatmp['v']
                v, v1, v2, v3 = data['v']
                v = np.append(v, vt)
                v1 = np.append(v1, v1t)
                v2 = np.append(v2, v2t)
                v3 = np.append(v3, v3t)
                data['v'] = (v, v1, v2, v3)
                # wavelength tables
                wavet, wavecpt = datatmp['wave']
                wave, wavecp = data['wave']
                wave = np.append(wave, wavet)
                wavecp = np.append(wavecp, wavecpt)
                data['wave'] = (wave, wavecp)
                # Visibility squared
                v2t, v2et = datatmp['v2']
                v2, v2e = data['v2']
                v2 = np.append(v2, v2t)
                v2e = np.append(v2e, v2et)
                data['v2'] = (v2, v2e)
                # closure phases
                cpt, cpet = datatmp['cp']
                cp, cpe = data['cp']
                cp = np.append(cp, cpt)
                cpe = np.append(cpe, cpet)
                data['cp'] = (cp, cpe)

    inform2('Done!')

    return data


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def inform(msg):
    print(bcolors.OKBLUE + msg + bcolors.ENDC)


def inform2(msg):
    print(bcolors.OKGREEN + msg + bcolors.ENDC)


def warn(msg):
    print(bcolors.WARNING + msg + bcolors.ENDC)


def log(msg, dir):
    f = open(dir+"log.txt", "a")
    f.write(msg+"\n")
    f.close()

#data = ReadFilesPionier('C:\\Users\\rik\\Desktop\\thesis\\OIfits\\','IRAS08544-4431_PIONIER_alloidata.fits')
#data_plot_PIONIER(data)
#read_PIONIER(data)
