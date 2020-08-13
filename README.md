# Organic
Object Reconstruction with Generative Adversarial Networks from InterferometriC data

In order to use the image reconstruction method two scripts need to be ran.


TrainGAN.py

First a GAN network must be trained. 
This can be done by running the TrainGAN.py
In order to do so a training dataset, consisting of fits files is needed.
Both the parameters governing the training routine and the network architectures can be altered in this file.  
The directory where the trained networks are stored can also be set. 




ImageReconstruction.py 

After a GAN is trained the image reconstruction can be preformed.
This can be preformed by running the ImageReconstruction.py script.

Here the various parameters governing the reconstruction can be altered.
The direcories ofthe relevant trained neural networks must also be suplied. 

A Datalikelihood loss function must be initialized beforehand and passed to the image reconstruction function.
three datalikelihood- cost functions can be chosen 
these differ by having 
- fixed sparco paramaters
- no sparco contributions
- variable sparco parameters, which are fitted during training (yet to be implemented!!!!), this datalikelihood only works with the ImagereconstructioAndFit function.



In the the results in the paper where produced in a conda environment with the following packages:
- Package                   version used             

- _libgcc_mutex          ;&  0.1            ;&           main
- _tflow_select             2.1.0                       gpu
- absl-py                   0.9.0                    py36_0
- asn1crypto                1.3.0                    py36_0
- astor                     0.8.1                    pypi_0    pypi
- astropy                   4.0              py36h7b6447c_0
- attrs                     19.3.0                     py_0
- blas                      1.0                         mkl
- blinker                   1.4                      py36_0
- c-ares                    1.15.0            h7b6447c_1001
- ca-certificates           2020.1.1                      0
- cachetools                4.0.0                    pypi_0    pypi
- cairo                     1.14.12              h8948797_3
- certifi                   2019.11.28               py36_0
- cffi                      1.14.0           py36h2e261b9_0
- chardet                   3.0.4                 py36_1003
- click                     7.0                      py36_0
- cryptography              2.8              py36h1ba5d50_0
- cudatoolkit               10.1.243             h6bb024c_0
- cudnn                     7.6.5                cuda10.1_0
- cupti                     10.1.168                      0
- cycler                    0.10.0                   py36_0
- dbus                      1.13.12              h746ee38_0
- expat                     2.2.6                he6710b0_0
- fontconfig                2.13.0               h9420a91_0
- freetype                  2.9.1                h8a8886c_1
- fribidi                   1.0.5                h7b6447c_0
- gast                      0.2.2                    pypi_0    pypi
- glib                      2.63.1               h5a9c865_0
- google-auth               1.11.2                     py_0
- google-auth-oauthlib      0.4.1                      py_2
- google-pasta              0.1.8                      py_0
- graphite2                 1.3.13               h23475e2_0
- graphviz                  2.40.1               h21bd128_2
- grpcio                    1.27.2           py36hf8bcb03_0
- gst-plugins-base          1.14.0               hbbd80ab_1
- gstreamer                 1.14.0               hb453b48_1
- h5py                      2.10.0           py36h7918eee_0
- harfbuzz                  1.8.8                hffaf4a1_0
- hdf5                      1.10.4               hb1b8bf9_0
- hypothesis                5.5.4                      py_0
- icu                       58.2                 h9c2bf20_1
- idna                      2.9                      pypi_0    pypi
- importlib_metadata        1.5.0                    py36_0
- intel-openmp              2020.0                      166
- jdcal                     1.4.1                      py_0
- jpeg                      9b                   h024ee3a_2
- keras                     2.3.1                    pypi_0    pypi
- keras-applications        1.0.8                      py_0
- keras-preprocessing       1.1.0                      py_1
- kiwisolver                1.1.0            py36he6710b0_0
- ld_impl_linux-64          2.33.1               h53a641e_7
- libedit                   3.1.20181209         hc058e9b_0
- libffi                    3.2.1                hd88cf55_4
- libgcc-ng                 9.1.0                hdf63c60_0
- libgfortran-ng            7.3.0                hdf63c60_0
- libpng                    1.6.37               hbc83047_0
- libprotobuf               3.11.4               hd408876_0
- libstdcxx-ng              9.1.0                hdf63c60_0
- libtiff                   4.1.0                h2733197_0
- libuuid                   1.0.3                h1bed415_2
- libxcb                    1.13                 h1bed415_1
- libxml2                   2.9.9                hea5a465_1
- markdown                  3.2.1                    pypi_0    pypi
- matplotlib                3.1.3                    py36_0
- matplotlib-base           3.1.3            py36hef1b27d_0
- mkl                       2020.0                      166
- mkl-service               2.3.0            py36he904b0f_0
- mkl_fft                   1.0.15           py36ha843d7b_0
- mkl_random                1.1.0            py36hd6b4f25_0
- more-itertools            8.2.0                      py_0
- ncurses                   6.2                  he6710b0_0
- numpy                     1.18.1                   pypi_0    pypi
- numpy-base                1.18.1           py36hde5b4d6_1
- oauthlib                  3.1.0                      py_0
- olefile                   0.46                     py36_0
- openssl                   1.1.1d               h7b6447c_4
- opt_einsum                3.1.0                      py_0
- packaging                 20.1                       py_0
- pango                     1.42.4               h049681c_0
- pcre                      8.43                 he6710b0_0
- pillow                    7.0.0            py36hb39fc2d_0
- pip                       20.0.2                   py36_1
- pixman                    0.38.0               h7b6447c_0
- pluggy                    0.13.1                   py36_0
- protobuf                  3.11.3                   pypi_0    pypi
- psutil                    5.6.7            py36h7b6447c_0
- py                        1.8.1                      py_0
- pyasn1                    0.4.8                      py_0
- pyasn1-modules            0.2.8                    pypi_0    pypi
- pycparser                 2.19                     py36_0
- pydot                     1.4.1                    py36_0
- pyjwt                     1.7.1                    py36_0
- pyopenssl                 19.1.0                   py36_0
- pyparsing                 2.4.6                      py_0
- pyqt                      5.9.2            py36h05f1152_2
- pysocks                   1.7.1                    py36_0
- pytest                    5.3.5                    py36_0
- pytest-arraydiff          0.3              py36h39e3cac_0
- pytest-astropy            0.8.0                      py_0
- pytest-astropy-header     0.1.2                      py_0
- pytest-doctestplus        0.5.0                      py_0
- pytest-openfiles          0.4.0                      py_0
- pytest-remotedata         0.3.2                    py36_0
- python                    3.6.10               h0371630_0
- python-dateutil           2.8.1                      py_0
- pyyaml                    5.3                      pypi_0    pypi
- qt                        5.9.7                h5867ecd_1
- readline                  7.0                  h7b6447c_5
- requests                  2.23.0                   pypi_0    pypi
- requests-oauthlib         1.3.0                      py_0
- rsa                       4.0                        py_0
- scipy                     1.4.1            py36h0b6359f_0
- setuptools                45.2.0                   py36_0
- sip                       4.19.8           py36hf484d3e_0
- six                       1.14.0                   py36_0
- sortedcontainers          2.1.0                    py36_0
- sqlite                    3.31.1               h7b6447c_0
- tensorboard               2.1.0                     py3_0
- tensorflow                2.1.0           gpu_py36h2e5cdaa_0
- tensorflow-base           2.1.0           gpu_py36h6c5654b_0
- tensorflow-estimator      2.1.0              pyhd54b08b_0
- tensorflow-gpu            2.1.0                h0d30ee6_0
- termcolor                 1.1.0                    pypi_0    pypi
- tk                        8.6.8                hbc83047_0
- tornado                   6.0.3            py36h7b6447c_3
- tqdm                      4.42.1                     py_0
- urllib3                   1.25.8                   py36_0
- wcwidth                   0.1.8                      py_0
- werkzeug                  1.0.0                      py_0
- wheel                     0.34.2                   py36_0
- wrapt                     1.12.0                   pypi_0    pypi
- xz                        5.2.4                h14c3975_4
- zipp                      2.2.0                      py_0
- zlib                      1.2.11               h7b6447c_3
- zstd                      1.3.7                h0b5b093_0




