#!/usr/bin/env python
import setuptools
setuptools.setup(name='OrganicOI',
version='0.0.22',
      description='An image reconstuction algorithm using GANs',
      author='Jacques Kluska',
      author_email='jacques.kluska@kuleuven.be',
      url='https://github.com/kluskaj/Organic',
      packages=setuptools.find_packages('.'),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ),
    install_requires=['astropy', 'matplotlib', 'numpy', 'scikit-learn', 'scipy', 'keras', 'tensorflow']
)
