#
# Copyright John Reid 2009, 2010, 2011, 2012
#

from setuptools import setup, find_packages
import os



def read(*fnames):
    """
    Utility function to read the README file.
    Used for the long_description.  It's nice, because now 1) we have a top level
    README file and 2) it's easier to type in the README file than to put a raw
    string in below ...
    """
    return open(os.path.join(os.path.dirname(__file__), *fnames)).read()



setup(
    name                   = "infpy",
    version                = read('infpy', 'VERSION').strip().split('-')[0],
    author                 = "John Reid",
    author_email           = "johnbaronreid@netscape.net",
    description            = "A python inference library",
    license                = "BSD",
    # metadata for upload to PyPI
    keywords               = "Inference probabilistic models Gaussian processes Dirichlet processes",
    url                    = "http://sysbio.mrc-bsu.cam.ac.uk/johns/infpy/docs/build/html/",   # project home page, if any
    packages               = find_packages(),
    package_data           = { 'infpy' : ['README', 'VERSION', 'LICENSE'] },
    install_requires       = ['distribute'],
    long_description       = read('infpy', 'README'),
    classifiers            = [
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)

