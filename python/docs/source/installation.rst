..
.. Copyright John Reid 2012
..
.. This is a reStructuredText document. If you are reading this in text format, it can be 
.. converted into a more readable format by using Docutils_ tools such as rst2html.
..

.. _Docutils: http://docutils.sourceforge.net/docs/user/tools.html



Installation
============

The *infpy* package depends on the python packages cookbook, numpy, scipy and matplotlib.
If these are present installation should be straightforward
with your preferred method.

On Ubuntu::

    sudo apt-get install python-numpy
    sudo apt-get install python-scipy
    sudo apt-get install python-matplotlib
    sudo apt-get install python-pip
    sudo pip install infpy cookbook

On a generic Linux/UNIX system with pip_ already installed::

    sudo pip install numpy scipy matplotlib infpy cookbook
    
Or you could use easy_install_ or install the packages by hand yourself from PyPI__.

.. __: http://pypi.python.org/pypi
.. _easy_install: http://packages.python.org/distribute/easy_install.html
.. _pip: http://pypi.python.org/pypi/pip

    