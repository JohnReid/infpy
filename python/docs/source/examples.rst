..
.. Copyright John Reid 2012
..
.. This is a reStructuredText document. If you are reading this in text format, it can be 
.. converted into a more readable format by using Docutils_ tools such as rst2html.
..

.. _Docutils: http://docutils.sourceforge.net/docs/user/tools.html



Examples
========

Here we give some examples of how to use infpy to use Gaussian processes models. We show how to

- examine the effects of different kernels
- model noisy data
- learn the hyperparameters of the model
- use periodic covariance functions

 

Data
----

We generate some noisy test data from a modulated sine using this script

.. literalinclude:: _static/Code/data_gen.py
    :language: python
    :lines: 5-

.. image::  _static/Images/simple-example-data.*
   
The :math:`f`\ s are noisy observations of the underlying :math:`Y`\ s.
How can we model this using GPs?




Test different kernels
----------------------

Using the following function as an interface to the infpy GP library

.. literalinclude:: _static/Code/infpy_predict.py
    :language: python
    :lines: 5-

we can test various different kernels
to see how well they fit the data. For instance a simple squared
exponential kernel with some noise

.. code-block:: python

	import infpy.gp.kernel_short_names as kernels
	K = kernels.SE() + kernels.Noise(.1) # create a kernel composed of a squared exponential kernel and a small noise term
	predict_values(K, ’simple-example-se’)

will generate

.. image::  _static/Images/simple-example-se.*

if we change the kernel so that the squared exponential term is given a
shorter characteristic length scale

.. code-block:: python

	K = kernels.SE([.1]) + kernels.Noise(.1) # Try a different kernel with a shorter characteristic length scale
	predict_values(K, ’simple-example-se-shorter’)

we will generate

.. image::  _static/Images/simple-example-se-shorter.*

Here the shorter length scale means that data points are less correlated
as the GP allows more variation over the same distance. The estimates of
the noise between the training points is now much higher.





Effects of noise
----------------

If we try a kernel with more noise

.. code-block:: python

	K = kernels.SE([4.]) + kernels.Noise(1.)
	predict_values(K, ’simple-example-more-noise’)

we get the following estimates showing that the training data does not
affect the predictions as much

.. image::  _static/Images/simple-example-more-noise.*




Learning the hyperparameters
----------------------------

Perhaps we are really interested in learning the hyperparameters. We can
acheive this as follows

.. code-block:: python

	K = kernels.SE([4.0]) + kernels.Noise(.1) # Learn kernel hyper-parameters
	predict_values(K, ’simple-example-learnt’, learn=True)

and the result is

.. image::  _static/Images/simple-example-learnt.*

where the learnt length-scale is about 2.6 and the learnt noise
level is about 0.03. 




UK gas consumption example
--------------------------

We can apply these techniques with periodic
covariance functions to UK gas consumption data
from R.

.. image::  _static/Images/gp-uk-gas-data.*

The data which has been shifted to have a mean of 0. 

.. image::  _static/Images/gp-uk-gas-general.*

A GP
incorporating some noise and a fixed long length scale squared
exponential kernel.

.. image::  _static/Images/gp-uk-gas-periodic.*

As before but with a periodic
term. 

.. image::  _static/Images/gp-uk-gas-periodic-reasonable-length.*

Now with a periodic term with a
reasonable period.



Modulated sine example
----------------------

.. image::  _static/Images/gp-modulated-sin-data.*

The data: a modulated noisy sine wave. 


.. image::  _static/Images/gp-modulated-sin-se.*

GP with a squared exponential kernel.


.. image::  _static/Images/gp-modulated-sin-se-shorter.*

GP with a squared
exponential kernel with a shorter length scale.


.. image::  _static/Images/gp-modulated-sin-more-noise.*

GP with
a squared exponential kernel with a larger noise term.


.. image::  _static/Images/gp-modulated-sin-learnt.*

GP with
a squared exponential kernel with learnt hyper-parameters.
This maximises the likelihood.


