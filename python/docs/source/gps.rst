..
.. Copyright John Reid 2012
..
.. This is a reStructuredText document. If you are reading this in text format, it can be 
.. converted into a more readable format by using Docutils_ tools such as rst2html.
..

.. _Docutils: http://docutils.sourceforge.net/docs/user/tools.html


What are Gaussian processes?
============================

Often we have an inference problem involving :math:`n` data,

.. math:: \mathcal{D} = \{(\boldsymbol{x_i},y_i)|i=1,\ldots,n, \boldsymbol{x_i} \in \mathcal{X}, y_i \in \mathbb{R}\}

where the :math:`\boldsymbol{x_i}` are the inputs and the :math:`y_i`
are the targets. We wish to make predictions, :math:`y_*`\ , for new
inputs :math:`\boldsymbol{x_*}`\ . Taking a Bayesian perspective we
might build a model that defines a distribution over all possible
functions, :math:`f: \mathcal{X} \rightarrow \mathbb{R}`\ . We can
encode our initial beliefs about our particular problem as a prior over
these functions. Given the data, :math:`\mathcal{D}`\ , and applying
Bayes’ rule we can infer a posterior distribution. In particular, for
any given :math:`\boldsymbol{x_*}` we can calculate or approximate a
predictive distribution over :math:`y_*` under this posterior.

*Gaussian processes* (GPs) are probability distributions over functions
for which this inference task is tractable. They can be seen as a
generalisation of the Gaussian probability distribution to the space of
functions. That is a multivariate Gaussian *distribution* defines a
distribution over a finite set of random variables, a Gaussian *process*
defines a distribution over an infinite set of random variables (for example
the real numbers). GP domains are not restricted to the real numbers, any
space with a dot product is suitable. Analagously to a multivariate
Gaussian distribution, a GP is defined by its mean, :math:`\mu`\ , and
covariance, :math:`k`\ . However for a GP these are themselves
functions, :math:`\mu: \mathcal{X} \rightarrow \mathbb{R}` and
:math:`k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}`\ . In
all that follows we assume :math:`\mu(\boldsymbol{x}) = 0` as without loss
of generality as we can always shift the data to accommodate any given
mean.


.. samples

.. image:: _static/Images/samples_from_prior.*
.. image:: _static/Images/samples_from_posterior.*

Samples from two Gaussian processes with the same
mean and covariance functions. Here :math:`\mathcal{X} = \mathbb{R}`\ .
The prior samples are taken from a
Gaussian process without any data and the posterior samples are taken
from a Gaussian process where the data are shown as black squares. The
black dotted line represents the mean of the process and the gray shaded
area covers twice the standard deviation at each input, :math:`x`\ . The
coloured lines are samples from the process, or more accurately, samples
at a finite number of inputs, :math:`x`\ , joined by lines.

The code to generate the above figures:

.. literalinclude:: _static/Code/figure_samples.py
    :language: python
    :lines: 5-


 





The covariance function, :math:`k`
----------------------------------

Assuming the mean function, :math:`\mu`\ , is 0 everywhere then our GP
is defined by two quantities: the data, :math:`\mathcal{D}`\ ; and its
covariance function (sometimes referred to as its *kernel*), :math:`k`\ .
The data is fixed so our modelling problem is exactly that of choosing a
suitable covariance function. Given different problems we certainly wish
to specify different priors over possible functions. Fortunately we have
available a large library of possible covariance functions each of which
represents a different prior on the space of functions.

.. covariance-examples

.. image:: _static/Images/covariance_function_se.*
.. image:: _static/Images/covariance_function_matern_52.*
.. image:: _static/Images/covariance_function_se_long_length.*
.. image:: _static/Images/covariance_function_matern_32.*
.. image:: _static/Images/covariance_function_rq.*
.. image:: _static/Images/covariance_function_periodic.*

Samples drawn
from GPs with the same data and different covariance functions. Typical
samples from the posterior of GPs with different covariance functions
have different characteristics. The periodic covariance function’s
primary characteristic is self explanatory. The other covariance
functions affect the smoothness of the samples in different ways.

The code to generate the above figures:

.. literalinclude:: _static/Code/figure_covariance_examples.py
    :language: python
    :lines: 5-



Combining covariance functions and noisy data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Furthermore the point-wise product and sum of covariance functions are
themselves covariance functions. In this way we can combine simple
covariance functions to represent more complicated beliefs we have about
our functions.

Normally we are modelling a system where we do not actually have access
to the target values, :math:`y`\ , but only noisy versions of them,
:math:`y+\epsilon`\ . If we assume :math:`\epsilon` has a Gaussian
distribution with variance :math:`\sigma_n^2` we can incorporate this
noise term into our covariance function. This requires that our noisy
GP’s covariance function, :math:`k_{\textrm{noise}}(x_1,x_2)` is aware
of whether :math:`x_1` and :math:`x_2` are the same input as we may have
two noisy measurements at the same point in :math:`\mathcal{X}`\ .

.. math:: k_{\textrm{noise}}(x_1,x_2) = k(x_1,x_2) + \delta(x_1=x_2) \sigma_n^2

.. image:: _static/Images/noise_low.*
.. image:: _static/Images/noise_mid.*
.. image:: _static/Images/noise_high.*

GP predictions with varying levels of
noise. The covariance function is a squared exponential with additive
noise of levels 0.0001, 0.1 and 1.

The code to generate the above figures:

.. literalinclude:: _static/Code/figure_noise.py
    :language: python
    :lines: 5-





Learning covariance function parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most of the commonly used covariance functions are parameterised. The
parameters
can be fixed if we are confident in our understanding of the problem.
Alternatively we can treat them as hyper-parameters in our Bayesian
inference task and optimise them through some technique such as maximum
likelihood estimation or conjugate gradient descent.

.. image:: _static/Images/learning_first_guess.*
.. image:: _static/Images/learning_learnt.*

The effects of learning covariance function
hyper-parameters. We see the predictions in the second figure seem
to fit the data more accurately.

The code to generate the above figures:

.. literalinclude:: _static/Code/figure_learning.py
    :language: python
    :lines: 5-



