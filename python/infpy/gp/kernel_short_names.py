#
# Copyright John Reid 2008
#

import infpy.gp

LN = infpy.LogNormalDistribution
Gamma = infpy.GammaDistribution
Constant = infpy.gp.ConstantKernel
Noise = infpy.gp.noise_kernel
SE = infpy.gp.SquaredExponentialKernel
RQ = infpy.gp.RationalQuadraticKernelAlphaParameterised
Periodic = infpy.gp.FixedPeriod1DKernel
Fix = infpy.gp.KernelParameterFixer
