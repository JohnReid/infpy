
import infpy.mixture.beta; reload(infpy.mixture.beta)
import infpy.mixture.beta as B
import numpy as np, logging

logging.basicConfig(level=logging.DEBUG)

tau = np.array([-1.27766550e+02, -1.96779820e-02])
nu = 4.998996

eta = B.estimate_beta_parameters_newton(tau, nu)
logging.info(eta)

eta = B.estimate_beta_parameters(tau, nu)
logging.info(eta)

