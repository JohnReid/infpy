from infpy.mixture import beta
from scipy.special import betaln
import pylab as pl, numpy as np, logging
from scipy.integrate import dblquad
import scipy.integrate as I

logging.basicConfig(level=logging.INFO)

max = 10.
num_points = 1000
step = max / num_points
a, b = np.ogrid[step:max:step,step:max:step]
betaln_a_b = betaln(a, b)
pl.subplots_adjust(wspace=.0, hspace=.5, top=.9, bottom=.06, right=1., left=0.)

tau = np.array([-1., -3.]) / 10.
nu = 0.4
g = tau[0] * (a-1.) + tau[1] * (b-1.) - nu * betaln_a_b
pl.clf()
pl.imshow(g, origin='lower', extent=[step,max,step,max])
title = '$\\tau = %s \\; \\nu = %.1f$' % (str(tau), nu)
logging.info('Title = %s', title)
pl.title(title)
pl.gca().get_xaxis().set_ticks_position('none')
pl.gca().get_yaxis().set_ticks_position('none')
#pl.colorbar()
pl.show()

A = beta.beta_conj_prior_log_partition(tau, nu)
logging.info("A(%s, %f) = %s", tau, nu, A)
exp_eta = beta.beta_conj_expected_eta(tau, nu, A)
logging.info("<eta> = %s", exp_eta)

def integrate_beta_conj_prior(tau, nu, A=None):
    if None == A:
        A = beta.beta_conj_prior_log_partition(tau, nu)
        
    def f(*eta):
        eta = np.array(eta)
        return np.exp(np.dot(tau, eta) - nu * betaln(*(eta+1.)) - A)
    
    result, abserr = dblquad(
        func=f, 
        a=-1.,
        b=np.infty,
        gfun=lambda x: -1,
        hfun=lambda x: np.infty,
    )
    return result, abserr

result, abserr = integrate_beta_conj_prior(tau, nu, A)
logging.info('Beta conjugate prior integrates to %f +/- %f', result, abserr)
    




