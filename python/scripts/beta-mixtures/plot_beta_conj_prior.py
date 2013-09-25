#
# Copyright John Reid 2011
#

"""
Visually examine the intractable beta conjugate prior.
"""

import pylab as pl, logging
from infpy.mixture import beta
from scipy.special import betaln

logging.basicConfig(level=logging.INFO)



def explore_beta_conj_prior():
    max = 10.
    num_points = 100
    step = max / num_points
    possible_taus = (
        -  .1 * np.ones(2),
        - 1.  * np.ones(2),
        np.array([-1., -3]),
        -10.  * np.ones(2),
    )
    possible_nus = (.1, 1., 10.)
    a, b = np.ogrid[step:max:step,step:max:step]
    betaln_a_b = betaln(a, b)
    pl.subplots_adjust(wspace=.0, hspace=.1, top=.9, bottom=.05, left=.07, right=1.)
    for t, tau in enumerate(possible_taus):
        for n, nu in enumerate(possible_nus):
            g = tau[0] * (a-1.) + tau[1] * (b-1.) - nu * betaln_a_b
            subplot = pl.subplot(len(possible_nus), len(possible_taus), n * len(possible_taus) + t + 1)
            pl.imshow(g, origin='lower', extent=[step,max,step,max])
            if 0 == n:
                pl.title('$\\tau = %s$' % tau)
            if n != len(possible_nus) - 1:
                pl.gca().xaxis.set_ticks([])
            if 0 == t:
                pl.gca().set_ylabel('$\\nu = %.1f$' % nu)
            else:
                pl.gca().yaxis.set_ticks([])
            subplot.get_xaxis().set_ticks_position('none')
            subplot.get_yaxis().set_ticks_position('none')
            #pl.colorbar()


    
pl.rcParams['font.size'] = 8
pl.figure(figsize=(4.652, 3.2))
explore_beta_conj_prior()
pl.savefig('beta-conj-prior.eps')
pl.savefig('beta-conj-prior.png')


