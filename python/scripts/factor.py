from sympy import *
from sympy.abc import mu, gamma, x
f = (- gamma * (x-mu)**2 - log(gamma) + log(2*pi)) / 2
g1 = Wild('g1', exclude=[gamma])
g2 = Wild('g2', exclude=[gamma])
g3 = Wild('g3', exclude=[gamma])
print f.expand().match(g1 * log(gamma) + g2 * gamma + g3)
print collect(f.expand(), x, evaluate=False)
print collect(f.expand(), [log(gamma), gamma], evaluate=False)
print collect(f.expand(), mu, evaluate=False)


gamma_P = Symbol('gamma_P')
gamma_Q = Symbol('gamma_Q')
mu_P = Symbol('mu_P')
mu_Q = Symbol('mu_Q')

H_Q = (log(2 * pi / gamma_Q) + 1) / 2
KL_Q_P = (log(gamma_Q/gamma_P) + gamma_P/gamma_Q + gamma_P * (mu_P-mu_Q)**2 - 1) / 2
exp_log_P_Q = -H_Q-KL_Q_P
