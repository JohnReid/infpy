
D = 5
K = 3
W = 2

eta = [var('eta_%d' % d) for d in xrange(D)]

alpha = var('alpha')
n_d = var('n_d')
d_term = [
    eta[d] ** (alpha - 1)
    *
    (1 - eta[d]) ** n_d
    for d in xrange(D)
]
p_z_x = reduce(sage.symbolic.expression.Expression.__mul__, d_term)
log_p_z_x = log(p_z_x).simplify_log()


a, b, c = var('a b c')
assume(a > 0)
assume(b > 0)
assume(c > 0)

#
# Beta
#
x = var('x')
beta_dist = x**(a - 1) * (1 - x)**(b - 1)
c = integral(beta_dist, x, 0, 1)

#
# Dirichlet
#
x_1, x_2, x_3 = var('x_1, x_2, x_3')
dirichlet_dist = x_1**(a - 1) * x_2**(b - 1) * (1 - x_2 - x_1)**(c - 1)
e = integral(dirichlet_dist, x_1, 0, 1)
f = integral(e, x_2, 0, 1)

#
# Gamma
#
gamma_dist = x**(a - 1) * exp(-b * x)
d = integral(gamma_dist, x, 0, 1)
