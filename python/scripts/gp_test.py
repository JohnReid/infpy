#
# Copyright John Reid 2006
#

import infpy, numpy, sys, math

#
# Set up some training data
#
training_points = [
        [ [ 1.05 ], 0 ],
        [ [ 1.08 ], 0.1 ],
        [ [ 2.0 ], 0.3 ],
        [ [ 2.5 ], 0.5 ],
        [ [ 2.7 ], 0.4 ],
        [ [ 3.0 ], 0 ],
        [ [ 4.0 ], 1.0 ]
]
# make the data's mean 0
mean = sum(
        v for (x, v) in training_points
) / len( training_points )
for tp in training_points: tp[1] -= mean


#
# Create a kernel
#
X = [ x for x,v in training_points ]
y = numpy.array( [ v for x,v in training_points ] )
K = (
        infpy.SquaredExponentialKernel(
                params = [ 0.4 ],
                priors = [ infpy.LogNormalDistribution( math.log( 0.45 ) ) ]
        )
        + infpy.noise_kernel(
                0.03,
                sigma_prior = infpy.LogNormalDistribution( math.log( 0.03 ) ) )
)
#K = infpy.noise_kernel( 0.1, prior = infpy.LogNormalDistribution( math.log( 0.03 ) ) )
#raise


#
# Create a gaussian process
#
gp = infpy.GaussianProcess(
        X,
        y,
        K )


#
# Sample from
#
def sample_from( display = True ):
    test_x = [ [x] for x in numpy.arrayrange(0,5,.05) ]
    test_y = infpy.gp_sample_from( gp, test_x )

    if display:
        from pylab import figure, plot, show, fill, title
        figure()
        infpy.gp_plot_prediction( gp.X, test_y )
        title('Log likelihood: %f\n%s' % ( gp.log_p_y_given_X, gp.k.params ) )
        show()

# sample_from( True )
# sys.exit()

#
# Predict value on test points
#
def predict_values( display = True ):
    test_x = [ [x] for x in numpy.arrayrange(0,5,.04) ]
    ( f_star_mean, V_f_star, log_p_y_given_X ) = gp.predict( test_x )

    if display:
        from pylab import figure, plot, show, fill, title
        figure()
        infpy.gp_plot_prediction( test_x, f_star_mean, V_f_star )
        plot(
                [ x[0] for (x, v) in training_points ],
                [ v for (x, v) in training_points ],
                'rs' )
        infpy.gp_title_and_show( gp )


#
# Learn hyperparameters
#
predict_values( display = True )
#sys.exit()
# print 'Params:', gp.k.get_parameters()
for initial_guess in [
#       [ 10.0 ],
#       [ 0.01 ],
        [ 0.01, 10.0 ],
        [ 0.01, 0.01 ],
        None,
]:
    print 'Learning from initial parameters:', initial_guess
    infpy.gp_learn_hyperparameters( gp, initial_guess )
    print dir(gp)
    print 'Learnt parameters: %s\nLL: %f' \
            % ( str( gp.k.params ), gp.LL )
    predict_values( display = True )
