#
# Copyright John Reid 2006
#


from infpy.gp import *
import unittest



class GPTest( unittest.TestCase ):
    def setUp( self ):
        #
        # Set up some training data
        #
        self.training_points = [
                ( [ 1.05 ], 0 ),
                ( [ 2.0 ], 0.3 ),
                ( [ 2.5 ], 0.5 ),
                ( [ 2.7 ], 0.4 ),
                ( [ 3.0 ], 0 ),
                ( [ 4.0 ], 1.0 )
        ]
        # make the data's mean 0
        mean = sum(
                x[0] for (x, v) in self.training_points
        ) / len( self.training_points )
        for x, v in self.training_points: x[0] -= mean



        #
        # Create a gaussian process
        #
        X = [ x for x,v in self.training_points ]
        y = numpy.array( [ v for x,v in self.training_points ] )
        K = SumKernel(
                SquaredExponentialKernel( params = [ 1.0 ] ),
                noise_kernel( 0.3 ) )
        self.gp = GaussianProcess(
                X,
                y,
                K )

        #
        # Some parameters we can test the gradients at
        #
        self.test_kernel_parameters = [
                [ 0.53501393, 0.29317574],
        ]

    def tearDown( self ):
        self.gp = self.training_points = self.test_kernel_parameters = None

    def testSampleFrom( self ):
        gp_sample_from(
                self.gp,
                [
                        [ 2.78 ],
                        [ 3.78 ],
                        [ 4.78 ],
                        [ 5.78 ],
                        [ 6.78 ],
                ]
        )

    def testLearn( self ):
        gp_learn_hyperparameters( self.gp, disp = False )

    def testGradients( self ):
        for x0 in self.test_kernel_parameters:
            # print x0
            ll = GP_LL_fn( self.gp )
            ll_deriv = GP_LL_deriv_fn( self.gp, needs_to_set_params = True )
            infpy.check_gradients( ll, ll_deriv, x0 )

if __name__ == "__main__":
    unittest.main()
