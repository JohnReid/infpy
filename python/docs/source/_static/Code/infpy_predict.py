#
# Copyright John Reid 2012
#

def predict_values(K, file_tag, learn=False):
    """
    Create a GP with kernel K and predict values.
    Optionally learn K's hyperparameters if learn==True.
    """
    gp = infpy.gp.GaussianProcess(X, f, K)
    if learn:
        infpy.gp.gp_learn_hyperparameters(gp)
    pylab.figure()
    infpy.gp.gp_1D_predict(gp, 90, x_min - 10., x_max + 10.)
    save_fig(file_tag)
    pylab.close()

