
import sys, logging, numpy
import rpy2.robjects as robjects
from ..dpmeans import dpmeans

def test_dp_means():
    logging.info(sys._getframe().f_code.co_name)
    r_faithful = robjects.r['faithful']
    faithful = numpy.array(r_faithful).T
    z = dpmeans(faithful, 25., progress_plots=True)
    logging.info(z)
    
        
    
    
    