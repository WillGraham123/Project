import emcee
import math

def prior_knowledge(p0):
    if 0 > p0[0] or p0[0] > 50:
        return -np.inf
    if -50 > p0[1] or p0[1] > 0:
        return -np.inf
    if -50 > p0[2] or p0[2] > 0:
        return -np.inf
    if -50 > p0[3] or p0[3] > 0:
        return -np.inf
    if -50 > p0[4] or p0[4] > 0:
        return -np.inf
    
    return 0.0
    
def prob_fn(p0):
    lp = prior_knowledge(p0)
    
    if not np.isfinite(lp):
        return -np.inf
    
    result = lp - math.exp(nn.predict(p0.reshape(1, 5)))
    
    return result