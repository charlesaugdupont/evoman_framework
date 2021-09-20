##################################################################################
# General helper methods that could be used across multiple modules.
##################################################################################

import numpy as np

def weight_limit(w, upper=1, lower=-1):
	"""
	Applies upper and lower limits to a weight.  
	"""
	return np.clip(w, lower, upper)