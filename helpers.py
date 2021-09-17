##################################################################################
# General helper methods that could be used across multiple modules.
##################################################################################

def weight_limit(w, upper=1, lower=-1):
	"""
	Applies upper and lower limits to a weight.  
	"""
	return max(min(w, upper), lower)