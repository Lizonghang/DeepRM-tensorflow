from pg import *
import parameters


params = parameters.Parameters()
params.compute_dependent_parameters()
launch(params)
