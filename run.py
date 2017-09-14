from pg import *
import parameters


params = parameters.Parameters()
params.batch_size = 3
params.compute_dependent_parameters()
launch(params)
