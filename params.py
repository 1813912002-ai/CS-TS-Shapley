## params.py
## Date: 01/23/2022
## Initialize default parameters


class Parameters(object):
    def __init__(self):
        self.params = {
            # For TMC Shapley
            'tmc_iter':500,
            'tmc_thresh':0.001,
            # For CS Shapley
            'cs_iter':500,
            'cs_thresh':0.001,
            # For TS Shapley
            'ts_iter':500,
            # For Influence Function
            'if_iter':30,
            'second_order_grad':False,
            'for_high_value':True
            }
    
    def update(self, new_params, verbose=True):
        if verbose:
            print("Overload the model parameters with the user specified ones: {}".format(new_params))
        for (key, val) in new_params.items():
            try:
                self.params[key] = val
            except KeyError:
                raise KeyError("Undefined key {} with value {}".format(key))
        # return self.params


    def get_values(self):
        return self.params


    def print_values(self):
        print("The current hyper-parameter setting:")
        for (key, val) in self.params.items():
            print("\t{} : {}".format(key, val))
