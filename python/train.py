#%%

from collections import OrderedDict

import pandas as pd                    # For file input/output
import numpy as np                     # For vectorized math operations

import pylogit as pl                   # For MNL model estimation and


PRIVATE_AUTO = 0
CARPOOL = 1
WALKING = 2
PUBLIC_TRANSIT = 3
ON_DEMAND_AUTO = 4
SHARED_BIKE = 5
BIKING = 6

all_trips = pd.read_csv('data/full_sample_run/variables_wide.csv')
# print(all_trips['mode'].value_counts() / len(all_trips))
all_trips = all_trips.sample(frac = 0.05, random_state=42)
# print(all_trips['mode'].value_counts() / len(all_trips))

all_trips['income_per_capita'] = all_trips['income_per_capita']/1000

# Create the list of individual specific variables
individual_variables = [
    'income_per_capita', 'employed', 
    'age_youngest', 'age_oldest',
    'rush_hour'
    ]

def add_mode_suffixes(prefix):
    """
        Create a dict of mode suffixes for the given prefix.
    """
    suffixes = ['_PRIVATE_AUTO', '_CARPOOL', '_WALKING', '_PUBLIC_TRANSIT', '_ON_DEMAND_AUTO', '_SHARED_BIKE', '_BIKING']
    return {i: prefix + suffixes[i] for i in range(len(suffixes))}

# Specify the variables that vary across individuals and some or all alternatives
# The keys are the column names that will be used in the long format dataframe.
# The values are dictionaries whose key-value pairs are the alternative id and
# the column name of the corresponding column that encodes that variable for
# the given alternative.
alt_varying_variables = {
    'vehicle_time': add_mode_suffixes('vt'),
    'travel_cost': add_mode_suffixes('tc'),
    'waiting_time': {PUBLIC_TRANSIT: 'wt_PUBLIC_TRANSIT',
                     ON_DEMAND_AUTO: 'wt_ON_DEMAND_AUTO'},
    'active_time': {WALKING: 'at_WALKING',
                    SHARED_BIKE: 'at_SHARED_BIKE',
                    BIKING: 'at_BIKING',
                    PUBLIC_TRANSIT: 'at_PUBLIC_TRANSIT'},
    }

availability_variables = {
    PRIVATE_AUTO: 'noncar_available',
    CARPOOL: 'noncar_available',
    WALKING: 'noncar_available',
    PUBLIC_TRANSIT: 'noncar_available',
    ON_DEMAND_AUTO: 'noncar_available',
    SHARED_BIKE: 'noncar_available',
    BIKING: 'noncar_available'
    }

# The 'custom_alt_id' is the name of a column to be created in the long-format data
# It will identify the alternative associated with each row.
custom_alt_id = 'mode_id'

all_trips['custom_id'] = np.arange(all_trips.shape[0], dtype=int) + 1

obs_id_column = 'custom_id'
choice_column = 'mode_choice_int'

# Perform the conversion to long-format
all_trips_long = pl.convert_wide_to_long(all_trips, 
                                        individual_variables, 
                                        alt_varying_variables, 
                                        availability_variables, 
                                        obs_id_column, 
                                        choice_column,
                                        new_alt_id_name=custom_alt_id)

#%%

# Specify the nesting values
nest_membership = OrderedDict()
nest_membership["Non-biking"] = [0, 1, 2, 3, 4]
nest_membership["Biking"] = [5, 6]

# NOTE: - Specification and variable names must be ordered dictionaries.
#       - Keys should be variables within the long format dataframe.
#         The sole exception to this is the "intercept" key.
#       - For the specification dictionary, the values should be lists
#         of integers or lists of lists of integers. Within a list, 
#         or within the inner-most list, the integers should be the 
#         alternative ID's of the alternative whose utility specification 
#         the explanatory variable is entering. Lists of lists denote 
#         alternatives that will share a common coefficient for the variable
#         in question.

param_specification = OrderedDict()
param_names = OrderedDict()

param_specification['intercept'] = [1, 2, 3, 4, [5, 6]]
param_names['intercept'] = ['ASC Carpool', 'ASC Walk', 'ASC Public Transit', 'ASC On-Demand Auto', 'ASC Biking']

# Specify the coefficients for the basic variables
# Biking alternatives have the same coefficients for each population descriptor
# param_specification['age_youngest'] = [1, 2, 3, 4, [5, 6]]
# param_names['age_youngest'] = ['Youngest Carpool', 'Youngest Walk', 'Youngest Public Transit', 'Youngest On-Demand Auto', 'Youngest Biking']

# param_specification['age_oldest'] = [1, 2, 3, 4, [5, 6]]
# param_names['age_oldest'] = ['Oldest Carpool', 'Oldest Walk', 'Oldest Public Transit', 'Oldest On-Demand Auto', 'Oldest Biking']

param_specification['income_per_capita'] = [1, 2, 3, 4, 5, 6]
param_names['income_per_capita'] = ['Income Carpool', 'Income Walk', 'Income Public Transit', 'Income On-Demand Auto','Income Shared Bike',  'Income Biking']

# param_specification['employed'] = [1, 2, 3, 4, [5, 6]]
# param_names['employed'] = ['Employed Carpool', 'Employed Walk', 'Employed Public Transit', 'Employed On-Demand Auto', 'Employed Biking']

# param_specification['rush_hour'] = [1, 2, 3, 4, [5, 6]]
# param_names['rush_hour'] = ['Rush Hour Carpool', 'Rush Hour Walk', 'Rush Hour Public Transit', 'Rush Hour On-Demand Auto', 'Rush Hour Biking']

# param_specification['commuting'] = [1, 2, 3, 4, [5, 6]]
# param_names['commuting'] = ['Commuting Carpool', 'Commuting Walk', 'Commuting Public Transit', 'Commuting On-Demand Auto', 'Commuting Biking']

# Specify the coefficients for the trip statistics variables
param_specification['vehicle_time'] = [[0, 1, 2, 3, 4, 5, 6]]
param_names['vehicle_time'] = ['Vehicle Time']

param_specification['travel_cost'] = [[0, 1, 2, 3, 4, 5, 6]]
param_names['travel_cost'] = ['Cost']

param_specification['waiting_time'] = [[PUBLIC_TRANSIT, ON_DEMAND_AUTO]]
param_names['waiting_time'] = ['Waiting Time']

param_specification['active_time'] = [[WALKING, SHARED_BIKE, BIKING, PUBLIC_TRANSIT]]
param_names['active_time'] = ['Active Time']

# Suggested method for scaling the similarity parameter
def logit_scale(x):
    return np.log(x / (1 - x))

# Provide the module with the needed input arguments to create
# an instance of the MNL model class
nested = pl.create_choice_model(data=all_trips_long,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=param_specification,
                                        model_type="Nested Logit",
                                        names=param_names,
                                        nest_spec=nest_membership)

# Specify the nesting values
# Using reasonable values from pylogit example, should test using different init values
init_nests = np.array([40, logit_scale(2.05**-1)])

# Specify the initial values and method for the optimization.
numCoef=sum([len(param_specification[s]) for s in param_specification])
init_coefs = np.zeros(numCoef)

# Create a single array of the initial values
init_values = np.concatenate((init_nests, init_coefs), axis=0)

# Note that the first value, in the initial values, is constrained
# to remain constant through the estimation process. This is because
# the first nest in nest_spec is a 'degenerate' nest with only one
# alternative, and the nest parameter of degenerate nests is not
# identified.
nested.fit_mle(init_values, constrained_pos=[0])



# %%

nested.get_statsmodels_summary()
# %%
all_trips['mode'].value_counts()
# %%
nested.print_summaries()
# %%
