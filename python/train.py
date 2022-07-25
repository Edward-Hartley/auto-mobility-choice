from collections import OrderedDict
from enum import IntEnum    # For recording the model specification 

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

all_trips = pd.read_csv('data/TODO') # TODO: replace path once file is created

# Create the list of individual specific variables
individual_variables = [
    'age', 'income_per_capita', 'employed', 
    'age_youngest', 'age_oldest',
    'commuting', 'rain_cover', 'rush_hour'
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
    'travel_time': add_mode_suffixes('tt_'),
    'travel_cost': add_mode_suffixes('travel_cost_'),
    'duration_variability': add_mode_suffixes('duration_variability_'),
    'waiting_time': {PUBLIC_TRANSIT: 'waiting_time_public_transit',
                     ON_DEMAND_AUTO: 'waiting_time_on_demand_auto'},
    'active_time': {WALKING: 'active_time_walking',
                    SHARED_BIKE: 'active_time_shared_bike',
                    BIKING: 'active_time_biking',
                    PUBLIC_TRANSIT: 'active_time_public_transit'},
    'rain_cover': {PRIVATE_AUTO: 'rain_cover',
                   CARPOOL: 'rain_cover',
                   PUBLIC_TRANSIT: 'rain_cover',
                   ON_DEMAND_AUTO: 'rain_cover'}
    }

availability_variables = {
    PRIVATE_AUTO: 'car_available',
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

obs_id_column = 'activity_id'
choice_column = 'mode_choice_int'

# Perform the conversion to long-format
all_trips_long = pl.convert_wide_to_long(all_trips, 
                                        individual_variables, 
                                        alt_varying_variables, 
                                        availability_variables, 
                                        obs_id_column, 
                                        choice_column,
                                        new_alt_id_name=custom_alt_id)

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

param_specification['intercept'] = [1, 2, 3, 4, 5, 6]
param_names['intercept'] = ['ASC Carpool', 'ASC Walk', 'ASC Public Transit', 'ASC On-Demand Auto', 'ASC Shared Bike', 'ASC Biking']

# Specify the coefficients for the basic variables
# Biking alternatives have the same coefficients for each population descriptor
param_specification['age_youngest'] = [0, 1, 2, 3, 4, [5, 6]]
param_names['age_youngest'] = ['Youngest Private Auto', 'Youngest Carpool', 'Youngest Walk', 'Youngest Public Transit', 'Youngest On-Demand Auto', 'Youngest Shared Bike', 'Youngest Biking']

param_specification['age_oldest'] = [0, 1, 2, 3, 4, [5, 6]]
param_names['age_oldest'] = ['Oldest Private Auto', 'Oldest Carpool', 'Oldest Walk', 'Oldest Public Transit', 'Oldest On-Demand Auto', 'Oldest Shared Bike', 'Oldest Biking']

param_specification['income_per_capita'] = [0, 1, 2, 3, 4, [5, 6]]
param_names['income_per_capita'] = ['Income Private Auto', 'Income Carpool', 'Income Walk', 'Income Public Transit', 'Income On-Demand Auto', 'Income Shared Bike', 'Income Biking']

param_specification['employed'] = [0, 1, 2, 3, 4, [5, 6]]
param_names['employed'] = ['Employed Private Auto', 'Employed Carpool', 'Employed Walk', 'Employed Public Transit', 'Employed On-Demand Auto', 'Employed Shared Bike', 'Employed Biking']

param_specification['rush_hour'] = [0, 1, 2, 3, 4, [5, 6]]
param_names['rush_hour'] = ['Rush Hour Private Auto', 'Rush Hour Carpool', 'Rush Hour Walk', 'Rush Hour Public Transit', 'Rush Hour On-Demand Auto', 'Rush Hour Shared Bike', 'Rush Hour Biking']

param_specification['commuting'] = [0, 1, 2, 3, 4, [5, 6]]
param_names['commuting'] = ['Commuting Private Auto', 'Commuting Carpool', 'Commuting Walk', 'Commuting Public Transit', 'Commuting On-Demand Auto', 'Commuting Shared Bike', 'Commuting Biking']

# Specify the coefficients for the trip statistics variables
param_specification['travel_time'] = [[0, 1, 2, 3, 4, 5, 6]]
param_names['travel_time'] = ['Travel Time']

param_specification['duration_variability'] = [[0, 1, 2, 3, 4, 5, 6]]
param_names['duration_variability'] = ['Duration Variability']

param_specification['cost'] = [[0, 1, 2, 3, 4, 5, 6]]
param_names['cost'] = ['Cost']

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



