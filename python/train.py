#%%

from collections import OrderedDict

import pandas as pd                    # For file input/output
import numpy as np                     # For vectorized math operations

import pylogit as pl                   # For MNL model estimation and


PRIVATE_AUTO = 0
WALKING = 1
PUBLIC_TRANSIT = 2
ON_DEMAND_AUTO = 3
SHARED_BIKE = 4
BIKING = 5
modes = ['PRIVATE_AUTO', 'WALKING', 'PUBLIC_TRANSIT', 'ON_DEMAND_AUTO', 'SHARED_BIKE', 'BIKING']


all_trips = pd.read_csv('data/full_sample_run/variables_wide.csv')
# print(all_trips['mode'].value_counts() / len(all_trips))
all_trips = all_trips.sample(frac = 0.05, random_state=42)
# print(all_trips['mode'].value_counts() / len(all_trips))

all_trips['income_per_capita'] = all_trips['income_per_capita']/1000

# Create the list of individual specific variables
individual_variables = [
    'income_per_capita', 'employed', 
    'age_youngest', 'age_oldest',
    'rush_hour', 'commuting',
    ]

def add_mode_suffixes(prefix):
    """
        Create a dict of mode suffixes for the given prefix.
    """
    suffixes = ['_PRIVATE_AUTO', '_WALKING', '_PUBLIC_TRANSIT', '_ON_DEMAND_AUTO', '_SHARED_BIKE', '_BIKING']
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
                     ON_DEMAND_AUTO: 'wt_ON_DEMAND_AUTO',
                     },
    'active_time': {WALKING: 'at_WALKING',
                    SHARED_BIKE: 'at_SHARED_BIKE',
                    BIKING: 'at_BIKING',
                    PUBLIC_TRANSIT: 'at_PUBLIC_TRANSIT'},
    }

availability_variables = {
    PRIVATE_AUTO: 'noncar_available',
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

all_trips_long['travel_cost'] = all_trips_long['travel_cost']/100
# all_trips_long['total_time'] = all_trips_long['vehicle_time'] + all_trips_long['waiting_time'] + all_trips_long['active_time']
all_trips_long['rain_cover'] = all_trips_long.apply(lambda row: 1 if row['mode_id'] in [PRIVATE_AUTO, PUBLIC_TRANSIT, ON_DEMAND_AUTO] else 0, axis=1)

all_observation_ids = all_trips_long['custom_id'].unique()
np.random.seed(4)
np.random.shuffle(all_observation_ids)
estimate_size = 5000
predict_size = 30000
sample_observation_ids = all_observation_ids[:estimate_size]
prediction_observations_ids = all_observation_ids[estimate_size: estimate_size + predict_size]

prediction_trips = all_trips_long.loc[all_trips_long['custom_id'].isin(prediction_observations_ids)].copy()
all_trips_long= all_trips_long.loc[all_trips_long['custom_id'].isin(sample_observation_ids)].copy()

#%%

# Specify the nesting values
nest_membership = OrderedDict()
nest_membership["Non-biking"] = [0, 1, 2, 3]
nest_membership["Biking"] = [4, 5]

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

param_specification['intercept'] = [1, 2, 3, [4, 5]]
param_names['intercept'] = ['ASC Walk', 'ASC Public Transit', 'ASC On-Demand Auto', 'ASC Biking']

# Specify the coefficients for the basic variables
# Biking alternatives have the same coefficients for some population descriptors
# Others were found to be significant using hypothesis testing.
# param_specification['age_youngest'] = [1, 2, 3, 4, 5]
# param_names['age_youngest'] = ['Youngest Walk', 'Youngest Public Transit', 'Youngest On-Demand Auto', 'Youngest Shared Biking', 'Youngest Biking']

# param_specification['age_oldest'] = [1, 2, 3, [4, 5]]
# param_names['age_oldest'] = ['Oldest Walk', 'Oldest Public Transit', 'Oldest On-Demand Auto', 'Oldest Biking']

# param_specification['income_per_capita'] = [1, 2, 3, 4, 5]
# param_names['income_per_capita'] = ['Income Walk', 'Income Public Transit', 'Income On-Demand Auto', 'Income Shared Biking', 'Income Biking']

# Found to be insignificant using hypothesis testing.
# param_specification['employed'] = [1, 2, 3, 4, [5, 6]]
# param_names['employed'] = ['Employed Carpool', 'Employed Walk', 'Employed Public Transit', 'Employed On-Demand Auto', 'Employed Biking']

# param_specification['rush_hour'] = [1, 2, 3, 4, 5, 6]
# param_names['rush_hour'] = ['Rush Hour Carpool', 'Rush Hour Walk', 'Rush Hour Public Transit', 'Rush Hour On-Demand Auto', 'Rush Hour Shared Biking', 'Rush Hour Biking']

# Found to be insignificant using hypothesis testing.
# param_specification['commuting'] = [1, 2, 3, [4, 5]]
# param_names['commuting'] = ['Commuting Walk', 'Commuting Public Transit', 'Commuting On-Demand Auto', 'Commuting Biking']

# Specify the coefficients for the trip statistics variables
# param_specification['vehicle_time'] = [[0, 1, 2, 3, 4, 5]]
# param_names['vehicle_time'] = ['Vehicle Time']

param_specification['total_time'] = [[0, 1, 2, 3, 4, 5]]
param_names['total_time'] = ['Total Time']

# param_specification['travel_cost'] = [[0, 1, 2, 3, 4, 5]]
# param_names['travel_cost'] = ['Cost']

# param_specification['waiting_time'] = [[PUBLIC_TRANSIT, ON_DEMAND_AUTO]]
# param_names['waiting_time'] = ['Waiting Time']

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
nested.fit_mle(init_values, constrained_pos=[0], disp=True)



# %%

nested.get_statsmodels_summary()
# %%
all_trips['mode'].value_counts()
# %%
# Predict probabilities for each observation in prediction set

# prediction_trips['active_time'] = prediction_trips.apply(lambda row: row['active_time'] if row['mode_id'] != 4 else 0, axis=1)
# prediction_trips['travel_cost'] = prediction_trips.apply(lambda row: row['travel_cost'] if row['mode_id'] != 4 else 0, axis=1)


prediction_array = nested.predict(prediction_trips)
prediction_trips['probability'] = prediction_array

# Label rows for readability
prediction_trips['mode'] = prediction_trips.apply(lambda row: modes[int(row['mode_id'])], axis=1)
all_trips_long['mode'] = all_trips_long.apply(lambda row: modes[int(row['mode_id'])], axis=1)


# Select subset of population to analyse
# prediction_trips = prediction_trips[prediction_trips['employed'] == 0]

# Calculate predicted proportions
predicted_proportions = prediction_trips.groupby(['mode'])['probability'].sum().sort_values(ascending=False)
actual_proportions = prediction_trips.groupby(['mode'])['mode_choice_int'].sum().sort_values(ascending=False)
sample_proportions = all_trips_long.groupby(['mode'])['mode_choice_int'].sum().sort_values(ascending=False)

# Calculate percentage difference between predicted and actual proportions
difference = 0
for mode in modes:
    difference += abs(predicted_proportions[mode] - actual_proportions[mode])
print("percentage_difference: ", difference/predict_size)

sample_difference = 0
for mode in modes:
    sample_difference += abs(sample_proportions[mode] * (predict_size / estimate_size) - actual_proportions[mode])
print("sample_percentage_difference: ", sample_difference/predict_size)

# %%

print(actual_proportions)
print(predicted_proportions)

### Current model: 210 -> 240 shared bikes, bikes have reasonable similarity to each other, nice

# %%
new_prediction_trips = prediction_trips.copy()
new_prediction_trips['active_time'] = new_prediction_trips.apply(lambda row: row['active_time'] if row['mode_id'] != 4 else 0, axis=1)
new_prediction_trips['travel_cost'] = new_prediction_trips.apply(lambda row: row['travel_cost'] if row['mode_id'] != 4 else 0, axis=1)


prediction_array = nested.predict(new_prediction_trips)
new_prediction_trips['probability'] = prediction_array

# Label rows for readability
new_prediction_trips['mode'] = new_prediction_trips.apply(lambda row: modes[int(row['mode_id'])], axis=1)
all_trips_long['mode'] = all_trips_long.apply(lambda row: modes[int(row['mode_id'])], axis=1)


# Select subset of population to analyse
# new_prediction_trips = new_prediction_trips[new_prediction_trips['employed'] == 0]

# Calculate predicted proportions
predicted_proportions = new_prediction_trips.groupby(['mode'])['probability'].sum().sort_values(ascending=False)
actual_proportions = new_prediction_trips.groupby(['mode'])['mode_choice_int'].sum().sort_values(ascending=False)
sample_proportions = all_trips_long.groupby(['mode'])['mode_choice_int'].sum().sort_values(ascending=False)

# Calculate percentage difference between predicted and actual proportions
difference = 0
for mode in modes:
    difference += abs(predicted_proportions[mode] - actual_proportions[mode])
print("percentage_difference: ", difference/predict_size)

sample_difference = 0
for mode in modes:
    sample_difference += abs(sample_proportions[mode] * (predict_size / estimate_size) - actual_proportions[mode])
print("sample_percentage_difference: ", sample_difference/predict_size)

# %%

print(actual_proportions)
print(predicted_proportions)

### Current model: 210 -> 240 shared bikes, bikes have reasonable similarity to each other, nice

# %%
