#%%

import numpy as np
import pandas as pd
import random

#%%

all_trips = pd.read_csv('./data/full_sample_run/trips_filtered.csv')
all_people = pd.read_csv('./data/full_sample_run/people_filtered.csv')

#%%

all_people['income_per_capita'] = all_people.apply(lambda row: row['household_income'] / row['household_size'], axis=1)
all_people['car_available'] = all_people.apply(lambda row: row['vehicles'] != 'zero', axis=1)
all_people['noncar_available'] = 1
all_people['employed'] = all_people.apply(lambda row: row['employment'] == 'employed', axis=1)


age_quantiles = pd.qcut(all_people['age'], 3, labels=range(3))

all_people[['age_youngest', 'age_oldest']] = pd.get_dummies(age_quantiles, prefix='age').drop(columns=['age_1'])

people_variables = all_people.drop(columns=['age', 'household_income', 'household_size', 'vehicles', 'employment', 'Unnamed: 0'])
people_variables.set_index('person_id', inplace=True, drop=True)

# %%

modes = ['PRIVATE_AUTO', 'CARPOOL', 'WALKING', 'PUBLIC_TRANSIT', 'ON_DEMAND_AUTO', 'SHARED_BIKE', 'BIKING']
modes_dict = {i: mode for i, mode in enumerate(modes)}
reverse_modes_dict = {mode: i for i, mode in enumerate(modes)}
cost_per_mile = [0, 1, 2, 3, 4, 5, 6]

walking_spd_mps = 4 / 3.6 # convert 4km/h to m/s
driving_spd_mps = 50 / 3.6 
cycle_spd_mps = 12 / 3.6

all_trips['mode_choice_int'] = all_trips['mode'].map(reverse_modes_dict)

# %%

n = all_trips.shape[0]

# In logit based model, commuting is implied by employed + rush hour
# all_trips['commuting'] = all_trips.apply(lambda row: (row['previous_activity_type'] == 'WORK') | (row['travel_purpose'] == 'WORK'), axis=1)
# Three highest traffic hours, could also add morning rush hour
all_trips['rush_hour'] = all_trips.apply(lambda row: row['start_local_hour'] in [15, 16, 17], axis=1)

# Calculate trip vehicle times, main TODO is here!
all_trips['vt_PRIVATE_AUTO'] = all_trips.apply(lambda row: row['distance_meters'] / driving_spd_mps, axis=1)
all_trips['vt_CARPOOL'] = 0.00001*np.random.rand(n, 1)
all_trips['vt_WALKING'] = all_trips.apply(lambda row: row['distance_meters'] / walking_spd_mps, axis=1)
all_trips['vt_PUBLIC_TRANSIT'] = 0.00001*np.random.rand(n, 1)
all_trips['vt_ON_DEMAND_AUTO'] = 0.00001*np.random.rand(n, 1)
all_trips['vt_SHARED_BIKE'] = 0.00001*np.random.rand(n, 1)
all_trips['vt_BIKING'] = 0.00001*np.random.rand(n, 1)

# Calculate trip costs
all_trips['tc_PRIVATE_AUTO'] = 0.00001*np.random.rand(n, 1)
all_trips['tc_CARPOOL'] = 0.00001*np.random.rand(n, 1)
all_trips['tc_WALKING'] = 0.00001*np.random.rand(n, 1)
all_trips['tc_PUBLIC_TRANSIT'] = 0.00001*np.random.rand(n, 1)
all_trips['tc_ON_DEMAND_AUTO'] = 0.00001*np.random.rand(n, 1)
all_trips['tc_SHARED_BIKE'] = 0.00001*np.random.rand(n, 1)
all_trips['tc_BIKING'] = 0.00001*np.random.rand(n, 1)

# Calculate waiting times
all_trips['wt_PUBLIC_TRANSIT'] = 0.00001*np.random.rand(n, 1)
all_trips['wt_ON_DEMAND_AUTO'] = 2.5 + np.random.rand(n, 1) # TODO: find better estimate for waiting time

# Calculate active times
all_trips['at_WALKING'] = 0.00001*np.random.rand(n, 1)
all_trips['at_PUBLIC_TRANSIT'] = 0.00001*np.random.rand(n, 1)
all_trips['at_SHARED_BIKE'] = 0.00001*np.random.rand(n, 1)
all_trips['at_BIKING'] = 0.00001*np.random.rand(n, 1)


# %%

# drop no longer necessary columns
trips_variables = all_trips.drop(columns = [
    'activity_id.2', 'activity_id.1',
    'travel_purpose', 'origin_us_tract', 'destination_us_tract',
    'previous_activity_type', 'start_time', 'start_local_hour',
    'end_time', 'end_local_hour', 'duration_seconds', 'distance_meters',
    'origin_bgrp', 'origin_bgrp_lat', 'origin_bgrp_lng', 'destination_bgrp',
    'destination_bgrp_lat', 'destination_bgrp_lng',
    'distance_from_bb', 'distance_from_destination',
    'hours_from_bb', 'origin_land_use_l1',
    'origin_land_use_l2', 'destination_land_use_l1',
    'destination_land_use_l2', 'origin_building_use_l1',
    'origin_building_use_l2', 'destination_building_use_l1',
    'destination_building_use_l2', 'vehicle_type', 'vehicle_fuel_type',
    'vehicle_fuel_technology',
    ])
trips_variables.set_index('activity_id', inplace=True, drop=False)

# %%

all_variables = trips_variables.join(people_variables, on='person_id')
variables_no_na = all_variables.dropna(axis=0)

dropped_ratios = variables_no_na['mode'].value_counts()/all_variables['mode'].value_counts()
min_ratio = dropped_ratios.min()

variables_no_na = variables_no_na[
    variables_no_na.apply(lambda row: min_ratio/dropped_ratios[row['mode']] > random.random(), axis=1)
    ]

variables_no_na.to_csv('data/full_sample_run/variables_wide.csv')

# %%
