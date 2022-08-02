#%%

import numpy as np
import pandas as pd
import random

from travel_costs import travel_costs_dict

#%%

all_trips = pd.read_csv('./data/full_sample_run/trips_filtered.csv')
all_people = pd.read_csv('./data/full_sample_run/people_filtered.csv')

#%%

all_people['income_per_capita'] = all_people.apply(lambda row: row['household_income'] / row['household_size']/10000, axis=1)
all_people['car_available'] = all_people.apply(lambda row: row['vehicles'] != 'zero', axis=1)
all_people['noncar_available'] = 1
all_people['employed'] = all_people.apply(lambda row: row['employment'] == 'employed', axis=1)


age_quantiles = pd.qcut(all_people['age'], 3, labels=range(3))

all_people[['age_youngest', 'age_oldest']] = pd.get_dummies(age_quantiles, prefix='age').drop(columns=['age_1'])

people_variables = all_people.drop(columns=['age', 'household_income', 'household_size', 'vehicles', 'employment', 'Unnamed: 0'])
people_variables.set_index('person_id', inplace=True, drop=True)

# %%

# Utility dictionaries and add integer encoding of mode choice
modes = ['PRIVATE_AUTO', 'CARPOOL', 'WALKING', 'PUBLIC_TRANSIT', 'ON_DEMAND_AUTO', 'SHARED_BIKE', 'BIKING']
modes_dict = {i: mode for i, mode in enumerate(modes)}
reverse_modes_dict = {mode: i for i, mode in enumerate(modes)}
all_trips['mode_choice_int'] = all_trips['mode'].map(reverse_modes_dict)

# In logit based model, commuting is implied by employed + rush hour
# all_trips['commuting'] = all_trips.apply(lambda row: (row['previous_activity_type'] == 'WORK') | (row['travel_purpose'] == 'WORK'), axis=1)
# Three highest traffic hours, could also add morning rush hour
all_trips['rush_hour'] = all_trips.apply(lambda row: row['start_local_hour'] in [15, 16, 17], axis=1)

# Sources: uber website, AAA, bluebikes site, MoCho
car_annual_cost_per_mile = 0.58
shared_auto_per_mile = 2.8
shared_auto_per_minute = 0.48
shared_auto_base_fare = 2.6
transit_cost = 2.25
bike_per_year = 220
shared_bike_cost = 3.25

# Remove trips finishing outside of the study area
all_trips = all_trips[all_trips.apply(lambda row: row['destination_bgrp'] in (all_trips.origin_bgrp.unique()), axis=1)]
# Format blockgroups to the required input for the travel costs function
blockgroups = all_trips[['origin_bgrp', 'origin_bgrp_lat', 'origin_bgrp_lng']].drop_duplicates()
bgrps = [{'bgrp_id': bgrp['origin_bgrp'], 'lat': bgrp['origin_bgrp_lat'], 'lng': bgrp['origin_bgrp_lng']} for _, bgrp in blockgroups.iterrows()]

# Retrieve travel costs
travel_costs = travel_costs_dict(bgrps)
#%%
# Add travel costs to table

n = all_trips.shape[0]

# Calculate trip vehicle times
all_trips['vt_PRIVATE_AUTO'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['drive'][row['destination_bgrp']]['vehicle_time'], axis=1)
all_trips['vt_CARPOOL'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['drive'][row['destination_bgrp']]['vehicle_time'], axis=1)
all_trips['vt_WALKING'] = 0
all_trips['vt_PUBLIC_TRANSIT'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['transit'][row['destination_bgrp']]['vehicle_time'], axis=1)
all_trips['vt_ON_DEMAND_AUTO'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['drive'][row['destination_bgrp']]['vehicle_time'], axis=1)
all_trips['vt_SHARED_BIKE'] = 0
all_trips['vt_BIKING'] = 0

# Calculate trip costs
all_trips['tc_PRIVATE_AUTO'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['drive'][row['destination_bgrp']]['distance'] * car_annual_cost_per_mile / 1600, axis=1)
all_trips['tc_CARPOOL'] = all_trips['tc_PRIVATE_AUTO'] / 2
all_trips['tc_WALKING'] = 0
all_trips['tc_PUBLIC_TRANSIT'] = transit_cost
all_trips['tc_ON_DEMAND_AUTO'] = all_trips.apply(lambda row: shared_auto_base_fare + travel_costs[row['origin_bgrp']]['drive'][row['destination_bgrp']]['distance'] * shared_auto_per_mile / 1600 + row['vt_ON_DEMAND_AUTO'] * shared_auto_per_minute, axis=1)
all_trips['tc_SHARED_BIKE'] = shared_bike_cost
all_trips['tc_BIKING'] = bike_per_year / (221 * 2) # commuting twice per working day

# Calculate waiting times
all_trips['wt_PUBLIC_TRANSIT'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['transit'][row['destination_bgrp']]['waiting_time'], axis=1)
all_trips['wt_CARPOOL'] = 1 + np.random.rand(n, 1)
all_trips['wt_ON_DEMAND_AUTO'] = 2.5 + np.random.rand(n, 1) # TODO: find better estimate for waiting time

# Calculate active times
all_trips['at_WALKING'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['walk'][row['destination_bgrp']]['active_time'], axis=1)
all_trips['at_PUBLIC_TRANSIT'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['transit'][row['destination_bgrp']]['active_time'], axis=1)
all_trips['at_SHARED_BIKE'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['bike'][row['destination_bgrp']]['active_time'], axis=1) + 2.5 # TODO: find better estimate for active time
all_trips['at_BIKING'] = all_trips.apply(lambda row: travel_costs[row['origin_bgrp']]['bike'][row['destination_bgrp']]['active_time'], axis=1)


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

# Add demographic information
all_variables = trips_variables.join(people_variables, on='person_id')
variables_no_na = all_variables.dropna(axis=0)

# Account for 'missing' people from people table
dropped_ratios = variables_no_na['mode'].value_counts()/all_variables['mode'].value_counts()
min_ratio = dropped_ratios.min()

variables_no_na = variables_no_na[
    variables_no_na.apply(lambda row: min_ratio/dropped_ratios[row['mode']] > random.random(), axis=1)
    ]

# Store the augmented data!
variables_no_na.to_csv('data/full_sample_run/variables_wide.csv')

# %%
