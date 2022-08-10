#%%

import numpy as np
import pandas as pd
import random

from travel_costs import travel_costs_dict

#%%

all_trips = pd.read_csv('./data/full_sample_run/trips_filtered.csv')
all_trips = all_trips.sample(frac = 0.2, random_state=42)
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
# Prepare for adding travel costs

all_trips['mode'] = all_trips.apply(lambda row: row['mode'] if row['mode'] != 'CARPOOL' else 'PRIVATE_AUTO', axis=1)

# Utility dictionaries and add integer encoding of mode choice
modes = ['PRIVATE_AUTO', 'WALKING', 'PUBLIC_TRANSIT', 'ON_DEMAND_AUTO', 'SHARED_BIKE', 'BIKING']
modes_dict = {i: mode for i, mode in enumerate(modes)}
reverse_modes_dict = {mode: i for i, mode in enumerate(modes)}
all_trips['mode_choice_int'] = all_trips['mode'].map(reverse_modes_dict)

# In logit based model, commuting is implied by employed + rush hour
all_trips['commuting'] = all_trips.apply(lambda row: (row['previous_activity_type'] == 'WORK') | (row['travel_purpose'] == 'WORK'), axis=1)
# Three highest traffic hours, could also add morning rush hour
all_trips['rush_hour'] = all_trips.apply(lambda row: row['start_local_hour'] in [15, 16, 17], axis=1)

# Sources: uber website, AAA, bluebikes site, MoCho
car_cost_per_mile = 0.15
car_cost_per_year = 3958
car_trips_per_year = 1865
shared_auto_per_mile = 2.8
shared_auto_per_minute = 0.48
shared_auto_base_fare = 2.6
transit_cost = 2.4
bike_per_year = 270
shared_bike_cost = 3.25
shared_bike_yearly_membership_cost = 110

# bbox for Blockgroups selected in sample - recalculate for new sample!
# make sure to change in driving network too
# enlarged to slightly beyond blockgroup boundaries to avoid isolated nodes
# bbox = (-71.13, 42.345, -71.04, 42.394)
# Bbox represents boston msa
bbox = (-71.79, 41.99, -70.67, 43.2566)

# Remove trips finishing outside of the study area
all_trips = all_trips[all_trips.apply(lambda row: 
    (row['destination_bgrp_lat'] < bbox[3]) &  
    (row['destination_bgrp_lat'] > bbox[1]) &
    (row['destination_bgrp_lng'] < bbox[2]) &
    (row['destination_bgrp_lng'] > bbox[0]), axis=1)]
# Format blockgroups to the required input for the travel costs function
blockgroups = all_trips[['destination_bgrp', 'destination_bgrp_lat', 'destination_bgrp_lng']].drop_duplicates()
bgrps = [{'bgrp_id': bgrp['destination_bgrp'], 'lat': bgrp['destination_bgrp_lat'], 'lng': bgrp['destination_bgrp_lng']} for _, bgrp in blockgroups.iterrows()]
ods = all_trips.apply(lambda row: str(row['origin_bgrp']) + ' ' + str(row['destination_bgrp']), axis=1)


#%%
# Retrieve travel costs
travel_costs_d, travel_costs_nd = travel_costs_dict(bgrps, ods)
#%%
# Add travel costs to table

n = all_trips.shape[0]

# Calculate trip vehicle times
all_trips['vt_PRIVATE_AUTO'] = all_trips.apply(lambda row: travel_costs_d[row['origin_bgrp']]['drive'][str(row['destination_bgrp'])]['vehicle_time'], axis=1)
# all_trips['vt_CARPOOL'] = all_trips.apply(lambda row: travel_costs_d[row['origin_bgrp']]['drive'][str(row['destination_bgrp'])]['vehicle_time'], axis=1)
all_trips['vt_WALKING'] = 0
all_trips['vt_PUBLIC_TRANSIT'] = all_trips.apply(lambda row: travel_costs_nd[row['origin_bgrp']]['transit'][str(row['destination_bgrp'])]['vehicle_time'], axis=1)
all_trips['vt_ON_DEMAND_AUTO'] = all_trips.apply(lambda row: travel_costs_d[row['origin_bgrp']]['drive'][str(row['destination_bgrp'])]['vehicle_time'], axis=1)
all_trips['vt_SHARED_BIKE'] = 0
all_trips['vt_BIKING'] = 0

# Calculate trip costs
all_trips['tc_PRIVATE_AUTO'] = all_trips.apply(lambda row: travel_costs_d[row['origin_bgrp']]['drive'][str(row['destination_bgrp'])]['distance'] * car_cost_per_mile / 1600 + car_cost_per_year / car_trips_per_year, axis=1)
# all_trips['tc_CARPOOL'] = all_trips['tc_PRIVATE_AUTO'] / 2
all_trips['tc_WALKING'] = 0
all_trips['tc_PUBLIC_TRANSIT'] = transit_cost
all_trips['tc_ON_DEMAND_AUTO'] = all_trips.apply(lambda row: shared_auto_base_fare + travel_costs_d[row['origin_bgrp']]['drive'][str(row['destination_bgrp'])]['distance'] * shared_auto_per_mile / 1600 + row['vt_ON_DEMAND_AUTO'] * shared_auto_per_minute, axis=1)
all_trips['tc_SHARED_BIKE'] = all_trips.apply(lambda row: shared_bike_yearly_membership_cost / (221 * 2) if row['commuting'] else shared_bike_cost, axis=1)
all_trips['tc_BIKING'] = bike_per_year / (221 * 2) # commuting twice per working day

# Calculate waiting times
all_trips['wt_PUBLIC_TRANSIT'] = all_trips.apply(lambda row: travel_costs_nd[row['origin_bgrp']]['transit'][str(row['destination_bgrp'])]['waiting_time'], axis=1)
# all_trips['wt_CARPOOL'] = 1 + np.random.rand(n, 1)
# all_trips['wt_ON_DEMAND_AUTO'] = all_trips['vt_PRIVATE_AUTO'] / 4
all_trips['wt_ON_DEMAND_AUTO'] = 4

# Calculate active times
all_trips['at_WALKING'] = all_trips.apply(lambda row: travel_costs_nd[row['origin_bgrp']]['walk'][str(row['destination_bgrp'])]['active_time'], axis=1)
all_trips['at_PUBLIC_TRANSIT'] = all_trips.apply(lambda row: travel_costs_nd[row['origin_bgrp']]['transit'][str(row['destination_bgrp'])]['active_time'], axis=1)
all_trips['at_SHARED_BIKE'] = all_trips.apply(lambda row: travel_costs_nd[row['origin_bgrp']]['bike'][str(row['destination_bgrp'])]['active_time'], axis=1) + 2.5 # TODO: find better estimate for active time
all_trips['at_BIKING'] = all_trips.apply(lambda row: travel_costs_nd[row['origin_bgrp']]['bike'][str(row['destination_bgrp'])]['active_time'], axis=1)


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
print(dropped_ratios)
variables_no_na = variables_no_na[
    variables_no_na.apply(lambda row: min_ratio/dropped_ratios[row['mode']] > random.random(), axis=1)
    ]

# Store the augmented data!
variables_no_na.to_csv('data/full_sample_run/variables_wide.csv')

# %%

# Format blockgroups to the required input for the travel costs function
blockgroups = all_trips[['destination_bgrp', 'destination_bgrp_lat', 'destination_bgrp_lng']].drop_duplicates()
bgrps = {bgrp['destination_bgrp']: {'lat': bgrp['destination_bgrp_lat'], 'lng': bgrp['destination_bgrp_lng']} for _, bgrp in blockgroups.iterrows()}

ods_df = all_trips[['origin_bgrp', 'destination_bgrp']].drop_duplicates()
ods = ods_df.groupby('origin_bgrp').apply(lambda x: list(x['destination_bgrp'])).to_dict()

key = 250250408013

distances = pd.DataFrame(map(lambda id: [bgrps[id]['lat'], bgrps[id]['lng'], travel_costs_nd[key]['drive'][str(id)]['distance']], ods[key]), ods[key])
distances.plot(kind='scatter', x=1, y=0, cmap='viridis', c=distances[2])
# %%
