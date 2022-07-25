#%%

import numpy as np
import pandas as pd

all_trips = pd.read_csv('./data/all_trips_filtered.csv')
all_people = pd.read_csv('./data/all_people.csv')

#%%

all_people['income_per_capita'] = all_people.apply(lambda row: row['household_income'] / row['household_size'], axis=1)
all_people['car_available'] = all_people.apply(lambda row: row['vehicles'] != 'zero', axis=1)
all_people['noncar_available'] = 1
all_people['employed'] = all_people.apply(lambda row: row['employment'] == 'employed', axis=1)


age_quantiles = pd.qcut(all_people['age'], 3, labels=range(3))

all_people[['age_youngest', 'age_oldest']] = pd.get_dummies(age_quantiles, prefix='age').drop(columns=['age_1'])

people_variables = all_people.drop(columns=['age', 'household_income', 'household_size', 'vehicles', 'employment', 'Unnamed: 0'])
people_variables.set_index('person_id', inplace=True, drop=False)

# %%

modes = ['PRIVATE_AUTO', 'CARPOOL', 'WALKING', 'PUBLIC_TRANSIT', 'ON_DEMAND_AUTO', 'SHARED_BIKE', 'BIKING']
modes_dict = {i: mode for i, mode in enumerate(modes)}
reverse_modes_dict = {mode: i for i, mode in enumerate(modes)}
cost_per_mile = [0, 1, 2, 3, 4, 5, 6]

walking_spd_mps = 4 / 3.6 # convert 4km/h to m/s
cycle_spd_mps = 12 / 3.6

all_trips['mode_choice_int'] = all_trips['mode'].map(reverse_modes_dict)

# %%

all_trips['commuting'] = all_trips.apply(lambda row: row['tour_type'] == 'COMMUTE', axis=1)
# When converting to long format only map rain_cover to appropriate modes
all_trips['rain_cover'] = 1
# Three highest traffic hours, could also add morning rush hour
all_trips['rush_hour'] = all_trips.apply(lambda row: row['start_local_hour'] in [15, 16, 17], axis=1)

# Calculate trip travel times, main todo is here!
all_trips['tt_PRIVATE_AUTO'] = all_trips.apply(lambda row: row['distance_miles'] * 1600 / walking_spd_mps)
all_trips['tt_WALK'] = all_trips.apply(lambda row: row['distance_miles'] * 1600 / walking_spd_mps)

# %%
