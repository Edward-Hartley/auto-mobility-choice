#%%
from data import *

# Source
all_trips = get_data('./data/all_trips_bluebikes_replacements.csv')
all_trips.set_index('activity_id', inplace=True, drop=False)

# filter out out_of_regio trips and convert to int64
all_trips = all_trips[all_trips['destination_bgrp'] != 'out_of_regio']
all_trips['destination_bgrp'] = all_trips['destination_bgrp'].astype(int)

# Remove unknown travel modes
all_trips = all_trips[all_trips['mode'] != 'OTHER_TRAVEL_MODE']
all_trips = all_trips[all_trips['mode'] != 'COMMERCIAL']

# Store the cleaned dataset
store_data(all_trips, './data/all_trips_filtered.csv')

# %%

# source
workers = get_data('/home/dwarddd/MIT/auto-mobility-choice/python/data/my_own_samples/population_thursday_sep2019-nov2019_northeast_6filters_created07-25-2022.csv')
residents = get_data('/home/dwarddd/MIT/auto-mobility-choice/python/data/my_own_samples/population2.csv')
all_people = combined_unique_rows(workers, residents, 'person_id')

# all_people = get_data('./data/my_own_samples/combined.csv')

# remove columns that are not needed
all_people.drop(['BLOCKGROUP_home', 'BLOCKGROUP_work', 'lat_home', 'lng_home', 'lat_work', 'lng_work', 'household_id', 'resident_type', 'race_ethnicity', 'wfh'], axis=1, inplace=True)
all_people.dropna(inplace=True)

all_people = all_people[all_people['age'] != '\\N']
all_people['age'] = all_people['age'].astype(int)
all_people['household_income'] = all_people['household_income'].astype(int)
all_people['household_size'] = all_people['household_size'].astype(int)


store_data(all_people, "./data/all_people.csv")

# %%
