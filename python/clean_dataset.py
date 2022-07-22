#%%
from data import *


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
