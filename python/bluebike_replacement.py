#%%
from data import *
from latLngMethods import get_distance

#%%

# From stored data (has to be generated) replace selected cycle trips with bluebikes trips
###############################################################################
def match_by_distance(bluebikes_trip, matching_trip, origin_threshold, destination_threshold, hrs, all_trips):
    matching_trip = matching_trip[matching_trip.apply(lambda x: get_distance(bluebikes_trip['start station latitude'], bluebikes_trip['start station longitude'], x['origin_lat'], x['origin_lng']) < origin_threshold, axis=1)]
    matching_trip = matching_trip[matching_trip.apply(lambda x: get_distance(bluebikes_trip['end station latitude'], bluebikes_trip['end station longitude'], x['destination_lat'], x['destination_lng']) < destination_threshold, axis=1)]
    if matching_trip.shape[0] != 0:
        trip_matched = matching_trip.iloc[0]
        all_trips.loc[trip_matched.activity_id, 'mode'] = 'SHARED_BIKE'
        all_trips.loc[trip_matched.activity_id, 'distance_from_bluebikes'] = (
            get_distance(bluebikes_trip['start station latitude'], bluebikes_trip['start station longitude'], trip_matched['origin_lat'], trip_matched['origin_lng'])
        )
        all_trips.loc[trip_matched.activity_id, 'distance_from_destination'] = (
            get_distance(bluebikes_trip['end station latitude'], bluebikes_trip['end station longitude'], trip_matched['destination_lat'], trip_matched['destination_lng'])
        )
        all_trips.loc[trip_matched.activity_id, 'hours_from_bluebikes'] = hrs
        return True
    return False

def replace_bike_trip(bluebikes_trip, all_trips):
    # find the matching trip in the all_trips dataframe
    for hrs in range(12):
        matching_trip = all_trips[all_trips['mode'] == 'BIKING']
        matching_trip = matching_trip[((matching_trip['start_local_hour'] + hrs) % 24 == bluebikes_trip['starthour'])]
        if match_by_distance(bluebikes_trip, matching_trip, 500, 1000, hrs, all_trips):
            return True
    for hrs in range(12):
        matching_trip = all_trips[all_trips['mode'] == 'BIKING']
        matching_trip = matching_trip[((matching_trip['start_local_hour'] + hrs) % 24 == bluebikes_trip['starthour'])]
        if match_by_distance(bluebikes_trip, matching_trip, 600, 10000, hrs, all_trips):
            return True
    return False

all_trips = get_data('./data/big_query_trips.csv')
all_trips.set_index('activity_id', inplace=True, drop=False)
all_trips['distance_from_bluebikes'] = 10000
all_trips['distance_from_destination'] = 10000
all_trips['hours_from_bluebikes'] = 13
# filter out out_of_regio trips and convert to int64
all_trips = all_trips[all_trips['destination_bgrp'] != 'out_of_regio']
all_trips['destination_bgrp'] = all_trips['destination_bgrp'].astype(int)
# reduce to only the relevant mode
all_other_trips, all_biking_trips = [x for _, x in all_trips.groupby(all_trips['mode'] == 'BIKING')]

#get bluebikes trips
bluebikes_trips_augmented = get_data('./data/bluebikes/relevant_bluebikes_trips_augmented.csv')

# baseline counts
print(all_trips['mode'].value_counts())

# See how many trips are missing from the all_trips dataframe
print(bluebikes_trips_augmented.shape[0])
print(bluebikes_trips_augmented.apply(lambda x: replace_bike_trip(x, all_biking_trips), axis=1).value_counts())
print(all_biking_trips['mode'].value_counts())

#analytics of performance of matching
distances = all_biking_trips[all_biking_trips['distance_from_bluebikes'] < 10000]
distances = distances['distance_from_bluebikes']
print("distances from origins:", distances.values.sum()/distances.shape[0])

distances = all_biking_trips[all_biking_trips['distance_from_destination'] < 10000]
distances = distances['distance_from_destination']
print("distances from destinations:", distances.values.sum()/distances.shape[0])

hours = all_biking_trips[all_biking_trips['hours_from_bluebikes'] < 13]
hours = hours['hours_from_bluebikes']
print(hours.values.sum()/hours.shape[0])

#%%

all_trips = pd.concat([all_other_trips, all_biking_trips])
print(all_trips['mode'].value_counts())

store_data(all_trips, './data/all_trips_bluebikes_replacements.csv')

# %%
