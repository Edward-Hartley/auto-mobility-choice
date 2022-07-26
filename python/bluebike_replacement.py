#%%
# Imports and magic

from data import *
from latLngMethods import lat_lng_to_geo_id
from bluebike_station_dictionary import get_stations_distances_dict

bb_path = '/home/dwarddd/MIT/auto-mobility-choice/python/data/full_sample_run/202104-bluebikes-tripdata.csv'
replica_path = '/home/dwarddd/MIT/auto-mobility-choice/python/data/full_sample_run/trips_thursday_mar2021-may2021_northeast_28filters_created07-26-2022.csv'

relevant_dates = ['2021-04-01', '2021-04-08', '2021-04-15', '2021-04-22', '2021-04-29']

#%%
# define private functions

def make_station_to_bgrp_dict(bb_trips):
    stations = bb_trips[['start station id', 'start station longitude', 'start station latitude']]
    stations = stations.drop_duplicates()
    stations['bgrp'] = stations.apply(lambda x: lat_lng_to_geo_id(x['start station latitude'], x['start station longitude']), axis=1)
    return stations.set_index('start station id').to_dict()['bgrp']


def sample_bb_trips(bb_trips, replica_trips, relevant_dates):


    station_to_bgrp_dict = make_station_to_bgrp_dict(bb_trips)

    blockgroups_queried = replica_trips.origin_bgrp.unique()

    # Filter to relevant blockgroups
    bb_trips = bb_trips.loc[bb_trips.apply(lambda row: int(station_to_bgrp_dict[row['start station id']]) in blockgroups_queried, axis=1)]
    # Filter to relevant dates
    bb_trips['start_date'] = bb_trips.apply(lambda row: row['starttime'].split(' ')[0], axis=1)
    bb_trips = bb_trips[bb_trips['start_date'].isin(relevant_dates)]

    bb_trips = bb_trips.sample(frac = 1/len(relevant_dates))

    return bb_trips

def augment_bb_trips(bb_trips):
    bb_trips['starthour'] = bb_trips.apply(lambda x: x['starttime'].split(' ')[1].split(':')[0], axis=1)
    return bb_trips

#%%
# Sample trips from bb data

# get data
bb_trips = get_data(bb_path)
replica_trips = get_data(replica_path)

# reduce bb to relevant sample
bb_trips = sample_bb_trips(bb_trips, replica_trips, relevant_dates)
bb_trips = augment_bb_trips(bb_trips)

#%%
# Generate distances 'matrix' between stations and blockgroups

stations_distances_dict = get_stations_distances_dict(bb_trips, replica_trips)

#%%

# Replace selected cycle trips with bb trips
###############################################################################
def match_by_distance(bb_trip, matching_trip, hrs, all_trips, distances_dict, threshold):
    start_id = bb_trip['start station id']
    end_id = bb_trip['end station id']
    trip_matched = matching_trip.loc[matching_trip.apply(lambda x: distances_dict[start_id][x['origin_bgrp']] + distances_dict[end_id][x['destination_bgrp']], axis=1).idxmin()]
    
    distance_from_bb = distances_dict[start_id][trip_matched['origin_bgrp']]
    distance_from_destination = distances_dict[end_id][trip_matched['destination_bgrp']]

    if distance_from_bb + distance_from_destination > threshold:
        return False

    all_trips.loc[trip_matched.activity_id, 'mode'] = 'SHARED_BIKE'
    all_trips.loc[trip_matched.activity_id, 'distance_from_bb'] = (
        distance_from_bb
    )
    all_trips.loc[trip_matched.activity_id, 'distance_from_destination'] = (
        distance_from_destination
    )
    all_trips.loc[trip_matched.activity_id, 'hours_from_bb'] = hrs
    return True

def replace_bike_trip(bb_trip, all_trips, distances_dict):
    # find the matching trip in the all_trips dataframe
    for hrs in range(12):
        matching_trip = all_trips[all_trips['mode'] == 'BIKING']
        matching_trip = matching_trip[((matching_trip['start_local_hour'] + hrs) % 24 == int(bb_trip['starthour']))]
        if matching_trip.shape[0] > 0:
            if match_by_distance(bb_trip, matching_trip, hrs, all_trips, distances_dict, 900):
                return True
    for hrs in range(12):
        matching_trip = all_trips[all_trips['mode'] == 'BIKING']
        matching_trip = matching_trip[((matching_trip['start_local_hour'] + hrs) % 24 == int(bb_trip['starthour']))]
        if matching_trip.shape[0] > 0:
            if match_by_distance(bb_trip, matching_trip, hrs, all_trips, distances_dict, 1500):
                return True
    return False

# source
# all_trips = get_data('./data/big_query_trips.csv')
# origin_trips = get_data('/home/dwarddd/MIT/auto-mobility-choice/python/data/my_own_samples/origin.csv')
# destination_trips = get_data('/home/dwarddd/MIT/auto-mobility-choice/python/data/my_own_samples/destination.csv')
# all_trips = combined_unique_rows(origin_trips, destination_trips, 'activity_id')
all_trips = replica_trips.copy()

all_trips.set_index('activity_id', inplace=True, drop=False)
all_trips['distance_from_bb'] = 10000
all_trips['distance_from_destination'] = 10000
all_trips['hours_from_bb'] = 13
# filter out out_of_regio trips and convert to int64
all_trips = all_trips[all_trips['destination_bgrp'] != 'out_of_region']
all_trips['destination_bgrp'] = all_trips['destination_bgrp'].astype(int)
# reduce to only the relevant mode
all_other_trips, all_biking_trips = [x for _, x in all_trips.groupby(all_trips['mode'] == 'BIKING')]

#get bb trips
bb_trips_augmented = bb_trips.copy()

# baseline counts
print(all_trips['mode'].value_counts())

# See how many trips are missing from the all_trips dataframe
print(bb_trips_augmented.shape[0])
print(bb_trips_augmented.apply(lambda x: replace_bike_trip(x, all_biking_trips, stations_distances_dict), axis=1).value_counts())
print(all_biking_trips['mode'].value_counts())

#%%

#analytics of performance of matching
distances = all_biking_trips[all_biking_trips['distance_from_bb'] < 10000]
distances = distances['distance_from_bb']
print("distances from origins:", distances.values.sum()/distances.shape[0])

distances = all_biking_trips[all_biking_trips['distance_from_destination'] < 10000]
distances = distances['distance_from_destination']
print("distances from destinations:", distances.values.sum()/distances.shape[0])

hours = all_biking_trips[all_biking_trips['hours_from_bb'] < 13]
hours = hours['hours_from_bb']
print(hours.values.sum()/hours.shape[0])

#%%

all_trips = pd.concat([all_other_trips, all_biking_trips])
print(all_trips['mode'].value_counts())

store_data(all_trips, './data/full_sample_run/trips_bb_replacements.csv')

# %%
