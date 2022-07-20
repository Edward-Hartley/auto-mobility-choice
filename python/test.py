#%%
import matplotlib.pyplot as plt
from data import *
from latLngMethods import get_distance

blockgroups_queried = ['250173526001', '250173531022', '250173521021','250173523002', 
    '250173523001', '250173524001', '250173524002', '250173531012', 
    '250173531021', '250173531022', '250173531023']
blockgroups_queried_ints = [250173526001, 250173531022, 250173521021,250173523002, 
    250173523001, 250173524001, 250173524002, 250173531012, 
    250173531021, 250173531022, 250173531023]

#%%

big_query_trips = get_data('./data/big_query_trips.csv')
replica_dates = ['2022-06-21'] 
# depends on dates in query, but this was the closest date i could get
trips = big_query_trips

# #%%

# resident_trips = get_data(trip_path_residents)
# worker_trips = get_data(trip_path_workers)
# trips = combined_unique_rows(resident_trips, worker_trips, 'activity_id')

# print (trips.columns)
# print (unique_values(trips, 'mode'))
# start_dates = transform_data_new_column(trips, "start_time", 'start_date', lambda x: x.split(' ')[0])
# replica_dates = unique_values(start_dates, 'start_date')
# print(replica_dates)

#%%
bluebikes_trips = get_data(bluebikes_path)
# print(bluebikes_trips.columns)
print(bluebikes_trips.shape)

#%%

# strip down to only the relevant dates
bluebikes_dates = transform_data_new_column(bluebikes_trips, 'starttime', 'startdate', lambda x: x.split(' ')[0])
bluebikes_dates = transform_data_new_column(bluebikes_trips, 'starttime', 'starthour', lambda x: x.split(' ')[1].split(':')[0])

# bluebikes_dates = bluebikes_dates[bluebikes_dates.startdate.isin(replica_dates)]
# #%%

# # map origin and destinations to geo ids (blockgroups)
# bluebikes_dates['origin_bgrp'] = bluebikes_dates.apply(lambda x: latLngToGeoId.lat_lng_to_geo_id(x['start station latitude'], x['start station longitude']), axis=1)
# bluebikes_dates['destination_bgrp'] = bluebikes_dates.apply(lambda x: latLngToGeoId.lat_lng_to_geo_id(x['end station latitude'], x['end station longitude']), axis=1)
# store_data(bluebikes_dates, './data/bluebikes/trips_with_bgrp.csv')

#%%

# match origin and dst to blockgroups and filter out trips that are not in the blockgroups
bluebikes_relevant = bluebikes_dates[bluebikes_dates.origin_bgrp.isin(blockgroups_queried_ints) | bluebikes_dates.destination_bgrp.isin(blockgroups_queried_ints)]
print(unique_values(bluebikes_relevant, 'startdate'))
print(bluebikes_relevant.shape)

#%%

cycle_trips = trips.loc[trips['mode'] == 'BIKING']
cycle_trips_relevant = cycle_trips[cycle_trips.origin_bgrp.isin(blockgroups_queried_ints) | cycle_trips.destination_bgrp.isin(blockgroups_queried)]

print(cycle_trips_relevant.shape)
#%%
cycle_total = 0
bluebikes_total = 0
for i in range(len(blockgroups_queried)):
    cycle_trips_blockgroup = cycle_trips[(cycle_trips['origin_bgrp'] == blockgroups_queried_ints[i]) | (cycle_trips['destination_bgrp'] == blockgroups_queried[i])]
    bluebikes_blockgroup = bluebikes_relevant[(bluebikes_relevant['origin_bgrp'] == blockgroups_queried_ints[i]) | (bluebikes_relevant['destination_bgrp'] == blockgroups_queried_ints[i])]
    cycle_trips_count = cycle_trips_blockgroup.shape[0]
    bluebikes_count = bluebikes_blockgroup.shape[0]
    cycle_total += cycle_trips_count
    bluebikes_total += bluebikes_count
    print("Blockgroup {}, cycle trips total {}, bluebikes trips {}, ratio {:.2f}, difference {}".format(blockgroups_queried[i], cycle_trips_count, bluebikes_count, bluebikes_count / cycle_trips_count, bluebikes_count - cycle_trips_count))
print("Cycle trips total {}, bluebikes trips {}, ratio {:.2f}, difference {}".format(cycle_total, bluebikes_total, bluebikes_total / cycle_total, bluebikes_total - cycle_total))

#%%


# From stored data (has to be generated) replace selected cycle trips with bluebikes trips
###############################################################################
def replace_bike_trip(bluebikes_trip, all_trips):
    # find the matching trip in the all_trips dataframe
    for hrs in range(6):
        matching_trip = all_trips[all_trips['mode'] == 'BIKING']
        matching_trip = matching_trip[((matching_trip['start_local_hour'] + hrs) % 24 == bluebikes_trip['starthour'])]
        matching_trip = matching_trip[matching_trip.apply(lambda x: get_distance(bluebikes_trip['start station latitude'], bluebikes_trip['start station longitude'], x['origin_lat'], x['origin_lng']) < 600, axis=1)]
        matching_trip = matching_trip[matching_trip.apply(lambda x: get_distance(bluebikes_trip['end station latitude'], bluebikes_trip['end station longitude'], x['destination_lat'], x['destination_lng']) < 600, axis=1)]
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

    # replace the matching trip with the bluebikes trip
    # all_trips.loc[all_trips['activity_id'] == matching_trip['activity_id'].values[0], 'mode'] = 'BIKING'
    # return all_trips

all_trips = get_data('./data/big_query_trips.csv')
all_trips.set_index('activity_id', inplace=True, drop=False)
all_trips['distance_from_bluebikes'] = 10000
all_trips['distance_from_destination'] = 10000
all_trips['hours_from_bluebikes'] = 13
# filter out out_of_regio trips and convert to int64
all_trips = all_trips[all_trips['destination_bgrp'] != 'out_of_regio']
all_trips['destination_bgrp'] = all_trips['destination_bgrp'].astype(int)
# reduce to only the relevant mode
all_biking_trips = all_trips[all_trips['mode'] == 'BIKING']

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
data = cycle_trips[['origin_bgrp_lng', 'origin_bgrp_lat']].value_counts()
data = data.to_frame(name='count').reset_index()
data = data.loc[data['origin_bgrp_lat'] > 42.350]
data = data.loc[data['origin_bgrp_lat'] < 42.38]
data = data.loc[data['origin_bgrp_lng'] > -71.13]
data = data.loc[data['origin_bgrp_lng'] < -71.06]
print(data['count'].sum())
data['count_squared'] = data['count'] ** 0.5
data.plot.scatter(
    x='origin_bgrp_lng', y='origin_bgrp_lat', title='trip_origins',
    c='count', cmap='viridis', s='count',
    sharex=True
)

#%%

def plot_speeds(data, title):
    data = data.loc[data['duration_seconds'] != 0]

    data_with_speed = data.apply(lambda x: x['distance_meters'] / x['duration_seconds'], axis=1)

    trip_speeds_1 = data_with_speed.round(0).value_counts()
    trip_speeds_1 = trip_speeds_1[trip_speeds_1 > 10].sort_index()
    trip_speeds_1.plot(kind='bar', title=title)
    plt.show()

private_trips = trips.loc[trips['mode'] == 'PRIVATE_AUTO']

plot_speeds(private_trips, 'Private Speeds')

shared_trips = trips.loc[trips['mode'] == 'ON_DEMAND_AUTO']

plot_speeds(shared_trips, 'Shared Speeds')

plt.show()

# %%