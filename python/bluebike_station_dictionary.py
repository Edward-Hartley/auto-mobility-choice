#%%

import pandas as pd
from data import *
from latLngMethods import get_distance

#%%

def blockgroups_dict(replica_trips):
    blockgroups_origins = replica_trips[['origin_bgrp', 'origin_bgrp_lat', 'origin_bgrp_lng']].copy()
    blockgroups_destinations = replica_trips[['destination_bgrp', 'destination_bgrp_lat', 'destination_bgrp_lng']].copy()
    blockgroups_origins.rename(columns={'origin_bgrp': 'bgrp', 'origin_bgrp_lat': 'lat', 'origin_bgrp_lng': 'lng'}, inplace=True)
    blockgroups_destinations.rename(columns={'destination_bgrp': 'bgrp', 'destination_bgrp_lat': 'lat', 'destination_bgrp_lng': 'lng'}, inplace=True)
    blockgroups = pd.concat([blockgroups_origins, blockgroups_destinations], axis=0)
    blockgroups = blockgroups[blockgroups['bgrp'] != 'out_of_region']
    blockgroups['bgrp'] = blockgroups['bgrp'].astype(int)
    blockgroups = blockgroups.drop_duplicates()
    blockgroups = blockgroups.set_index('bgrp', drop=True).to_dict('index')
    return blockgroups

# %%

def stations_dict(bb_trips):

    # get all stations lat/lng
    stations_origins = bb_trips[['start station id', 'start station latitude', 'start station longitude']].copy()
    stations_destinations = bb_trips[['end station id', 'end station latitude', 'end station longitude']].copy()
    stations_origins.rename(columns={'start station id': 'station_id', 'start station latitude': 'lat', 'start station longitude': 'lng'}, inplace=True)
    stations_destinations.rename(columns={'end station id': 'station_id', 'end station latitude': 'lat', 'end station longitude': 'lng'}, inplace=True)
    stations = pd.concat([stations_origins, stations_destinations], axis=0)
    stations = stations.drop_duplicates()
    stations_dict = stations.set_index('station_id', drop=True).to_dict('index')
    return stations_dict

# %%

def get_stations_distances_dict(bb_trips, replica_trips):

    # Get dicts of blockgroups/stations and their lat/lng
    blockgroups = blockgroups_dict(replica_trips)
    stations = stations_dict(bb_trips)

    # For each station, calculate the distance to each blockgroup
    stations_distances_dict = {
        station_id: {
            bgrp: get_distance(station_loc['lat'], station_loc['lng'], bgrp_loc['lat'], bgrp_loc['lng']) for bgrp, bgrp_loc in blockgroups.items()
        } for station_id, station_loc in stations.items()
    }

    return stations_distances_dict

# %%
