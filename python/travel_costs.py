#%%
# Import packages

import re
import matplotlib
import numpy as np
import pandas as pd

import pandana
import urbanaccess as ua
import json, pickle

pd.options.display.float_format = '{:.4f}'.format

# %matplotlib inline

#%%
def transit_walking_biking_nets():

    urbanaccess_net = ua.network.load_network(filename='transit_net.h5')


    edges = urbanaccess_net.net_edges.copy()
    other_edges, transit_edges = [x for _, x in edges.groupby(edges['net_type'] == 'transit')]
    other_edges, walk_edges = [x for _, x in other_edges.groupby(other_edges['net_type'] == 'walk')]

    walk_edges['vehicle_time'] = 0
    walk_edges['active_time'] = walk_edges['weight']
    walk_edges['waiting_time'] = 0

    transit_edges['vehicle_time'] = transit_edges['weight']
    transit_edges['active_time'] = 0
    transit_edges['waiting_time'] = 0

    other_edges['vehicle_time'] = 0
    other_edges['active_time'] = 0
    other_edges['waiting_time'] = other_edges['weight']

    edges = pd.concat([walk_edges, transit_edges, other_edges])

    transit_ped_net = pandana.Network(
        urbanaccess_net.net_nodes["x"],
        urbanaccess_net.net_nodes["y"],
        edges["from_int"],
        edges["to_int"],
        edges[['vehicle_time', 'active_time', 'waiting_time', 'weight']], 
        twoway=False)

    walk_nodes = urbanaccess_net.net_nodes[urbanaccess_net.net_nodes['net_type'] == 'walk']

    walking_biking_net = pandana.Network(
        walk_nodes["x"],
        walk_nodes["y"],
        walk_edges["from_int"],
        walk_edges["to_int"],
        walk_edges[['vehicle_time', 'active_time', 'waiting_time']],
        twoway=False)

    return transit_ped_net, walking_biking_net
# %%
def gen_driving_net():
    urbanaccess_net = ua.network.load_network(filename='driving_net.h5')

    driving_edges = urbanaccess_net.net_edges.copy()
    driving_nodes = urbanaccess_net.net_nodes.copy()

    # This assumption is applied to a large proportion of the road map!
    # There are many NaNs
    driving_edges.fillna(30, inplace=True)
    driving_edges['maxspeed'] = driving_edges.apply(lambda x: re.sub('[^0-9]', '', str(x['maxspeed'])) if (re.sub('[^0-9]', '', str(x['maxspeed'])) != '') else 25, axis=1)
    driving_edges['maxspeed'] = driving_edges['maxspeed'].astype(int)

    driving_edges['vehicle_time'] = (driving_edges['distance'] / driving_edges['maxspeed']) * (60 / 1600) # mph -> m/min
    driving_edges['active_time'] = 0
    driving_edges['waiting_time'] = 0

    driving_net = pandana.Network(
        driving_nodes["x"],
        driving_nodes["y"],
        driving_edges["from"],
        driving_edges["to"],
        driving_edges[['vehicle_time', 'active_time', 'waiting_time', 'distance']],
        twoway=False)
    return driving_net
# %%
def sum_path_by_column(path, edges, column):
    time = 0
    for i in range(len(path) - 1):
        time += edges[str(path[i]) + '_' + str(path[i+1])][column]
    return time


# test data
replica_path = '/home/dwarddd/MIT/auto-mobility-choice/python/data/full_sample_run/trips_thursday_mar2021-may2021_northeast_28filters_created07-26-2022.csv'
replica_trips = pd.read_csv(replica_path)
# blockgroups = replica_trips[['origin_bgrp', 'origin_bgrp_lat', 'origin_bgrp_lng']].drop_duplicates()
# bgrps = [{'bgrp_id': bgrp['origin_bgrp'], 'lat': bgrp['origin_bgrp_lat'], 'lng': bgrp['origin_bgrp_lng']} for _, bgrp in blockgroups.iterrows()]

# %%

# TODO: instead of setting the unreachable nodes to max distance
# remove the nodes from the network and re-run the node assignment

# Calculates the travel costs for each mode for each pair of bgrps
# Parameters:
# bgrps: list of bgrp dicts where each dict has the following keys:
#   bgrp_id: string
#   lat: float
#   lng: float
# Returns:
# (Bgrp, (drive | walk | bike | transit), bgrp) -> (wait_time, vehicle_time, active_time)
def gen_travel_costs_dict(bgrps, ods):
    driving_net = gen_driving_net()
    #%%
    transit_ped_net, walking_biking_net = transit_walking_biking_nets()
    #%%

    # Change in line with transit_net and driving_net.py
    bbox = (-71.79, 41.99, -70.67, 43.2566)

    # Remove trips finishing outside of the study area
    all_trips = replica_trips[replica_trips.apply(lambda row: 
        (row['destination_bgrp_lat'] < bbox[3]) &  
        (row['destination_bgrp_lat'] > bbox[1]) &
        (row['destination_bgrp_lng'] < bbox[2]) &
        (row['destination_bgrp_lng'] > bbox[0]), axis=1)]
    # Format blockgroups to the required input for the travel costs function
    blockgroups = all_trips[['destination_bgrp', 'destination_bgrp_lat', 'destination_bgrp_lng']].drop_duplicates()
    bgrps = [{'bgrp_id': bgrp['destination_bgrp'], 'lat': bgrp['destination_bgrp_lat'], 'lng': bgrp['destination_bgrp_lng']} for _, bgrp in blockgroups.iterrows()]

    ods_df = all_trips[['origin_bgrp', 'destination_bgrp']].drop_duplicates()
    ods = ods_df.groupby('origin_bgrp').apply(lambda x: list(x['destination_bgrp'])).to_dict()

    # We have list of Origin, Destination
    # and id -> lat lng dict
    # we want oring, destination -> distance
    # we make list 

    # !!! x=longitude, y=latitude
    bgrp_xs = pd.Series([bgrp['lng'] for bgrp in bgrps], [bgrp['bgrp_id'] for bgrp in bgrps])
    bgrp_ys = pd.Series([bgrp['lat'] for bgrp in bgrps], [bgrp['bgrp_id'] for bgrp in bgrps])
    # bgrp_nodes_driving = driving_net.get_node_ids(bgrp_xs, bgrp_ys)
    bgrp_nodes_transit = transit_ped_net.get_node_ids(bgrp_xs, bgrp_ys)
    bgrp_nodes_walking_biking = walking_biking_net.get_node_ids(bgrp_xs, bgrp_ys)
    #%%
    edges = transit_ped_net.edges_df[['from', 'to', 'vehicle_time', 'active_time', 'waiting_time']]
    edges['fromto'] = edges['from'].astype(str) + '_' + edges['to'].astype(str)
    edges.drop_duplicates(inplace=True)
    edges.dropna(inplace=True)
    edges = edges.groupby('fromto').mean()
    edges.drop(columns=['from', 'to'], axis=1, inplace=True)
    edges = edges.to_dict(orient='index')

    #%%

    travel_costs = {}
    # switch = 3
    walk_to_bike_time_ratio = 4 # 3mph -> 12mph
    highest_vehicle = 0
    highest_active = 0
    highest_waiting = 0
    highest_distance = 0
    count = 0

    for bgrp_id in ods.keys():
        print(count)
        count += 1
        bgrp_id
        travel_costs[bgrp_id] = {}
        destinations = ods[bgrp_id]
        for mode in ['walk', 'bike', 'transit']: # in ['drive', 'walk', 'bike', 'transit']:
            travel_costs[bgrp_id][mode] = {}
            if mode == 'drive':
                # Lists to find paths from starting blockgroup to all others
                origin_node = bgrp_nodes_driving[str(bgrp_id)]
                nodes_a = [origin_node] * len(destinations)
                nodes_b = list(map(lambda id: bgrp_nodes_driving[id], destinations))
                # Calculate paths and store the vehicle times
                vehicle_times = pd.Series(driving_net.shortest_path_lengths(nodes_a, nodes_b, 'vehicle_time'), destinations)
                # TODO: replace this with path sum over the shortest path in terms of vehicle time
                distance = pd.Series(driving_net.shortest_path_lengths(nodes_a, nodes_b, 'distance'), destinations)
                # Remove outliers (for any unreachable nodes the time is set to a very high value)
                next_max = vehicle_times[vehicle_times != vehicle_times.max()].max()
                if next_max > highest_vehicle:
                    highest_vehicle = next_max
                elif next_max == 0:
                    next_max = highest_vehicle
                vehicle_times = vehicle_times.apply(lambda x: next_max if x == vehicle_times.max() else x)
                next_max = distance[distance != distance.max()].max()
                if next_max > highest_distance:
                    highest_distance = next_max
                elif next_max == 0:
                    next_max = highest_distance
                # Store the vehicle times and set the other costs to 0
                distance = distance.apply(lambda x: next_max if x == distance.max() else x)
                active_times = pd.Series([0] * len(destinations), destinations)
                waiting_times = pd.Series([0] * len(destinations), destinations)
            # Repeat above for walking and biking
            elif mode == 'walk':
                nodes_a = [bgrp_nodes_walking_biking[str(bgrp_id)]] * len(destinations)
                nodes_b = list(map(lambda id: bgrp_nodes_walking_biking[id], destinations))
                active_times = pd.Series(walking_biking_net.shortest_path_lengths(nodes_a, nodes_b, 'active_time'), destinations)
                next_max = active_times[active_times != active_times.max()].max()
                if next_max > highest_active:
                    highest_active = next_max
                elif next_max == 0:
                    next_max = highest_active
                active_times = active_times.apply(lambda x: next_max if x == active_times.max() else x)
                vehicle_times = pd.Series([0] * len(destinations), destinations)
                waiting_times = pd.Series([0] * len(destinations), destinations)

            elif mode == 'bike':
                nodes_a = [bgrp_nodes_walking_biking[str(bgrp_id)]] * len(destinations)
                nodes_b = list(map(lambda id: bgrp_nodes_walking_biking[id], destinations))
                active_times = pd.Series(walking_biking_net.shortest_path_lengths(nodes_a, nodes_b, 'active_time'), destinations)
                next_max = active_times[active_times != active_times.max()].max()
                if next_max > highest_active:
                    highest_active = next_max
                elif next_max == 0:
                    next_max = highest_active
                active_times = active_times.apply(lambda x: next_max if x == active_times.max() else x)
                active_times = active_times.apply(lambda x: x / walk_to_bike_time_ratio)
                vehicle_times = pd.Series([0] * len(destinations), destinations)
                waiting_times = pd.Series([0] * len(destinations), destinations)

            elif mode == 'transit':
                # Lists to find paths from starting blockgroup to all others
                nodes_a = [bgrp_nodes_transit[str(bgrp_id)]] * len(destinations)
                nodes_b = list(map(lambda id: bgrp_nodes_transit[id], destinations))
                # Find shortest paths by total time
                shortest_paths = pd.Series(transit_ped_net.shortest_paths(nodes_a, nodes_b, 'weight'), destinations)
                # Calculate the separate times for each type of time spent by summing each edge
                vehicle_times = shortest_paths.apply(lambda path: sum_path_by_column(path, edges, 'vehicle_time'))
                active_times = shortest_paths.apply(lambda path: sum_path_by_column(path, edges, 'active_time'))
                waiting_times = shortest_paths.apply(lambda path: sum_path_by_column(path, edges, 'waiting_time'))
                # Remove outliers (for any unreachable nodes the time is set to a very high value)
                next_max = vehicle_times[vehicle_times != vehicle_times.max()].max()
                if next_max > highest_vehicle:
                    highest_vehicle = next_max
                elif next_max == 0:
                    next_max = highest_vehicle
                vehicle_times = vehicle_times.apply(lambda x: next_max if x == vehicle_times.max() else x)
                next_max = active_times[active_times != active_times.max()].max()
                if next_max > highest_active:
                    highest_active = next_max
                elif next_max == 0:
                    next_max = highest_active
                active_times = active_times.apply(lambda x: next_max if x == active_times.max() else x)
                next_max = waiting_times[waiting_times != waiting_times.max()].max()
                if next_max > highest_waiting:
                    highest_waiting = next_max
                elif next_max == 0:
                    next_max = highest_waiting
                waiting_times = waiting_times.apply(lambda x: next_max if x == waiting_times.max() else x)
            # Fill in travel costs dictionary
            for bgrp2_id in destinations:
                travel_costs[bgrp_id][mode][bgrp2_id] = {}
                travel_costs[bgrp_id][mode][bgrp2_id]['waiting_time'] = waiting_times[bgrp2_id]
                travel_costs[bgrp_id][mode][bgrp2_id]['vehicle_time'] = vehicle_times[bgrp2_id]
                travel_costs[bgrp_id][mode][bgrp2_id]['active_time'] = active_times[bgrp2_id]
                if mode == 'drive':
                    travel_costs[bgrp_id][mode][bgrp2_id]['distance'] = distance[bgrp2_id]
    with open('./data/travel_costs_non_driving.p', 'wb') as f:
        pickle.dump(travel_costs, f, protocol=pickle.HIGHEST_PROTOCOL) # save travel costs to file
    return travel_costs
# %%

def travel_costs_dict(bgrps, ods):
    """
    Returns a dictionary of travel costs between blockgroups.
    """
    try :
        with open('./data/travel_costs_driving.p', 'rb') as f:
            travel_costs_driving = pickle.load(f)
    except FileNotFoundError:
        # travel_costs = gen_travel_costs_dict(bgrps, ods)
        print('driving costs not found')
    try :
        with open('./data/travel_costs_non_driving.p', 'rb') as f:
            travel_costs_non_driving = pickle.load(f)
    except FileNotFoundError:
        # travel_costs = gen_travel_costs_dict(bgrps, ods)
        print('non-driving costs not found')
    return travel_costs_driving, travel_costs_non_driving
