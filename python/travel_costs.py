#%%
# Import packages

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

    #%%

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

    driving_edges.fillna(25, inplace=True)
    driving_edges['maxspeed'] = driving_edges.apply(lambda x: x['maxspeed'].split(' ')[0] if (x['maxspeed'] != 25) else 25, axis=1)
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

def sum_path_by_column(path, edges, column):
    time = 0
    for i in range(len(path) - 1):
        time += edges.loc[(edges['from'] == path[i]) & (edges['to'] == path[i+1]), column].values[0]
    return time


# test data
# replica_path = '/home/dwarddd/MIT/auto-mobility-choice/python/data/full_sample_run/trips_thursday_mar2021-may2021_northeast_28filters_created07-26-2022.csv'
# replica_trips = pd.read_csv(replica_path)
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
def gen_travel_costs_dict(bgrps):
    driving_net = gen_driving_net()
    transit_ped_net, walking_biking_net = transit_walking_biking_nets()
    #%%

    # !!! x=longitude, y=latitude
    bgrp_xs = pd.Series([bgrp['lng'] for bgrp in bgrps], [bgrp['bgrp_id'] for bgrp in bgrps])
    bgrp_ys = pd.Series([bgrp['lat'] for bgrp in bgrps], [bgrp['bgrp_id'] for bgrp in bgrps])
    bgrp_nodes_driving = driving_net.get_node_ids(bgrp_xs, bgrp_ys)
    bgrp_nodes_transit = transit_ped_net.get_node_ids(bgrp_xs, bgrp_ys)
    bgrp_nodes_walking_biking = walking_biking_net.get_node_ids(bgrp_xs, bgrp_ys)

    #%%

    travel_costs = {}
    switch = 3
    walk_to_bike_time_ratio = 4 # 3mph -> 12mph
    highest_vehicle = 0
    highest_active = 0
    highest_waiting = 0

    for bgrp in bgrps:
        bgrp_id = bgrp['bgrp_id']
        travel_costs[bgrp_id] = {}
        for mode in ['drive', 'walk', 'bike', 'transit']:
            travel_costs[bgrp_id][mode] = {}
            if mode == 'drive':
                # Lists to find paths from starting blockgroup to all others
                nodes_a = [bgrp_nodes_driving[bgrp_id]] * len(bgrp_nodes_driving)
                nodes_b = bgrp_nodes_driving.values
                # Calculate paths and store the vehicle times
                vehicle_times = pd.Series(driving_net.shortest_path_lengths(nodes_a, nodes_b, 'vehicle_time'), bgrp_nodes_driving.keys())
                # TODO: replace this with path sum over the shortest path in terms of vehicle time
                distance = pd.Series(driving_net.shortest_path_lengths(nodes_a, nodes_b, 'distance'), bgrp_nodes_driving.keys())
                # Remove outliers (for any unreachable nodes the time is set to a very high value)
                next_max = vehicle_times[vehicle_times != vehicle_times.max()].max()
                if next_max > highest_vehicle:
                    highest_vehicle = next_max
                elif next_max == 0:
                    next_max = highest_vehicle
                # Store the vehicle times and set the other costs to 0
                vehicle_times = vehicle_times.apply(lambda x: next_max if x == vehicle_times.max() else x)
                active_times = pd.Series([0] * len(bgrp_nodes_driving), bgrp_nodes_driving.keys())
                waiting_times = pd.Series([0] * len(bgrp_nodes_driving), bgrp_nodes_driving.keys())
            # Repeat above for walking and biking
            elif mode == 'walk':
                nodes_a = [bgrp_nodes_walking_biking[bgrp_id]] * len(bgrp_nodes_walking_biking)
                nodes_b = bgrp_nodes_walking_biking.values
                active_times = pd.Series(walking_biking_net.shortest_path_lengths(nodes_a, nodes_b, 'active_time'), bgrp_nodes_walking_biking.keys())
                next_max = active_times[active_times != active_times.max()].max()
                if next_max > highest_active:
                    highest_active = next_max
                elif next_max == 0:
                    next_max = highest_active
                active_times = active_times.apply(lambda x: next_max if x == active_times.max() else x)
                vehicle_times = pd.Series([0] * len(bgrp_nodes_driving), bgrp_nodes_driving.keys())
                waiting_times = pd.Series([0] * len(bgrp_nodes_driving), bgrp_nodes_driving.keys())

            elif mode == 'bike':
                nodes_a = [bgrp_nodes_walking_biking[bgrp_id]] * len(bgrp_nodes_walking_biking)
                nodes_b = bgrp_nodes_walking_biking.values
                active_times = pd.Series(walking_biking_net.shortest_path_lengths(nodes_a, nodes_b, 'active_time'), bgrp_nodes_walking_biking.keys())
                next_max = active_times[active_times != active_times.max()].max()
                if next_max > highest_active:
                    highest_active = next_max
                elif next_max == 0:
                    next_max = highest_active
                active_times = active_times.apply(lambda x: next_max if x == active_times.max() else x)
                active_times = active_times.apply(lambda x: x / walk_to_bike_time_ratio)
                vehicle_times = pd.Series([0] * len(bgrp_nodes_driving), bgrp_nodes_driving.keys())
                waiting_times = pd.Series([0] * len(bgrp_nodes_driving), bgrp_nodes_driving.keys())

            elif mode == 'transit':
                # Lists to find paths from starting blockgroup to all others
                nodes_a = [bgrp_nodes_transit[bgrp_id]] * len(bgrp_nodes_transit)
                nodes_b = bgrp_nodes_transit.values
                # Find shortest paths by total time
                shortest_paths = pd.Series(transit_ped_net.shortest_paths(nodes_a, nodes_b, 'weight'), bgrp_nodes_transit.keys())
                # Calculate the separate times for each type of time spent by summing each edge
                edges = transit_ped_net.edges_df
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
            for bgrp2 in bgrps:
                bgrp2_id = bgrp2['bgrp_id']
                travel_costs[bgrp_id][mode][bgrp2_id] = {}
                travel_costs[bgrp_id][mode][bgrp2_id]['waiting_time'] = waiting_times[bgrp2_id]
                travel_costs[bgrp_id][mode][bgrp2_id]['vehicle_time'] = vehicle_times[bgrp2_id]
                travel_costs[bgrp_id][mode][bgrp2_id]['active_time'] = active_times[bgrp2_id]
                if mode == 'drive':
                    travel_costs[bgrp_id][mode][bgrp2_id]['distance'] = distance[bgrp2_id]
    with open('./data/travel_costs.p', 'wb') as f:
        pickle.dump(travel_costs, f, protocol=pickle.HIGHEST_PROTOCOL) # save travel costs to file
    return travel_costs
# %%

def travel_costs_dict(bgrps):
    """
    Returns a dictionary of travel costs between blockgroups.
    """
    try :
        with open('./data/travel_costs.p', 'rb') as f:
            travel_costs = pickle.load(f)
    except FileNotFoundError:
        travel_costs = gen_travel_costs_dict(bgrps)
    return travel_costs
