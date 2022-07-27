#%%
# Import packages

import numpy as np
import pandas as pd

import pandana
import urbanaccess as ua

pd.options.display.float_format = '{:.2f}'.format
# %%

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
    urbanaccess_net.net_edges["from_int"],
    urbanaccess_net.net_edges["to_int"],
    urbanaccess_net.net_edges[['vehicle_time', 'active_time', 'waiting_time']], 
    twoway=False)

walk_nodes = urbanaccess_net.net_nodes[urbanaccess_net.net_nodes['net_type'] == 'walk']

walking_biking_net = pandana.Network(
    walk_nodes["x"],
    walk_nodes["y"],
    walk_edges["from_int"],
    walk_edges["to_int"],
    walk_edges[['vehicle_time', 'active_time', 'waiting_time']],
    twoway=False)
# %%

urbanaccess_net = ua.network.load_network(filename='driving_net.h5')

driving_edges = urbanaccess_net.net_edges.copy()







# Output:
# (Bgrp, bgrp) -> (drive | walk | bike | transit) -> (wait_time, vehicle_time, active_time)
# %%
