#%%
#Importing the necessary packages

import osmnet

# Transit graph = pedestrian + transit
# Walking graph = predestrian
# Biking graph = driving graph = driving

import matplotlib
matplotlib.use('agg')  # allows notebook to be tested in Travis

import pandas as pd

import matplotlib.pyplot as plt
import time

import urbanaccess as ua
from urbanaccess.config import settings
from urbanaccess.gtfsfeeds import feeds
from urbanaccess import gtfsfeeds
from urbanaccess.gtfs.gtfsfeeds_dataframe import gtfsfeeds_dfs
from urbanaccess.network import ua_network, load_network

# Commented for linter
# %matplotlib inline

#%%
settings.to_dict()
feeds.to_dict()

# %%

gtfsfeeds.search(search_text='Massachusetts Bay Transportation Authority',
                 search_field=None,
                 match='contains',
                 add_feed=True)
# %%

gtfsfeeds.download()
# %%

validation = True
verbose = True
# bbox for Blockgroups selected in sample - recalculate for new sample!
bbox = (-73.3493, 41.2566, -68.679, 45.0668)
remove_stops_outsidebbox = True
append_definitions = True

loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=None,
                                           validation=validation,
                                           verbose=verbose,
                                           bbox=bbox,
                                           remove_stops_outsidebbox=remove_stops_outsidebbox,
                                           append_definitions=append_definitions)
# %%

# loaded_feeds.stops.plot(kind='scatter', x='stop_lon', y='stop_lat', s=0.1)
# %%

ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds,
                                   day='thursday',
                                   timerange=['07:00:00', '10:00:00'],
                                   calendar_dates_lookup=None)
# %%
urbanaccess_net = ua.network.ua_network
# urbanaccess_net.transit_nodes.plot(kind='scatter', x='x', y='y', s=0.1)
# %%

nodes, edges = osmnet.load.network_from_bbox(bbox=bbox)
# %%
