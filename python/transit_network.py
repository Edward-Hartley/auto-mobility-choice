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

# %%
def store_transit_walking_network(bbox):
    # Searching for GTFS feeds - find the feed for the city of interest
    gtfsfeeds.search(search_text='Massachusetts Bay Transportation Authority',
                    search_field=None,
                    match='contains',
                    add_feed=True)
    # Downloading GTFS data - uses global variable 'feeds'
    gtfsfeeds.download()
    # Load GTFS data into an UrbanAccess transit data object
    validation = True
    verbose = True

    remove_stops_outsidebbox = True
    append_definitions = True

    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=None,
                                            validation=validation,
                                            verbose=verbose,
                                            bbox=bbox,
                                            remove_stops_outsidebbox=remove_stops_outsidebbox,
                                            append_definitions=append_definitions)

    # loaded_feeds.stops.plot(kind='scatter', x='stop_lon', y='stop_lat', s=0.1)
    # Create a transit network
    ua.gtfs.network.create_transit_net(gtfsfeeds_dfs=loaded_feeds,
                                    day='thursday',
                                    timerange=['07:00:00', '10:00:00'],
                                    calendar_dates_lookup=None)
    # The UrbanAccess network object
    urbanaccess_net = ua.network.ua_network
    # urbanaccess_net.transit_nodes.plot(kind='scatter', x='x', y='y', s=0.1)
    # Download OSM data
    nodes, edges = osmnet.load.network_from_bbox(bbox=bbox, network_type='walk')
    # Create a pedestrian network
    ua.osm.network.create_osm_net(osm_edges=edges,
                                osm_nodes=nodes,
                                travel_speed_mph=3)
    # urbanaccess_net.osm_nodes.plot(kind='scatter', x='x', y='y', s=0.1)
    # Create an integrated transit and pedestrian network

    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                                headways=False)
    # Calculate headways for the network

    ua.gtfs.headways.headways(gtfsfeeds_df=loaded_feeds,
                            headway_timerange=['07:00:00','10:00:00'])
    # Integrate headways into the network
    ua.network.integrate_network(urbanaccess_network=urbanaccess_net,
                                headways=True,
                                urbanaccess_gtfsfeeds_df=loaded_feeds,
                                headway_statistic='mean')

    # Save the network to disk
    filename = 'transit_net.h5'

    ua.network.save_network(urbanaccess_network=urbanaccess_net,
                            filename=filename,
                            overwrite_key = True)
# %%

# ua.plot.plot_net(nodes=urbanaccess_net.net_nodes,
#                  edges=urbanaccess_net.net_edges,
#                  bbox=bbox,
#                  fig_height=30, margin=0.02,
#                  edge_color='#999999', edge_linewidth=1, edge_alpha=1,
#                  node_color='black', node_size=1.1, node_alpha=1, node_edgecolor='none', node_zorder=3, nodes_only=False)

#%%

# bbox for Blockgroups selected in sample - recalculate for new sample!
# make sure to change in driving network too
bbox = (-71.119, 42.3554, -71.0512, 42.3836)

# These may appear functional but they are not - the ua_network is global
store_transit_walking_network(bbox)



# %%
