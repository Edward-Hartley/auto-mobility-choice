#%%
import matplotlib.pyplot as plt
from data import *
from latLngMethods import get_distance, get_OSRM_distance

#%%

# Calculate ratio of bluebikes trips to cycle trips per trip distance
# Estimate using OSRM distance

all_trips = get_data('./data/big_query_trips.csv')
all_trips = all_trips[all_trips['mode'] == 'BIKING']

bluebikes_trips_augmented = get_data('./data/bluebikes/relevant_bluebikes_trips_augmented.csv')

# convert to km
# Checks reveal this is a poor way to predict replica distances from replica lat/lng
all_trips['distance_km'] = all_trips.apply(lambda x: x['distance_miles'] * 1.6, axis=1)
# all_trips['distance_straight'] = all_trips.apply(lambda x: get_distance(x['origin_lat'], x['origin_lng'], x['destination_lat'], x['destination_lng'])/1600, axis=1)
# all_trips['distance_estimated'] = all_trips.apply(lambda x: get_OSRM_distance('driving', x['origin_lat'], x['origin_lng'], x['destination_lat'], x['destination_lng'])/1000, axis=1)

# estimate cycled distances for bluebikes trips using parameters from the all_trips dataframe
bluebikes_trips_augmented['distance_estimated'] = bluebikes_trips_augmented.apply(lambda x: (get_OSRM_distance('biking', x['start station latitude'], x['start station longitude'], x['end station latitude'], x['end station longitude']) / 1000), axis=1)

cycle_total = 0
bluebikes_total = 0
boundaries = [0, 1, 2.5, 5, 10, 10000]

cycle_total = 0
bluebikes_total = 0

for i in range(1, len(boundaries)):
    all_trips_distance = all_trips[(all_trips['distance_km'] >= boundaries[i-1]) & (all_trips['distance_km'] < boundaries[i])]
    bluebikes_distance = bluebikes_trips_augmented[(bluebikes_trips_augmented['distance_estimated'] >= boundaries[i-1]) & (bluebikes_trips_augmented['distance_estimated'] < boundaries[i])]
    cycle_trips_count = all_trips_distance.shape[0]
    bluebikes_count = bluebikes_distance.shape[0]
    cycle_total += cycle_trips_count
    bluebikes_total += bluebikes_count
    print("Distance {} to {}\n\t total {} bluebikes {}, ratio {:.2f}, difference {}".format(boundaries[i-1], boundaries[i], cycle_trips_count, bluebikes_count, bluebikes_count / cycle_trips_count, bluebikes_count - cycle_trips_count))
print("Cycle trips total {}, bluebikes trips {}\n ratio {:.2f}, difference {}".format(cycle_total, bluebikes_total, bluebikes_total / cycle_total, bluebikes_total - cycle_total))

# %%

#%%

# Calculate ratio of bluebikes trips to cycle trips per trip distance
# Estimate using function of duration and stright-line distance

all_trips = get_data('./data/big_query_trips.csv')
all_trips = all_trips[all_trips['mode'] == 'BIKING']
bluebikes_trips_augmented = get_data('./data/bluebikes/relevant_bluebikes_trips_augmented.csv')

# check that replica data distances are calculated stright line
all_trips['distance_km'] = all_trips.apply(lambda x: x['distance_miles'] * 1.6, axis=1)
all_trips['distance_straight'] = all_trips.apply(lambda x: get_distance(x['origin_lat'], x['origin_lng'], x['destination_lat'], x['destination_lng'])/1600, axis=1)
all_trips['distance_estimated'] = all_trips.apply(lambda x: (0.0060 + x['duration_minutes'] * 0.1609 + x['distance_straight'] * 0.1791) * 1.6 , axis=1)

# estimate cycled distances for bluebikes trips using parameters from the all_trips dataframe
bluebikes_trips_augmented['distance_straight'] = bluebikes_trips_augmented.apply(lambda x: get_distance(x['start station latitude'], x['start station longitude'], x['end station latitude'], x['end station longitude'])/1600, axis=1)
bluebikes_trips_augmented['distance_estimated'] = bluebikes_trips_augmented.apply(lambda x: (0.0060 + (x['tripduration'] / 60) * 0.1609 + x['distance_straight'] * 0.1791) * 1.6 , axis=1)

cycle_total = 0
bluebikes_total = 0
boundaries = [0, 1, 2.5, 5, 10, 10000]
for i in range(1, len(boundaries)):
    all_trips_distance = all_trips[(all_trips['distance_km'] >= boundaries[i-1]) & (all_trips['distance_km'] < boundaries[i])]
    bluebikes_distance = bluebikes_trips_augmented[(bluebikes_trips_augmented['distance_estimated'] >= boundaries[i-1]) & (bluebikes_trips_augmented['distance_estimated'] < boundaries[i])]
    cycle_trips_count = all_trips_distance.shape[0]
    bluebikes_count = bluebikes_distance.shape[0]
    cycle_total += cycle_trips_count
    bluebikes_total += bluebikes_count
    print("Distance {} to {}\n\t total {} bluebikes {}, ratio {:.2f}, difference {}".format(boundaries[i-1], boundaries[i], cycle_trips_count, bluebikes_count, bluebikes_count / cycle_trips_count, bluebikes_count - cycle_trips_count))
print("Cycle trips total {}, bluebikes trips {}\n ratio {:.2f}, difference {}".format(cycle_total, bluebikes_total, bluebikes_total / cycle_total, bluebikes_total - cycle_total))


# import statsmodels.api as sm

# all_trips['distance_straight'] = all_trips.apply(lambda x: get_distance(x['origin_lat'], x['origin_lng'], x['destination_lat'], x['destination_lng'])/1600, axis=1)
# all_trips['distance_squared'] = all_trips.apply(lambda x: x['distance_straight'] ** 2, axis=1)
# X = all_trips[["duration_minutes", 'distance_straight']] ## X usually means our input variables (or independent variables)
# y = all_trips["distance_miles"] ## Y usually means our output/dependent variable
# X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# # Note the difference in argument order
# model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
# predictions = model.predict(X)

# # Print out the statistics
# model.summary()
# %%