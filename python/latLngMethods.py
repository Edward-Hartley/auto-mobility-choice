import json
from time import sleep
import urllib.request
from geopy.distance import distance

def lat_lng_to_geo_id(lat, lng):
    """
    Converts a lat/lng pair to a geo id
    """
    url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&censusYear=2010&format=json'.format(lat, lng)
    response = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}))
    data = response.read()
    data = data.decode('utf-8')
    data = json.loads(data)
    return data['Block']['FIPS'][0:-3]

def get_distance(lat1, lng1, lat2, lng2):
    """
    Calculates the distance between two points on the Earth (specified in decimal degrees)
    Returns the distance in meters
    """
    return distance((lat1, lng1), (lat2, lng2)).km * 1000

def get_OSRM_distance(mode, startLat, startLon, endLat, endLon):
    """
    Returns the directions from start to end in the specified mode
    """
    strLL=str(startLon) + ','+str(startLat)+';'+str(endLon)+ ','+str(endLat)
    try:
        with urllib.request.urlopen('http://router.project-osrm.org/route/v1/'+str(mode)+'/'+strLL+'?overview=false') as url:
            data=json.loads(url.read().decode())
            #in meters and seconds
        return data['routes'][0]['distance']
        # if the call request is unsuccessful, wait and try again
    except:
        print("sleeping")
        sleep(0.1)
        return get_OSRM_distance(mode, startLat, startLon, endLat, endLon)
        