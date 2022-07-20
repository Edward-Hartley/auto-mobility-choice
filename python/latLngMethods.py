import json
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