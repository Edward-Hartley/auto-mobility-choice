import json
import urllib.request

def lat_lng_to_geo_id(lat, lng):
    url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&censusYear=2010&format=json'.format(lat, lng)
    response = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}))
    data = response.read()
    data = data.decode('utf-8')
    data = json.loads(data)
    return data['Block']['FIPS'][0:-3]