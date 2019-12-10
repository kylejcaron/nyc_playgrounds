from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    
def parse_numbers(text):
    # converts 1 to 1st, 2 to 2nd, and so on
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
    words = text.split(' ')            
    parsed_words = [ordinal(int(word)) if is_number(word) else word for word in words]
    return ' '.join(parsed_words)

def extract_street_name(text):
    '''Extracts a single street name from the string of streetnames
    present in the Playground_Location column of the Playgrounds dataframe'''
    # r matches a street name
    r = '(\d+\s[\w]+\s[(st)|(ave))]+)|([\w]+\s(((st)|(ave))))'
    st_name = re.search(r, text.lower())
    if st_name:
        return st_name[0]
    else:
        return text

def search_street_name(street_name, geolocator, borough):
    parsed_street_name = parse_numbers(street_name) + ' ' + borough
    
    # Search geopy with st, borough, state
    zip_search_state = geolocator.geocode(parsed_street_name + ' NY')
    
    # Search geopy with st, borough. secondary state_search needed to make sure its in NY
    zip_search = geolocator.geocode(parsed_street_name)
        
    # If search result with state is not null
    if zip_search_state:
        # Search for presence of a zipcode the output
        regex_search = re.search('\d{5}', zip_search_state.raw['display_name'])
        if regex_search:
            return regex_search[0]
        # Sometimes search results won't list a zipcode, but will list coordinates,
        # Which can be used for a secondary search:
        else:
            lat = zip_search_state.raw['lat']
            lon = zip_search_state.raw['lon']
            coordinate_query = '{}, {}'.format(lat, lon)
            return geolocator.reverse(coordinate_query).raw['address']['postcode'][:5]
    
    # Sometimes search results will only appear without the state in the search query:
    elif zip_search: 
        # Since we're not searching with the state in the query, 
        # It's necessary to confirm that the search results are in NY
        state_confirmation = re.search('New York', zip_search.raw['display_name'])
        if state_confirmation:
            regex_search = re.search('\d{5}', zip_search.raw['display_name'])
            if regex_search:
                return regex_search[0]
            # Sometimes search results won't list a zipcode, but will list coordinates,
            # Which can be used for a secondary search
            else:
                lat = zip_search_state.raw['lat']
                lon = zip_search_state.raw['lon']
                coordinate_query = '{}, {}'.format(lat, lon)
                return geolocator.reverse(coordinate_query).raw['address']['postcode'][:5]
        else:
            return np.NaN
    else:
        return np.NaN
    
def test(row, geolocator):
    boroughs = {'X':'Bronx','B':'Brooklyn', 
                'M':'Manhattan', 'Q':'Queens',
                'R':'Staten Island'}
    
    borough_code = row['Prop_ID'][:1]
    borough = boroughs[borough_code]
    #print(row['Playground_Location'])
    return search_street_name(extract_street_name(row['Playground_Location']), geolocator, borough)



