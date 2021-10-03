# %%
import os
import json
import requests
from requests.auth import HTTPBasicAuth

# %%
PLANET_API_KEY = "bc4df0ffb0574dbbaac768ec377a695b"
item_type = "PSScene4Band"
# %%

geojson_geometry = {
    "type": "Polygon",
    "coordinates": [
        [
        [
            80.92477798461914,
            7.858280356069636
        ],
        [
            80.94846725463867,
            7.901980735249777
        ],
        [
            80.93439102172852,
            7.948396693361106
        ],
        [
            80.91911315917969,
            7.971517704409957
        ],
        [
            80.89096069335936,
            7.942786251482477
        ],
        [
            80.89284896850586,
            7.8953494573016005
        ],
        [
            80.90263366699219,
            7.858790503815376
        ],
        [
            80.92477798461914,
            7.858280356069636
        ]
        ]
    ]
}

## define AOI region of the forest for request for satellite feed
geometry_filter = {
    "type":"GeometryFilter",
    "field_name":"geometry",
    "config": geojson_geometry
}

## define images acquired with in a date range
date_range_filter = {
    "type": "DateRangeFilter",
    "field_name": "acquired",
    "config": {
        "gte" : "2021-08-23T00:00:00.000Z",
        "lte" : "2021-08-24T00:00:00.000Z"
    }    
}

## define cloud cover precentage
cloud_cover_filter = {
    "type": "RangeFilter",
    "field_name": "cloud_cover",
    "config": {
        "lte": 0.8
    }
}

## define satellite feed standard
stand_filter = {
    "type":"StringInFilter",
    "field_name":"quality_category",
    "config": ["standard"]
}

zoom_level_filter = {
   "type":"NumberInFilter",
   "field_name":"gsd",
   "config":[
        3
   ]
}

## composite filter
composite_filter = {
    "type" : "AndFilter",
    "config" : [
        geometry_filter,
        date_range_filter,
        cloud_cover_filter,
        # stand_filter,
        # zoom_level_filter
    ]
}

## API request object
search_request = {
    "item_types" : [item_type],
    "filter" : composite_filter
}

# fire off the POST request
search_result = \
  requests.post(
    'https://api.planet.com/data/v1/quick-search',
    auth=HTTPBasicAuth(PLANET_API_KEY, ''),
    json=search_request)

print(json.dumps(search_result.json(), indent=2))
# %%
# extract image IDs only
image_ids = [feature['id'] for feature in search_result.json()['features']]
print(image_ids)
# %%
id0 = image_ids[0]
id0_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, id0)

# Returns JSON metadata for assets in this ID. Learn more: planet.com/docs/reference/data-api/items-assets/#asset
result = \
  requests.get(
    id0_url,
    auth=HTTPBasicAuth(PLANET_API_KEY, '')
  )

# List of asset types available for this particular satellite image
print(result.json().keys())
# %%

# This is "inactive" if the "analytic" asset has not yet been activated; otherwise 'active'
print(result.json()['analytic']['status'])
# %%

# Parse out useful links
links = result.json()[u"analytic"]["_links"]
self_link = links["_self"]
activation_link = links["activate"]

# Request activation of the 'analytic' asset:
activate_result = \
  requests.get(
    activation_link,
    auth=HTTPBasicAuth(PLANET_API_KEY, '')
  )
# %%
activation_status_result = \
  requests.get(
    self_link,
    auth=HTTPBasicAuth(PLANET_API_KEY, '')
  )
    
print(activation_status_result.json()["status"])
# %%
download_link = activation_status_result.json()["location"]
print(download_link)
# %%
