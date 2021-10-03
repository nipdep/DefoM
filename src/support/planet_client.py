# %%
from planet import api
import json 
import os 

# %%
PLANET_API_KEY = "bc4df0ffb0574dbbaac768ec377a695b"
item_type = "PSScene4Band"

client = api.ClientV1(PLANET_API_KEY)
# %%
def p(data):
    print(json.dumps(data, indent=2))
# %%
with open("../../data/data/map.json") as f:
    geom = json.loads(f.read())
# %%

from planet.api import filters

from datetime import datetime
start_date = datetime(year=2021, month=8, day=1)
end_date = datetime(year=2021, month=8, day=2)

date_filter = filters.date_range('acquired', gte=start_date, lte=end_date)
cloud_filter = filters.range_filter('cloud_cover', lte=0.1)
and_filter = filters.and_filter(date_filter, cloud_filter)
# %%
item_types = ["REOrthoTile", "PSOrthoTile"]
req = filters.build_search_request(and_filter, item_types)
# %%

res = client.quick_search(req)
# %%
for item in res.items_iter(4):
    print(item['id'], item['properties']['item_type'])
# %%
with open('../../data/data/results.json','w') as f:
    res.json_encode(f,1000)
# %%

print(item['id'])
# %%

assets = client.get_assets(item).get()
# %%

for asset in sorted(assets.keys()):
    print(asset)
# %%
activation = client.activate(assets['analytic'])
activation.response.status_code
# %%
