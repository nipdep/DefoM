
# %%
from sentinelhub import SHConfig
from utils import plot_image

config = SHConfig()
config.instance_id = '464e0857-a72d-4c5d-b0d5-a96b724301d7'
config.sh_client_id = '2a713948-9158-4185-85df-4f0d271b644d'
config.sh_client_secret = ":%lG1hmr?J#nkvmjq/9.(K*Tz[q:7ES-kN>{4A}P"
config.save()
# %%
if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
# %%
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, DownloadRequest

from utils import plot_image
# %%
betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]
resolution = 60
betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)

print(f'Image shape at {resolution} m resolution: {betsiboka_size} pixels')

# %%
evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=('2020-06-12', '2020-06-13'),
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config
)
# %%
true_color_imgs = request_true_color.get_data()
# %%
print(f'Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.')
print(f'Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}')
# %%
image = true_color_imgs[0]
print(f'Image type: {image.dtype}')

# plot function
# factor 1/255 to scale between 0-1
# factor 3.5 to increase brightness
plot_image(image, factor=3.5/255, clip_range=(0,1))
# %% [markdown]
#polygon batch requesting

# %%

import os
import datetime as dt

import geopandas as gpd

from sentinelhub import SentinelHubBatch, SentinelHubRequest, Geometry, CRS, DataCollection, \
    MimeType, SHConfig, bbox_to_dimensions

geo_path = '../../data/data/map.geojson'

# %%
area_gdf = gpd.read_file(geo_path)
# Geometry of an entire area
full_geometry = Geometry(area_gdf.geometry.values[0], crs=CRS.WGS84)

# %%
area_gdf.plot();
# %%
evalscript_true_color = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

request = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=('2021-06-01', '2021-06-02'),
            mosaicking_order='leastCC'
        )
    ],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.PNG)
    ],
    geometry=full_geometry,
    size=(512, 512),
    config=config
)

image = request.get_data()[0]

plot_image(image, factor=3.5/255, clip_range=(0,1))
# %% [markdown]
## large area splitting

# %%
import itertools

import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon, MultiLineString

from sentinelhub import BBoxSplitter, OsmSplitter, TileSplitter, CustomGridSplitter, UtmZoneSplitter, UtmGridSplitter
from sentinelhub import BBox, read_data, CRS, DataCollection

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon

# %%
INPUT_FILE = '../../data/data/map.json'
geo_json = read_data(INPUT_FILE)
# %%
forest_area = shape(geo_json['features'][0]['geometry'])
# %%
def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False):
    area_bbox = splitter.get_area_bbox()
    minx, miny, maxx, maxy = area_bbox
    lng, lat = area_bbox.middle
    w, h = maxx - minx, maxy - miny
    minx = minx - area_buffer * w
    miny = miny - area_buffer * h
    maxx = maxx + area_buffer * w
    maxy = maxy + area_buffer * h

    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    base_map = Basemap(projection='mill', lat_0=lat, lon_0=lng, llcrnrlon=minx, llcrnrlat=miny,
                       urcrnrlon=maxx, urcrnrlat=maxy, resolution='l', epsg=4326)
    base_map.drawcoastlines(color=(0, 0, 0, 0))

    area_shape = splitter.get_area_shape()
    if isinstance(area_shape, Polygon):
        area_shape = [area_shape]
    for polygon in area_shape:
        if isinstance(polygon.boundary, MultiLineString):
            for linestring in polygon.boundary:
                ax.add_patch(plt_polygon(np.array(linestring), closed=True, facecolor=(0, 0, 0, 0), edgecolor='red'))
        else:
            ax.add_patch(plt_polygon(np.array(polygon.boundary), closed=True, facecolor=(0, 0, 0, 0), edgecolor='red'))

    bbox_list = splitter.get_bbox_list()
    info_list = splitter.get_info_list()

    cm = plt.get_cmap('jet', len(bbox_list))
    legend_shapes = []
    for i, (bbox, info) in enumerate(zip(bbox_list, info_list)):
        wgs84_bbox = bbox.transform(CRS.WGS84).get_polygon()

        tile_color = tuple(list(cm(i))[:3] + [alpha])
        ax.add_patch(plt_polygon(np.array(wgs84_bbox), closed=True, facecolor=tile_color, edgecolor='green'))

        if show_legend:
            legend_shapes.append(plt.Rectangle((0,0),1,1, fc=cm(i)))

    if show_legend:
        legend_names = []
        for info in info_list:
            legend_name = '{},{}'.format(info['index_x'], info['index_y'])

            for prop in ['grid_index', 'tile']:
                if prop in info:
                    legend_name = '{},{}'.format(info[prop], legend_name)

            legend_names.append(legend_name)

        plt.legend(legend_shapes, legend_names)
    plt.tight_layout()
    plt.show()
# %%
osm_splitter = OsmSplitter([forest_area], CRS.WGS84, zoom_level=14)

print(repr(osm_splitter.get_bbox_list()[0]))
print(osm_splitter.get_info_list()[0])
print(len(osm_splitter.get_bbox_list()))
# %%
show_splitter(osm_splitter, show_legend=True)
# %%
SentinelHubBatch.get_tiling_grid(5, config=config)
# %%
