from sentinelhub import SHConfig
from shapely.geometry import shape
from sentinelhub import OsmSplitter, BBox, read_data, CRS, Geometry, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
import geopandas as gpd

class SentilhubClient(object):

    def __init__(self):
        self.config = SHConfig()
        self.config()

        pass

    def config(self):
        if not self.config.sh_client_id or not self.config.sh_client_secret:
            self.config.instance_id = '464e0857-a72d-4c5d-b0d5-a96b724301d7'
            self.config.sh_client_id = '2a713948-9158-4185-85df-4f0d271b644d'
            self.config.sh_client_secret = ":%lG1hmr?J#nkvmjq/9.(K*Tz[q:7ES-kN>{4A}P"
            self.config.save()

    def get_status(self):
        pass

    def get_json(self):
        pass

    def split_forest_area(self):
        """
        INPUT:
            forest_map : TODO
        OUTPUT:
            list of BBOX objects in the form of [((lat1, lon1), (lat2, lon2)), ...]
        """
        file = self.get_json()
        geo_json = read_data(file)
        forest_area = shape(geo_json['features'][0]['geometry'])
        osm_splitter = OsmSplitter([forest_area], CRS.WGS84, zoom_level=14)
        return osm_splitter.get_bbox_list()

    def get_forest(self):

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

        file = self.get_json()
        geo_json = read_data(file)
        full_geometry = Geometry(geo_json['features'][0]['geometry'], crs=CRS.WGS84)

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
            config=self.config
        )
        image = request.get_data()[0]
        return image

    def get_tile_batch():

        pass

    def get_tile(self, bbox_coords, resolution, start_date, end_date):
        """
        INPUT:
            bbox_coords : list of floats in format [lat1, lon1, lat2, lon2]
            resolution : res of the tile [int]
            start_date : start date to collect of given coordinates [String format : '2020-06-01']
            end_date : last date excluding to consider [same format as above]
        OUTPUT:
            png image 
        """
        tile_bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        tile_size = bbox_to_dimensions(tile_bbox, resolution=resolution)

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
                    mosaicking_order='leastCC'
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.PNG)
            ],
            bbox=tile_bbox,
            size=tile_size,
            config=self.config
        )
        true_color_imgs = request_true_color.get_data()
        return true_color_imgs

