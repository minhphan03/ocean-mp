"""
Class to create speed heatmaps as described in heatmap.ipynb file (direction not incorporated yet).
WARNING: legacy code not polished, may need more development for furhter configuration. Demonstration purposes only.
"""

import pandas as pd
import numpy as np
import xarray as xr
import math
import sys, os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from scipy.interpolate import griddata
from typing import List, Union, Tuple

import geopy.distance as gp
from geopy.point import Point

import warnings
warnings.filterwarnings('ignore')


sys.path.append(os.pardir)
plt.ioff()

class Heatmap:
    """
    Graph interpolated heat maps from discrete data points in a wind dataset (predefined) for demonstration purposes. No saving method.
    """
    def __init__(self, POINT: str, corner_distance: float = 250) -> None:
        self.point = self.get_point(POINT.upper())
        self.DISTANCE = math.sqrt(corner_distance**2 + corner_distance**2) # 500x500km box
        self.ds = xr.open_mfdataset('data/6h_agg*.nc')

    def graph(self, time_idxs: List[int] = [0], variable_names: tuple = {'u', 'v'}):
        """
        Main function to visualize data from 
        
        Parameters:
        -------
        time_idxs: list
            time index (specified in the dataset) we want to graph on (prefer)
        
        variable_names: tuple
            Datasets may have different names for horizontal and vertical components. 
            Declare their names to prevent confusion. Default values are 'u' and 'v.'

        """
        u, v = variable_names
        ul_lat_idx, ul_lon_idx, lr_lat_idx, lr_lon_idx = self.find_index(self.DISTANCE, return_bounds=False)

        num_axes = len(time_idxs)
        fig, axes = plt.subplots(1, num_axes, figsize=(10*num_axes, 10)) # ideally about 2-4 graphs is sufficient

        for index, time_idx in enumerate(time_idxs):
            # extract the "box" containing all the data points within the area
            box = self.ds.isel(time=time_idx,lat=slice(lr_lat_idx,(ul_lat_idx+1)), lon=slice(ul_lon_idx,(lr_lon_idx+1)))
            
            # calculate speed from x and y components
            box = box.assign(speed = lambda x: np.sqrt(x[u]**2 + x[v]**2))
            wind = box['speed']

            # return wind 
            df = self.build_dataframe(wind, self.point)

            # get bounds for 100x100km average box & calculate distance
            ul_lat_sample, ul_lon_sample, lr_lat_sample, lr_lon_sample = self.find_index( math.sqrt(50**2 + 50**2), return_bounds=True)

            ul_hdist, ul_vdist = self._calculate((ul_lon_sample, ul_lat_sample))
            lr_hdist, lr_vdist = self._calculate((lr_lon_sample, lr_lat_sample))
            
            # plot cables, boxes, and anchor point

            if num_axes == 1:
                self._heatmap(df, axes, fig, ul_hdist, ul_vdist, lr_hdist, lr_vdist)
            else:
                self._heatmap(df, axes[index], fig, ul_hdist, ul_vdist, lr_hdist, lr_vdist)

        return fig


    def _heatmap(self, df: pd.DataFrame, ax: plt.Axes, fig: Figure, ul_hdist: float, ul_vdist: float, lr_hdist: float, lr_vdist: float):
        """Plot an interpolated heatmap
        
        Parameters:
        -------
        df: pd.DataFrame
            table/DataFrame containing wind data at the discrete points within the distance from the point
        
        ax: plt.Axes
            Axes object on which we draw the heatmap on
        
        fig: plt.Figure
            Figure object to contain the Axes object
        
        ul_hdist: float
            Actual horizontal distance of the farthest point on the upper left corner boundary of our focused area

        ul_vdist: float
            Actual vertical distance of the farthest point on the upper left corner boundary of our focused area

        lr_hdist: float
            Actual horizontal distance of the farthest point on the lower right corner boundary of our focused area

        lr_vdist: float
            Actual vertical distance of the farthest point on the lower right corner boundary of our focused area
        """
        # using interpolation for continuous plotting heating
        vertical_list = df['vertical'].to_numpy()
        horizontal_list = df['horizontal'].to_numpy()
        wind_list = df['wind_speed'].to_numpy()
        min_h_dist = np.amin(horizontal_list)
        max_h_dist = np.amax(horizontal_list)
        min_v_dist = np.amin(vertical_list)
        max_v_dist = np.amax(vertical_list)
        
        # define grid
        hi = np.linspace(min_h_dist, max_h_dist, 500)
        vi = np.linspace(min_v_dist, max_v_dist, 500)
        hi, vi = np.meshgrid(hi, vi)
        
        # interpolate
        zi = griddata((horizontal_list, vertical_list), wind_list, (hi, vi), method='linear')
        
        # plot interpolated gradient
        image = ax.imshow(zi, extent=(min_h_dist, max_h_dist, min_v_dist, max_v_dist), origin='lower', vmin=0, vmax=20)
        
        # plot center point
        ax.scatter(0, 0, s=10,marker="o", c='white') 

        # plot 100x100km box & actual data coverage box (due to discretion)
        rect = patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='white', facecolor='none', label='actual boundary 100x100km box')
        rect_data = patches.Polygon([
            (ul_hdist, ul_vdist),
            (lr_hdist, ul_vdist),
            (lr_hdist, lr_vdist),
            (ul_hdist, lr_vdist)
        ], linewidth=1, edgecolor='red', facecolor='none', label='data boundary')
        ax.add_patch(rect)
        ax.add_patch(rect_data)
        
        # add colorbar
        fig.colorbar(image, ax=ax)

        ax.legend(handles=[rect, rect_data])

        # add title (not yet implemented)

    def build_dataframe(self, data_array: xr.DataArray, point: tuple)-> pd.DataFrame:
        """
        Builds and returns a Pandas DataFrame containing discrete points' distances
        from the center point and their wind speed

        Parameters:
        --------
        data_array: xr.DataArray
            a 2D array containing wind speed data with contained dimension information
        
        point: tuple
            tuple containing coordinates of the center point
        """
        raw = []
        wind_list = np.nan_to_num(data_array.values.flatten())
        lat_list = data_array.coords['lat'].values
        lon_list = data_array.coords['lon'].values
        map_ = [(j, i) for i in lat_list for j in lon_list]
        # print(map_)
        for tuple_, speed in list(zip(map_, wind_list)):
            # unpack lon and lat
            h_dist, v_dist = self._calculate(tuple_)
            # print(h_dist, v_dist, tuple_)
            raw.append({
                'horizontal': h_dist,
                'vertical': v_dist,
                'wind_speed': speed
            })
        df = pd.DataFrame(raw)
        return df


    def _calculate(self, reference_point: tuple) -> Tuple[float, float]:
        """
        Calculates the Vincenty distance between two points

        Parameters:
        -------
        center_point: tuple
            contains longitude and latitude coordinates of the center point of the map
        
        reference_point: tuple
            contains longitude and latitude coordinates of the reference point 
            in which we wants to calculate the distance from
        """

        # return horizontal and vertical distances
        clon, clat = self.point
        rlon, rlat = reference_point
        
        # calculate distance, flip the signs to reflect the center point at (0,0)
        v_dist = gp.distance((clat, clon), (rlat, clon)).km
        v_dist = -v_dist if rlat < clat else v_dist
        h_dist = gp.distance((clat, clon), (clat, rlon)).km
        h_dist = -h_dist if rlon < (clon + 360) else h_dist
        return h_dist, v_dist


    def find_index(self, distance: float ,return_bounds: bool) -> Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]]:
        """
        Returns the index positions of the boundaries within a distance from a point
        in the xarray dataset to visualize areas using interpolation using Vincenty method.
        Note that the map model this function depends on is 0.25x0.25deg, 30-60N and 220-250E

        Parameters:
        ----------
        dist: float
            Distance (diagonally) we want to find the index positions from the point
        
        return_bounds: bool, optional
            A flag used to indicate whether to return boundaries coordinates or not
            (default is false: return indices instead)

        Returns:
        --------
        float | int
            Depends on value of return_bounds, returns the value for latitude and longitude boundaries
            of the upper left and lower right, in respective order
        """
        lon, lat = self.point
        initial_point = Point(lat, lon)
        
        # our map covers 30-60N and 220-250E and have 0.25 degrees grid
        lon_idx = round(((360 + lon) - 220)/0.25) # West degree coordinates have negative signs
        lat_idx = round((lat - 30)/0.25)
        
        # upper-left corner
        ul_point_dist = gp.distance(kilometers=distance).destination(point=initial_point, bearing=315)
        ul_lat, ul_lon = ul_point_dist.latitude, ul_point_dist.longitude
        
        # find index of points closest to this corner, round DOWN longitude and UP latitude
        ul_lat_idx = math.ceil((ul_lat-30)/0.25)
        ul_lon_idx = math.floor(((360 + ul_lon) - 220)/0.25)
        
        # coordinates of the upper left corner of the data slicer
        ul_lat_sample = ul_lat_idx*0.25+30
        ul_lon_sample = ul_lon_idx*0.25 + 220
        
        # lower-right corner
        lr_point_dist = gp.distance(kilometers=distance).destination(initial_point, 135)
        lr_lat, lr_lon = lr_point_dist.latitude, lr_point_dist.longitude
        
        # find index of points closest to this corner, ROUND UP LATITUDE AND DOWN LONGITUDE
        lr_lat_idx = math.floor((lr_lat-30)/0.25)
        lr_lon_idx = math.ceil(((360 + lr_lon) - 220)/0.25)
        
        # coordinates of the lower right corner of the data slicer
        lr_lat_sample = lr_lat_idx*0.25 + 30
        lr_lon_sample = lr_lon_idx*0.25 + 220
        
        if return_bounds:
            return ul_lat_sample, ul_lon_sample, lr_lat_sample, lr_lon_sample # return data boundaries
        else:
            return ul_lat_idx, ul_lon_idx, lr_lat_idx, lr_lon_idx # return indices


    def get_point(self, point: str) -> Tuple[float, float]:
        """
        Returns coordinates for one of the five sites for graphing

        Parameters:
        ------
        point: str
            Code name for site
        
        Returns:
        -----
        tuple
            longitude and latitude of the site
        """

        points = {
            "EC": (-129.9728, 45.9396),
            "AB": (-129.754, 45.8168),
            "CC": (-130.0089, 45.9546),
            "SB": (-125.39, 44.5153),
            "SH": (-125.1479, 44.5691)
        }
        if point in points:
            return points[point]
        else:
            raise ValueError("Wrong value for site code name.")

    def save(self, fig: Figure, output_dir = None):
        """
        Save Figure
        """
        pass
