"""
Build time series graphs for average 7-day window comparisons between wind speed and energy sources.
Configure data file destinations before use.
"""

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
plt.ioff()

import panel as pn

from re import search
from glob import glob
import os, warnings
from shutil import rmtree
import scripts.svg_stack as ss
from typing import Union, Optional
from time import sleep

from IPython import get_ipython
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesGraph:
    """
    Produces interactive average 1 week window time series graphs
    to compare phases between wind and energy
    """
    YEARS = list(range(2015, 2023))

    FREQ_RANGE_DROPDOWN =  [
        '0.1-95Hz', '0.1-10Hz',
        '1-10Hz', '10-95Hz'
    ]

    SITE_NAMES = {
        'ab': 'Axial Base',
        'sb': 'Slope Base',
        'cc': 'Central Caldera',
        'ec': 'Eastern Caldera',
        'sh': 'Southern Hydrate'
    }

    def __init__(self, site: str, output_directory: Union[str, os.PathLike] = None) -> None:
        """
        Initialize the graph object

        Parameters:
        ------
        site: str
            code name (lowercased initials) for one of the five sites
        
        output_directory: str | Path
            absolute path where to save the dashboard product
        """

        if site.lower() not in list(self.SITE_NAMES.keys()):
            raise ValueError("Invalid input for object declaration. Consult documentation for valid arguments available.")
        

        # get data for graphing depending on sites and source
        self.wind = xr.open_dataset(f'data/wind/{site}_modified.nc')
        
        self.obs = xr.open_dataarray(f'data/obs/obs_{site}_modified.nc')
        self.hyd = xr.open_dataarray(f'data/hyd/{site}_01_95Hz_v2_modified.nc')
        
        self.site = site
        self.site_name = self.SITE_NAMES[site]
        
        if output_directory == None:
            self.outputdir_path = os.curdir
        else:
            if not os.path.exists(output_directory):
                raise TypeError('Invalid path for destination folder: %s' % (output_directory))
            self.outputdir_path = output_directory

        warnings.filterwarnings('ignore')

        self.dropdown1 = []
        self.dropdown2 = []
        self.dropdown3 = []
    

    def graph(self, split_wind_energy: bool = True):
        """
        Main function to produce a graph 

        Parameters:
        ------
        split_wind_energy: bool
            Flag to whether compare wind with each energy source (True), or keep them separate 
            (wind as one graph, energy source on dual axis on another).
        """
        ipython = get_ipython()
        if ipython is not None: # if script is ran on jupyter notebook, suppress all graph outputs
            print("Suppressing all graph outputs... will output error when showing graphs")
            ipython.run_line_magic("matplotlib", "notebook") # disable showing graph on Jupyter (not working at the moment)

        split_state = 'split' if split_wind_energy else 'nonsplit'
        self.tempdir_path = os.path.join(self.outputdir_path,f"{self.site}-{split_state}-timeseries")
        if not os.path.exists(self.tempdir_path):
            os.makedirs(self.tempdir_path)

        for d1 in self.FREQ_RANGE_DROPDOWN:
            freq = search(r'(.*)-(.*)Hz', d1)
            min_freq = float(freq.group(1))
            max_freq = float(freq.group(2))

            df_freq = self._generate_df(freq_start = min_freq, freq_end = max_freq)

            # build graphs for each year
            for year in self.YEARS:
                df_year = df_freq[df_freq['year'] == year]
                
                TimeSeriesGraphHelper(
                    data = df_year,
                    year=year,
                    site_name=self.site_name, 
                    freq_range=d1, 
                    dir_name=self.tempdir_path,
                    split_wind_energy=split_wind_energy
                ).build_plots()

            # build cumulative graph
            TimeSeriesGraphHelper(
                data=df_freq,
                year='All',
                site_name=self.site_name, 
                freq_range=d1, 
                dir_name=self.tempdir_path,
                split_wind_energy=split_wind_energy
            ).build_plots()

        self._stack()
        self._save()

        # delete temp folder
        try:
            rmtree(self.tempdir_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        finally:
            sleep(1)

        
        if ipython is not None: # if script is ran on jupyter notebook, suppress all graph outputs
            ipython.run_line_magic("matplotlib", "inline") # disable showing graph on Jupyter (not working at the moment)


    def _stack(self):
        """
        Stacks rasterized graphs before final processing with `panel`
        """
        paths = glob(os.path.join(self.tempdir_path, '*.svg'))

        pattern_spec = r'^.*_.*_(\S*-\S*)_(.*)_(.*)(?:\.svg)$' # [YAXIS1]_[YAXIS2]_[FREQ_RANGE]_[YEAR]_[STDEV]
        svg_storage = {}

        dropdown1_items = set()
        dropdown2_items = set()
        dropdown3_items = set()

        for file in paths:
            file_name = os.path.basename(file)
            captured = search(pattern_spec, file_name)

            dropdown1_items.add(captured.group(1)) # freqrange
            dropdown2_items.add(captured.group(2)) # year
            dropdown3_items.add(captured.group(3)) # stdev
            key = (captured.group(1), captured.group(2), captured.group(3))

            if key not in svg_storage:
                svg_storage[key] = [file_name]
            else:
                svg_storage[key].append(file_name)

        transformed_path = os.path.join(self.tempdir_path, 'transformed')
        if not os.path.exists(transformed_path):
            os.makedirs(transformed_path)

        for key, list_of_files in svg_storage.items():
            doc = ss.Document()
            layoutV = ss.VBoxLayout()

            for file in list_of_files:
                
                layoutV.addSVG(os.path.join(self.tempdir_path, file), alignment=ss.AlignCenter)

            doc.setLayout(layoutV)
            doc.save(os.path.join(transformed_path, f"{'_'.join(key)}.svg"))

        self.dropdown1 = list(dropdown1_items) # freq range dropdown items
        self.dropdown2 = sorted(list(dropdown2_items)) # year dropdown items
        self.dropdown3 = list(dropdown3_items) # std dropdown values (Y/N)

    def _save(self):
        """
        Combine processed SVG graphs and generate dropdowns for the final interactive graph,
        then save final product as a HTML file.
        """
        
        # Create the dropdown widgets
        freqrange_dropdown = pn.widgets.Select(name='frequency range', options=self.dropdown1)
        year_dropdown = pn.widgets.Select(name='year', options=self.dropdown2)
        std_dropdown = pn.widgets.Select(name='1 standard deviation range', options=self.dropdown3)

        # Function to update the displayed image based on dropdown selections
        def update_image(event):
            selected_option1 = freqrange_dropdown.value
            selected_option2 = year_dropdown.value
            selected_option3 = std_dropdown.value
            
            # Replace the following with your logic to determine the image file path
            image_path = os.path.join(self.tempdir_path,f'transformed/{selected_option1}_{selected_option2}_{selected_option3}.svg')
            
            # Update the image widget
            image_object.object = pn.pane.SVG(image_path, width=900, height=1500).object # read documentation me thinks

        # Attach the update_image function to the on_change event of the dropdowns
        freqrange_dropdown.param.watch(update_image, 'value')
        year_dropdown.param.watch(update_image, 'value')
        std_dropdown.param.watch(update_image, 'value')

        # Initial image (replace with default image path)
        initial_image_path = os.path.join(self.tempdir_path,f'transformed/{self.dropdown1[0]}_{self.dropdown2[0]}_{self.dropdown3[0]}.svg')
        image_object = pn.pane.SVG(initial_image_path, width=1200, height=1200)

        # Create a Panel app layout
        app_layout = pn.Row(
            image_object,
            pn.Column(freqrange_dropdown, year_dropdown, std_dropdown)
        )

        # Show the app
        app_layout.servable()

        # Optionally, save the app to a standalone HTML file
        
        filename = self.tempdir_path.split('/')[-1]

        # print(filename)
        app_layout.save(filename=os.path.join(self.outputdir_path,f'{filename}.html'), embed=True)

    
    def _generate_df(self, freq_start: float, freq_end: float):
        """
        Slice data according to frequency range and take statistic measures to graph

        Parameters:
        -------
        freq_start: float
            lower end of the frequency range to slice

        freq_end: float
            higher end of the frequency range to slice
        """
        data = self.wind['speed'].to_dataframe(name='wind')
        data['obs'] = self.obs.sel(frequency=slice(freq_start, freq_end)).mean('frequency').to_pandas()
        data['hyd'] = self.hyd.sel(frequency=slice(freq_start, freq_end)).mean('frequency').to_pandas()
        
        # take average window
        data_group = data.groupby([pd.Grouper(freq='W')]).agg(['mean', 'std']).reset_index() # mid-week

        data_group['year'] = data_group['time'].values.astype('datetime64[Y]').astype(int)+1970

        return data_group
    

class TimeSeriesGraphHelper:
    """
    Helper class to build and save a single interactive time series graph 
    temporarily into local machine storage for later build
    """
    COLORS = {
        'wind': 'green',
        'hyd' : 'blue',
        'obs': 'red'
    }

    YAXIS_LABELS = {
        'obs': 'hydrophone energy (dB rel 1 (m/s)^2/Hz)',
        'hyd': 'obs energy (dB rel 1 (µPa)^2/Hz)',
        'wind': 'wind speed (m/s)'
    }

    def __init__(self, data: pd.DataFrame, year: Union[str, int], site_name: str, freq_range: str, dir_name: str, split_wind_energy: bool) -> None:
        """
        Initializes helper object to graph and save individual dashboard panels

        Parameters:
        -------
        year: int | str
            Year that goes into the title of the graph. In the case of cumulative graph, `year` takes
            the value 'All'

        data: pd.DataFrame
            Sliced Pandas DataFrame that contains data points (time, speed, and energy) for graphing

        site_name: str
            Full name of the site that goes on the title of the graph
        
        dir_name: str | Path
            Pathname of the temporary directory to store the graphs

        freq_range: str
            Dropdown string for the frequency range specification, written in the form of []-[]Hz (see `FREQ_RANGE_DROPDOWN` constant in main object documentation)
        
        split_wind_energy: bool
            Flag to whether compare wind with each energy source (True), or keep them separated 
            with wind as one graph, and energy source on dual axes on another (False).
            
        """
        self.year = year
        self.data= data
        self.site_name = site_name
        self.freqrange_value = freq_range # freq range

        self.split_wind_energy = split_wind_energy
        
         # show trendline: linear, piecewise, no
        self.dir_name = dir_name
        
        # self.min_energy = self.data['energy'].min()


    def build_plots(self):
        """
        Creates wind and energy plots for further processing
        """
        
        for std_choice in ['Y', 'N']:
            if self.split_wind_energy:
                self._plot(var1='wind', var2='hyd', std_option=std_choice)
                self._plot(var1='wind', var2='obs', std_option=std_choice)
            else:
                self._plot(var1='wind', std_option=std_choice)
                self._plot(var1='hyd', var2='obs', std_option=std_choice)


    def _plot(self, var1: str, std_option: str = 'Y', var2: Optional[str] = None) -> None:
        """
        Builds and saves comparison plots in SVG

        Parameters:
        ------
        var1: str
            Name of the first variable to graph (either 'wind', 'obs', or 'hyd')
        var2: str | none
            Name of the second variable to graph (optional). If not specified, duplicate the first to preserve graph consistency.

        std_option: str
            Flag whether to graph 1 standard deviation area chart around the line graph.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        _ = ax.plot(self.data['time'], self.data[var1]['mean'], color = self.COLORS[var1]);
        ax.set_ylabel(self.YAXIS_LABELS[var1], color=self.COLORS[var1])

        if var2 is not None:
            ax2 = ax.twinx()
            _ = ax2.plot(self.data['time'], self.data[var2]['mean'], color = self.COLORS[var2]);
            ax2.set_ylabel(self.YAXIS_LABELS[var2], color=self.COLORS[var2])

        if std_option == 'Y':
            _ = ax.fill_between(
                self.data['time'], # x axis
                y1=self.data[var1]['mean']+self.data[var1]['std'], 
                y2=self.data[var1]['mean']-self.data[var1]['std'], 
                color=self.COLORS[var1], 
                alpha=0.2
            )
            
            if var2 is not None:
                _ = ax2.fill_between(
                    self.data['time'], # x axis
                    y1=self.data[var2]['mean']+self.data[var2]['std'], 
                    y2=self.data[var2]['mean']-self.data[var2]['std'], 
                    color=self.COLORS[var2], 
                    alpha=0.2
                )
        fig.tight_layout()

        var1_name = var1
        var2_name = var1_name if var2 == None else var2

        fig.savefig(os.path.join(self.dir_name, f'{var1_name}_{var2_name}_{self.freqrange_value}_{self.year}_{std_option}.svg'), format='svg')

