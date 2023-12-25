import pandas as pd
import numpy as np
import xarray as xr
import holoviews as hv
import panel as pn
hv.extension('bokeh')

from re import search
from glob import glob
import os, warnings, sys
from shutil import rmtree
import scripts.svg_stack as ss
from bokeh.io import export_svgs

from typing import Union

from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

# add driver to work on Windows
options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)

# add parent directory for data access
sys.path.append(os.pardir)

class SpectralGraph:
    """
    Creates an interactive dashboard containing yearly spectral density graphs with specified legend parameter,
    using dropdowns to control frequency ranges, standard deviation ranges, and model line :math:`y=-7*log(x) + b` 
    [log-transformed] version of function :math:`y = x^{-7} + b`.
    Read more about data sourcea and further details in README.md file.
    """
    YEARS = list(range(2015, 2023))
    GROUP_DICT = {
        'speed': {
            '0.0_1.5m/s': '0.0', '1.5_3.0m/s': '1.5', '3.0_4.5m/s': '3.0', '4.5_6.0m/s': '4.5', 
            '6.0_7.5m/s': '6.0', '7.5_9.0m/s': '7.5', '9.0_10.5m/s': '9.0', '>10.5m/s': '10.5', 'ALL': 'all'
        },
        'direction': { 
            'North': 'N', 'East': 'E', 'West': 'W', 'South': 'S', 'ALL': 'all'
        },

        'duration': {
            '0_12h': '0.0', '12_24h': '12.0', '24_36h': '24.0', '36_48h': '36.0', '>48h': '48.0', 'ALL': 'all'
        }
    
    }

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

    def __init__(self, site: str, source: str, output_directory: Union[str, os.PathLike] = None) -> None:
        """
        Initializes a graphing object

        Parameters:
        ------
        site: str
            Code name for the site, one of ['ab', 'sb', 'cc', 'ec', 'sh']
        
        source: str
            Code name for the energy source, one of ['hyd', 'obs']

        output_directory: str | Path
            Where to store the graphs. By default (if no argument is provided), graphs are stored in the same directory where the file is run.
        """

        # get data for graphing depending on sites and source
        if site.lower() not in list(self.SITE_NAMES.keys()) or source.lower() not in ['obs', 'hyd']:
            raise ValueError("Invalid input for object declaration. Consult documentation for valid arguments available.")
        
        self.wind_ds = xr.open_dataset(f'data/wind/{site}_modified.nc')
        if source == 'obs':
            self.energy_ds = xr.open_dataarray(f'data/obs/obs_{site}_modified.nc')
        else:
            self.energy_ds = xr.open_dataarray(f'data/hyd/{site}_01_95Hz_v2_modified.nc')
        
        self.site = site
        self.source = source
        self.site_name = self.SITE_NAMES[site]
        
        # placeholders for the dropdowns
        self.dropdown1 = []
        self.dropdown2 = []
        self.dropdown3 = []

        self.ylabel_unit = '(m/s)^2/Hz' if source == 'obs' else '(µPa)^2/Hz'
        
        if output_directory == None:
            self.outputdir_path = os.curdir
        else:
            if not os.path.exists(output_directory):
                raise TypeError('Invalid path for destination folder: %s' % (output_directory))
            self.outputdir_path = output_directory
        
        self.df = self._generate_df()
        warnings.filterwarnings('ignore')

    

    def graph(self, legend: str ='speed'):
        """
        Main function to execute for graphing. 

        Parameters:
        -----
        legend: str
            group type to specify to group points, which is then used to graph spectral graphs 
            with discrete lines annotated with a legend using elements from the group
        """

        # create temporary directory to store images
        self.tempdir_path = os.path.join(self.outputdir_path,f"{self.source}-{self.site}-{legend}")
        if not os.path.exists(self.tempdir_path):
            os.makedirs(self.tempdir_path)

        # specify legend
        legend_args = {key : value for value, key in self.GROUP_DICT[legend].items()}  # reverse key and value to conform to object construction


        for d1 in self.FREQ_RANGE_DROPDOWN:
            # extract min and max frequency to slice from dataset
            freq = search(r'(.*)-(.*)Hz', d1)
            min_freq = float(freq.group(1))
            max_freq = float(freq.group(2))

            # slice dataset
            df_res = self.df[(self.df['frequency'] >= min_freq) & (self.df['frequency'] <= max_freq)]

            for year in self.YEARS:
                df_year = df_res[df_res['year'] == year]
                SingleSpectralGraph(
                    data=df_year,
                    year=year,
                    site_name=self.site_name, 
                    freq_range=d1, 
                    legend_name= f'{legend}_group',
                    legend_args= legend_args,
                    dir_name=self.tempdir_path,
                    ylabel_unit=self.ylabel_unit
                ).build_plots()

            SingleSpectralGraph(
                data=df_res,
                year='All',
                site_name=self.site_name, 
                freq_range=d1, 
                legend_name= f'{legend}_group',
                legend_args=legend_args, # reverse key and value to conform to object construction
                dir_name=self.tempdir_path,
                ylabel_unit=self.ylabel_unit
            ).build_plots()

        # perform stack and build final dashboard, then save
        self._stack()
        self._save()

        # delete temp folder
        try:
            rmtree(self.tempdir_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    def _stack(self):
        """
        Stacks rasterized graphs and saves complete panels before final processing
        """
        paths = glob(os.path.join(self.tempdir_path, '*.svg'))

        pattern_spec = r'^.*_(\S*-\S*)_(.*)_(.*)(?:\.svg)$' #[YEAR]_[FREQRANGE | DD1]_[DD2]_[DD3]
        svg_storage = {}

        dropdown1_items = set() # prevent duplicates
        dropdown2_items = set()
        dropdown3_items = set()

        # get distinct groups to perform groupby
        for file_path in paths:
            file_name = os.path.basename(file_path)
            captured = search(pattern_spec, file_name)

            dropdown1_items.add(captured.group(1))
            dropdown2_items.add(captured.group(2))
            dropdown3_items.add(captured.group(3))
            key = (captured.group(1), captured.group(2), captured.group(3))

            if key not in svg_storage:
                svg_storage[key] = [file_name]
            else:
                svg_storage[key].append(file_name)

        transformed_path = os.path.join(self.tempdir_path, 'transformed')
        if not os.path.exists(transformed_path):
            os.makedirs(transformed_path)

        # stack individual yearly graphs within group
        for key, list_of_files in svg_storage.items():
            doc = ss.Document()
            layoutV = ss.VBoxLayout()
            for ind, file in enumerate(list_of_files):
                if ind % 2 == 0:
                    layoutH = ss.HBoxLayout()
                
                layoutH.addSVG(os.path.join(self.tempdir_path, file), alignment=ss.AlignCenter)
                
                if (ind + 1) % 2 == 0 or ind == (len(list_of_files)-1): # 2 columns
                    layoutV.addLayout(layoutH)

            doc.setLayout(layoutV)
            doc.save(os.path.join(transformed_path, f"{'_'.join(key)}.svg"))
        
        # Define options for the dropdowns
        self.dropdown1 = list(dropdown1_items)
        self.dropdown2 = list(dropdown2_items)
        self.dropdown3 = list(dropdown3_items)
        

        return dropdown1_items, dropdown2_items

    def _save(self):
        """
        Combine processed SVG graphs and generate dropdowns for the final interactive graph,
        then save final product as a HTML file.
        """

        # Create the dropdown widgets
        freqrange_dropdown = pn.widgets.Select(name='frequency range', options=self.dropdown1)
        std_dropdown = pn.widgets.Select(name='standard deviation area', options=self.dropdown2)
        model_dropdown3 = pn.widgets.Select(name='model line y = x**(-7) + b', options=self.dropdown3)

        # Function to update the displayed image based on dropdown selections
        def update_image(event):
            selected_option1 = freqrange_dropdown.value
            selected_option2 = std_dropdown.value
            selected_option3 = model_dropdown3.value
            
            # Replace the following with your logic to determine the image file path
            image_path = os.path.join(self.tempdir_path,f'transformed/{selected_option1}_{selected_option2}_{selected_option3}.svg')
            
            # Update the image widget
            image_object.object = pn.pane.SVG(image_path, width=900, height=1500).object # read documentation me thinks

        # Attach the update_image function to the on_change event of the dropdowns
        freqrange_dropdown.param.watch(update_image, 'value')
        std_dropdown.param.watch(update_image, 'value')
        model_dropdown3.param.watch(update_image, 'value')

        # Initial image (replace with default image path)
        initial_image_path = os.path.join(self.tempdir_path,f'transformed/{self.dropdown1[0]}_{self.dropdown2[0]}_{self.dropdown3[0]}.svg')
        image_object = pn.pane.SVG(initial_image_path, width=1500, height=1500)

        # Create a Panel app layout
        app_layout = pn.Row(
            image_object,
            pn.Column(freqrange_dropdown, std_dropdown, model_dropdown3)
        )

        # Show the app
        app_layout.servable()

        # Optionally, save the app to a standalone HTML file
        
        filename = self.tempdir_path.split('/')[-1]

        # print(filename)
        app_layout.save(filename=os.path.join(self.outputdir_path,f'{filename}.html'), embed=True)


    
    def _generate_df(self) -> pd.DataFrame:
        """
        Generates the main Pandas DataFrame to build the graphs

        Returns:
        -----
        pd.DataFrame
            Combined, complete DataFrame holding all wind and energy data for graphing
        """
        wind_data = self.wind_ds[['speed_group', 'direction_group', 'duration_group']].to_pandas().astype('str')
        wind_data['year'] = self.wind_ds['time'].values.astype('datetime64[Y]').astype(int)+1970
        wind_data = wind_data.reset_index()

        df = pd.DataFrame(np.repeat(wind_data.values, self.energy_ds['frequency'].shape[0], axis=0))
        df.columns = wind_data.columns

        df['frequency'] = np.tile(self.energy_ds['frequency'].values, self.wind_ds['time'].shape[0])
        df['energy'] = self.energy_ds.transpose('time', 'frequency').to_numpy().flatten()

        return df



class SingleSpectralGraph:
    """
    A helper class to generate single spectral density graphs and rasterized into temporary SVG files 
    for further processing
    """

    COLOR_MAP = ['red', 'green', 'blue', 'black', 'cyan', 'purple', 'orange', 'brown', 'white']

    def __init__(self,  data: pd.DataFrame, legend_name: str, legend_args: dict, site_name: str, year: Union[int, str],dir_name: Union[str, os.PathLike], freq_range: str, ylabel_unit: str) -> None:
        """
        Initializes SingleSpectralGraph object

        Parameters:
        --------
        year: int | str
            Year that goes into the title of the graph. In the case of cumulative graph, `year` takes
            the value 'All'

        legend_name: str
            Specifies the name of the legend (either 'duration', 'speed', or 'direction')
        
        legend_args: dict
            Directory that maps data codes in DataFrame to displayed legend titles
        
        data: pd.DataFrame
            Sliced Pandas DataFrame that contains data points (frequency and energy) for the graph

        site_name: str
            Full name of the site that goes on the title of the graph
        
        dir_name: str | Path
            Pathname of the temporary directory to store the graphs

        freq_range: str
            Dropdown string for the frequency range specification, written in the form of []-[]Hz (see `FREQ_RANGE_DROPDOWN` constant in main object documentation)
        
        ylabel_unit: str
            Unit specified for either obs ((m/s)^2/Hz) or hydrophone ((µPa)^2/Hz) energy data
        """
        self.year = year
        self.site_name = site_name
        self.freqrange_value = freq_range # freq range

        # legend
        self.legend_name = legend_name
        self.legend_args = legend_args
        
        # show trendline: linear, piecewise, no

        self.tempdir_path = dir_name
        self.data = data.replace('', np.nan).dropna()

        self.ylabel_unit = ylabel_unit
        self.max_energy = self.data['energy'].max()

        # mapping all chart required parameters together (key, legend title, color, data) for consistency
        self.mapper = [{
            "legend_key": key,
            "legend_val": value,
            "color": color,
            "data": pd.DataFrame(),
            "num_points": 0
        } for key, value, color in zip(self.legend_args.keys(), self.legend_args.values(), self.COLOR_MAP)]


    def build_plots(self):
        """
        Builds a Holoviews object and rasterized it as a SVG graphic file for further processing
        """

        ## splitting raw data into groups according to legend keys
        for item in self.mapper:
            filtered_data = self.data[self.data[self.legend_name] == item['legend_key']]
            
            # group by
            grouped = filtered_data.groupby(['frequency'])['energy'].agg(['mean', 'std']).reset_index()
            grouped['lb'] = grouped['mean'] - 1/2*grouped['std']
            grouped['ub'] = grouped['mean'] + 1/2*grouped['std']

            item['data'] = grouped

            if item['legend_val'] == 'ALL':
                item['num_points'] = self.data['time'].nunique()
            else:
                item['num_points'] = filtered_data['time'].nunique()

        base_graph = self.base_graph()

        std_area = self.std_area_chart()

        model_line = self.model_line(self.data['frequency'].unique())

        # 1/2 standard deviation dropdown options
        for std_option in ['Y', 'N']:
            # threshold line dropdown option
            for model_option in ['Y', 'N']:
                graph = base_graph
                
                if model_line is not None and model_option == 'Y':
                    graph = graph * model_line
                if std_option == 'Y':
                    graph = graph * std_area
                
                self._export_svg(graph, "{}.svg".format(f"{self.year}_{self.freqrange_value}_{std_option}_{model_option}"))


    def base_graph(self):
        graph_dict = {"{} ({})".format(item['legend_val'], item['num_points']): 
            hv.Curve(
                data=item['data'],
                kdims='frequency',
                vdims='mean'
            ).opts(color=item['color']) 
 
            for item in self.mapper
        }

        return hv.NdOverlay(graph_dict, kdims=self.legend_name).opts(
            framewise=True, 
            height=400,
            width=600,
            xlabel='frequency (Hz)',
            ylabel='spectral level (dB rel 1 %s)' % (self.ylabel_unit),
            logx=True
        )


    def std_area_chart(self):
        std_area = lambda data, color : hv.Area(
            data=data,
            kdims='frequency',
            vdims=['lb', 'ub']
        ).opts(fill_alpha=0.2, color=color, line_alpha=0, framewise=True, logx=True)

        return hv.Overlay([std_area(item['data'], item['color']) for item in self.mapper]).opts(framewise=True)


    def model_line(self, freqs:list) -> hv.Curve:
        """
        Builds a regression line based on type linear or piecewise

        Parameters:
        -----
        freqs: list
            List of x-axis values to graph the model line on. Currently cutoff at 6Hz.
        
        Returns:
        -----
        hv.Curve
            Model line Curve object to add into the graph
        """
        freqs = sorted(freqs[freqs < 6])
        try:
            points = [(x,-7*np.log10(x) + self.max_energy) for x in freqs]
            return hv.Curve(points).opts(logx=True, framewise=True)

        except (TypeError, RuntimeError, ValueError) as e:
            print("Error trying to draw best fit line")
            print(e)
            return None
    
  
    def _export_svg(self, obj, filename):
        """
        Rasterizes and exports Holoviews object, specifically an NdOverlay object, to SVG file for further processing
        """
        if not os.path.exists(self.tempdir_path):
            os.makedirs(self.tempdir_path)
        plot_state = hv.renderer('bokeh').get_plot(obj).state
        plot_state.output_backend = 'svg'
        export_svgs(plot_state, filename=os.path.join(self.tempdir_path, filename), webdriver=driver)

