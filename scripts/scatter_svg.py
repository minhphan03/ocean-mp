import pandas as pd
import numpy as np
import xarray as xr
import holoviews as hv
import panel as pn
hv.extension('bokeh')

from scipy.optimize import curve_fit

from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from re import search
from glob import glob
import os, warnings, sys
from shutil import rmtree
import scripts.svg_stack as ss
from bokeh.io import export_svgs

from typing import Union

# add driver to work on Windows
options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)

# add parent directory for data access
sys.path.append(os.pardir)

class ScatterGraph:
    """
    Creates an interactive dashboard containing yearly scatterplots of energy level on speed,
    using dropdowns to control frequency ranges and regression line types (linear and piecewise). 
    Read more about data source and further details in README.md file.
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

    def __init__(self, site: str, source: str, output_directory: Union[str, os.PathLike] = None) -> None:
        """
        Initialize the graph object

        Parameters:
        ------
        site: str
            code name (lowercased initials) for one of the five sites
        
        source: str
            code name (lowercased initials) for one of the two energy sources
        
        output_directory: str | Path
            absolute path where to save the dashboard product
        """
        if site.lower() not in list(self.SITE_NAMES.keys()) or source.lower() not in ['obs', 'hyd']:
            raise ValueError("Invalid input for object declaration. Consult documentation for valid arguments available.")
        
        # get data for graphing depending on sites and source
        self.wind_ds = xr.open_dataset(f'data/wind/{site}_modified.nc')
        
        if source == 'obs':
            self.energy_ds = xr.open_dataarray(f'data/obs/obs_{site}_modified.nc')
        else:
            self.energy_ds = xr.open_dataarray(f'data/hyd/{site}_01_95Hz_v2_modified.nc')
        
        self.site = site
        self.source = source
        self.site_name = self.SITE_NAMES[site]

        # placeholders for dropdown items
        self.dropdown1 = []
        self.dropdown2 = []
        
        if output_directory == None:
            self.outputdir_path = os.curdir
        else:
            if not os.path.exists(output_directory):
                raise TypeError('Invalid path for destination folder: %s' % (output_directory))
            self.outputdir_path = output_directory
       
        self.ylabel_unit = '(m/s)^2/Hz' if source == 'obs' else '(µPa)^2/Hz'
        
        self.df = self._generate_df()

        warnings.filterwarnings('ignore')
    

    def graph(self):
        """
        Main function to execute for graphing dashboard and saving final product to chosen directory.
        """

        # create temprary directory to store images
        self.tempdir_path = os.path.join(self.outputdir_path,f"{self.source}-{self.site}-scatter")
        if not os.path.exists(self.tempdir_path):
            os.makedirs(self.tempdir_path)

        for d1 in self.FREQ_RANGE_DROPDOWN:
            freq = search(r'(.*)-(.*)Hz', d1)
            min_freq = float(freq.group(1))
            max_freq = float(freq.group(2))

            df_res = self.df[(self.df['frequency'] >= min_freq) & (self.df['frequency'] <= max_freq)]

            for year in self.YEARS:
                df_year = df_res[df_res['year'] == year]

                if df_res.empty:
                    continue
                else:
                    SingleScatterGraph(
                        data=df_year,
                        year=year,
                        site_name=self.site_name, 
                        freq_range=d1, 
                        dir_name=self.tempdir_path,
                        ylabel_unit=self.ylabel_unit
                    ).build_plots()
            SingleScatterGraph(
                data=df_res,
                year='All',
                site_name=self.site_name, 
                freq_range=d1, 
                dir_name=self.tempdir_path,
                ylabel_unit=self.ylabel_unit
            ).build_plots()
        
        # stack and produce final dashboards
        self._stack()
        self._save()

        # delete temp folder
        try:
            rmtree(self.tempdir_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    def _stack(self):
        """
        Stacks rasterized graphs and saves complete panels into temporary directory before final processing
        """
        paths = glob(os.path.join(self.tempdir_path, '*.svg'))

        pattern_spec = r'^.*_(\S*-\S*)_(.*)(?:\.svg)$' #[YEAR]_[FREQRANGE | DD1]_[DD2]
        svg_storage = {}

        dropdown1_items = set()
        dropdown2_items = set()

        for file_path in paths:
            file_name = os.path.basename(file_path)
            captured = search(pattern_spec, file_name)

            dropdown1_items.add(captured.group(1))
            dropdown2_items.add(captured.group(2))
            key = (captured.group(1), captured.group(2))

            if key not in svg_storage:
                svg_storage[key] = [file_name]
            else:
                svg_storage[key].append(file_name)

        transformed_path = os.path.join(self.tempdir_path, 'transformed')
        if not os.path.exists(transformed_path):
            os.makedirs(transformed_path)

        # print("hello")
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

        self.dropdown1 = list(dropdown1_items)
        self.dropdown2 = list(dropdown2_items)

    def _save(self):
        """
        Combine processed SVG graphs and generate dropdowns for the final interactive graph,
        then save final product as a HTML file.
        """

        # Define your options for the dropdowns

        # Create the dropdown widgets
        freqrange_dropdown = pn.widgets.Select(name='frequency range', options=self.dropdown1)
        reg_dropdown = pn.widgets.Select(name='regression type', options=self.dropdown2)

        # Function to update the displayed image based on dropdown selections
        def update_image(event):
            selected_option1 = freqrange_dropdown.value
            selected_option2 = reg_dropdown.value
            
            # Replace the following with your logic to determine the image file path
            image_path = os.path.join(self.tempdir_path,f'transformed/{selected_option1}_{selected_option2}.svg')
            
            # Update the image widget
            image_object.object = pn.pane.SVG(image_path, width=900, height=1500).object # read documentation me thinks

        # Attach the update_image function to the on_change event of the dropdowns
        freqrange_dropdown.param.watch(update_image, 'value')
        reg_dropdown.param.watch(update_image, 'value')

        # Initial image (replace with default image path)
        initial_image_path = os.path.join(self.tempdir_path,f'transformed/{self.dropdown1[0]}_{self.dropdown2[0]}.svg')
        image_object = pn.pane.SVG(initial_image_path, width=1500, height=1500)

        # Create a Panel app layout
        app_layout = pn.Row(
            image_object,
            pn.Column(freqrange_dropdown, reg_dropdown)
        )

        # Show the app
        app_layout.servable()

        # Optionally, save the app to a standalone HTML file
        
        filename = self.tempdir_path.split('/')[-1]

        print(f"Saving file in {self.outputdir_path}")
        app_layout.save(filename=os.path.join(self.outputdir_path,f'{filename}.html'), embed=True)


    
    def _generate_df(self) -> pd.DataFrame:
        """
        Generates the main Pandas DataFrame to build the graphs

        Returns:
        -------
        pd.DataFrame
            Combined, complete DataFrame holding all wind and energy data for graphing
        """
        wind_data = self.wind_ds['speed'].to_dataframe()
        wind_data = wind_data.reset_index()
        wind_data['year'] = wind_data['time'].values.astype('datetime64[Y]').astype(int)+1970

        df = pd.DataFrame(np.repeat(wind_data.values, self.energy_ds['frequency'].shape[0], axis=0))
        df.columns = wind_data.columns

        df['frequency'] = np.tile(self.energy_ds['frequency'].values, self.wind_ds['time'].shape[0])
        df['energy'] = self.energy_ds.transpose('time', 'frequency').to_numpy().flatten()

        return df


class SingleScatterGraph:
    """
    A helper class to generate single scatter graphs and rasterized into temporary SVG files 
    for further processing
    """

    def __init__(self, year: Union[str , int], data: pd.DataFrame, site_name: str, dir_name: Union[str, os.PathLike], freq_range: str, ylabel_unit: str) -> None:
        """
        Initializes SingleScatterGraph object

        Parameters:
        --------
        year: int | str
            Year that goes into the title of the graph. In the case of cumulative graph, `year` takes
            the value 'All'

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
        
         # show trendline: linear, piecewise, no
        self.tempdir_name = dir_name
        self.data = data.replace('', np.nan).dropna().groupby('time').agg({'speed': 'mean', 'energy': 'mean'}).reset_index()

        self.ylabel_unit = ylabel_unit
        self.min_energy = self.data['energy'].min()


    def build_plots(self) -> hv.Overlay:
        """
        Builds a Holoviews object and save it as a SVG graphic file for further processing
        """

        num_data = hv.Text(5, self.min_energy, f"Number of points: {self.data['time'].nunique()}").opts(align='end', fontsize=10)

        base_graph = hv.Points(self.data, kdims=['speed', 'energy']).opts(
                        height=400,
                        width=600,
                        size=10,
                        xlabel='speed (m/s)',
                        ylabel=f'spectral level (dB rel 1 {self.ylabel_unit})',
                        title = f"{self.site_name}, {self.year}",
                        logx=True,
                        framewise=True
        )*num_data

        # dropdown2: trendline
        model_line_linear = self.model_line('linear')
        model_line_piecewise = self.model_line('piecewise')


        for dropdown2_option in ['linear', 'piecewise', 'none']:
            if model_line_linear is not None and dropdown2_option == 'linear':
                self._export_svg(base_graph * model_line_linear, "{}.svg".format(f"{self.year}_{self.freqrange_value}_{dropdown2_option}"))
            elif model_line_piecewise is not None and dropdown2_option == 'piecewise':
                self._export_svg(base_graph * model_line_piecewise, "{}.svg".format(f"{self.year}_{self.freqrange_value}_{dropdown2_option}"))
            else:
                self._export_svg(base_graph, "{}.svg".format(f"{self.year}_{self.freqrange_value}_{dropdown2_option}"))



    def model_line(self, line_feature) -> hv.Curve:
        """
        Builds a regression line based on type linear or piecewise
        """
        def linear(x, a, b):
            return a*np.log10(x) + b

        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0, x >= x0], 
                                [lambda x: k1*np.log10(x) + (y0 - k1*np.log10(x0)), 
                                lambda x: k2*np.log10(x) + (y0 - k2*np.log10(x0))])
        
        try:
            
            speed = self.data['speed']
            energy = self.data['energy']

            speed_ordered, energy_ordered = zip(*[(x, y) for x, y in sorted(zip(speed, energy)) if not np.isnan(x) and not np.isnan(y)])

            if line_feature == 'linear':
                popt, _ = curve_fit(linear, speed_ordered, energy_ordered, maxfev=5000)
                a, b = popt

                energy_pred_ordered = linear(speed_ordered, *popt)
                mse = np.mean((np.array(energy_ordered)-np.array(energy_pred_ordered))**2) 
                
                # since it's linear, speed up rendering by including as few points as possible
                speed_endpoints = list([speed.min(), speed.max()])

                return hv.Curve((speed_endpoints, linear(speed_endpoints, *popt)),label=f'{a:.2f}*log10(x)+({b:.2f}), mse={mse:.2f}').opts(
                    color='red',
                    height=400,
                    width=400,
                    logx=True
                )
            
            else:
                popt, _ = curve_fit(piecewise_linear, speed_ordered, energy_ordered, maxfev=5000)

                # attempt to render all since we don't know actual "breaking" point
                x0, y0, k1, k2 = popt

                energy_pred_ordered = piecewise_linear(speed_ordered, *popt)
                mse = np.mean((np.array(energy_ordered)-np.array(energy_pred_ordered))**2)

                return hv.Curve((speed_ordered, energy_pred_ordered),label=f'(x0, y0)=({x0:.2f}, {y0:.2f}), k1={k1:.2f}, k2={k2:.2f}, mse={mse:.2f}').opts(
                    color='red',
                    height=400,
                    width=400,
                    logx=True
                )
    

        except (TypeError, RuntimeError, ValueError) as e:
            print("Error trying to draw best fit line")
            print(e)
            return None
    
  
    def _export_svg(self, obj, filename):
        """
        Exports holoviews object (Overlay) to SVG file
        """
        plot_state = hv.renderer('bokeh').get_plot(obj).state
        plot_state.output_backend = 'svg'
        export_svgs(plot_state, filename=os.path.join(self.tempdir_name, filename), webdriver=driver)

