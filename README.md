# Clean datasets and code files to create specialized dashboards using Holoviews and Matplotlib (no data files included due to copyright)
Developed by Minh Phan under supervision of Prof. Shima Abadi

## Why SVG export?

SVGs are vectorized files, meaning that they are lighter than PNGs, and take less time to render. Before this approach, we attempted to export complete dashboards constructed using HoloMaps/DynamicMaps from HoloViews, but they were often unstable and risk-prone due to the massive number of graphs rendered. This approach, while does not reduce the rendering time, offers a more stable solution to control our dashboard production pipeline. 

## Graphing Libraries

For our spectral density and scatter graphs, I utilized HoloViews per gradual development. However, for the time series graphs, I used Matplotlib's pyplot package as it supports dual axes, which the other does not.

## Data Source
Wind data are downloaded from NOAA CoastWatch Blended Sea Winds dataset. Despite the two different versions, data was taken from the same source and this update should not dramatically affect the outcome. Read more [in the Product Overview section](https://coastwatch.noaa.gov/cwn/products/noaa-ncei-blended-seawinds-nbs-v2.html).

- 2015/1 to 2018/10: version 1 (not available for public, with existing data now updated to version 2)
- 2018/10 - 2022/12: version 2 ([current version](https://coastwatch.noaa.gov/cwn/products/noaa-ncei-blended-seawinds-nbs-v2.html))

Cleaned energy data are recorded from Ocean Data Lab's sensors (low frequency hydrophone mean spectrogram data and obs zarr datasets)

## On Data Files
- Wind, obs, and hyd directories contain cleaned and synthesized datasets used in the scripts. 
- 6h_agg_x_component.nc and 6h_agg_y_component.nc are original wind data NetCDF4 files containing version 1 wind data that I incorporated into the cleaned datasets. They only serve to demonstrate the heatmaps. You can opt out of downloading them if you don't need to work on the heatmaps, as they are quite large in size.

## Requirements
1. Python 3.8 or older (for installing packages)
2. Windows OS (Linux is fine, too; however, this specific package is fine-tuned to use in Windows per request, with added support for downloaded drivers.) These scripts have not been tested using Linux.
3. Mozilla Firefox browser: for graphing and saving dashboards to local machine (Bokeh requirement)

## Installing environments & requirements
0. Make sure that you extract all files and folder as arranged into a new folder of your own.
1. Navigate to the folder you extracted data into, then run `py -3.8 -m venv [NAME_OF_ENVIRONMENT]` to create a new Python 3.8 virtual environment of your choice with customized name. A new folder with the same name will appear.
2. Activate environment: `[NAME_OF_ENVIRONMENT]/Scripts/activate`, for Windows.
3. Install necessary packages and library `pip install -r requirements.txt`.

## Running environments

1. Jupyter Notebooks: Disable duplicated heatmaps output by applying
```python
import matplotlib.pyplot as plt
plt.ioff() # disable interactive mode
```

## Errors & Exceptions
1. `Error trying to draw best fit line: not enough values to unpack (Expect 2, got 0)`: Happens when there is no data available to fit a regression, mostly in some of the yearly data graphing cases.

2. Opening Firefox tabs after executing scripts: this is a result from the inherent Bokeh (the backend renderer of Holoviews) exporting feature using browser drivers to render and export graphic files. Activating drivers results in selenium (the library to handle interacting with browsers using Python) opening Firefox windows in the background. Manually delete them after finishing running scripts.

3. `Javascript Error: IPython is not defined`: Graphing Matplotlib in Jupyter Notebooks will always print graphs, and since we produce lots of them, printing them out all at once will overload the memory. The script for producing time series dashboards includes a magic command that will disable the IPython output, which will prompt this error. Files are saved locally, so this error should not matter. Restart notebook or run `` to enable output at your own risk.

## Notes
1. Piecewise regression mostly turned out to look linear. This happens most of the time, with only few exceptions where a piecewise shape takes place.
2. Per request, all data is removed from 6/1/2019-7/15/2019 and 7/1/2021-8/15/2021
