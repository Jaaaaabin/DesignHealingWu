
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import clhs as cl

ds = xr.tutorial.open_dataset('air_temperature') # use xr.tutorial.load_dataset() for xarray<v0.11.0
df=ds["air"][0,:,:].to_dataframe().reset_index()[["lat","lon","air"]]
# set temperature and relative humidity, relative humidty is normal distribution
df["temp"] = df["air"]-273.15
df["rh"] = np.random.normal(50, 12, 1325)
df.shape[0]

num_sample=15
# cLHS
sampled=cl.clhs(df[["temp","rh"]], num_sample, max_iterations=1000)
clhs_sample=df.iloc[sampled["sample_indices"]]
# random sample, as a comparison
random_sample=df.sample(num_sample)