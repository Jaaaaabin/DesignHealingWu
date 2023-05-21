"""
This is the principal module of the healing project.
here you put all utilised external packages.
"""

# import packages

from datetime import datetime
from itertools import product, combinations, chain
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from natsort import natsorted
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
from SALib.plotting.hdmr import plot as hdmrplot
from SALib.sample import saltelli
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import namedtuple

import copy
import csv
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import plotly
import plotly.express as px
import plotly.graph_objects as go
import random
import seaborn as sns
import shutil
import tables
