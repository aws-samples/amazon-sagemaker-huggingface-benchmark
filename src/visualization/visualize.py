# -*- coding: utf-8 -*-
"""
Custom scripts to visualize results/plot automatically elsewhere in the repository. 

Import this file as module in the desired exploratory notebook using the following code:
    # import local package modules
    # Ref: https://github.com/manifoldai/docker-cookiecutter-data-science/issues/4 
    import sys 
    sys.path.append("..")
    from src.visualization import visualize as viz
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
from scipy.interpolate import griddata

def make_surface(x, y, z, **kwargs):

    # interpret kwargs
    fig_title = kwargs.pop('fig_title')
    x_title = kwargs.pop('x_title')
    y_title = kwargs.pop('y_title')
    z_title = kwargs.pop('z_title')
    
    # make plot coords
    xi = np.linspace(min(x), max(x), num=100)
    yi = np.linspace(min(y), max(y), num=100)

    x_grid, y_grid = np.meshgrid(xi,yi)

    #Grid data
    z_grid = griddata((x,y),z,(x_grid,y_grid),method='cubic')

    # Plotly 3D Surface
    fig = go.Figure(go.Surface(x=x_grid,y=y_grid,z=z_grid,
                        colorscale='viridis'))
    fig.update_layout(scene = dict(
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    zaxis_title=z_title), title = fig_title,
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))

    fig.show()
    return


def make_3D_scatter(X, Y, Z, **kwargs):
    '''Create a 3D scatter plot based on individual datapoints.'''
    # fill results that are TBD with zero for now
    Z = np.asarray([num if str(num) != "nan" else 0 for num in Z])

    ax = plt.axes(projection='3d')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    zlabel = kwargs.pop('zlabel')

    ax.scatter(X, Y, Z, c=Z, cmap = 'viridis',linewidth=0.5);
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    
    ax.zaxis.set_label_text(zlabel)
    ax.ticklabel_format(axis='both', style='plain')
    plt.show()
    return 

def make_3D_surface(X, Y, Z, **kwargs):
    '''Makes a triangulated 3D surface plot based on individual datapoints.'''
    # fill any results that are TBD with zero for now
    Z = np.asarray([num if str(num) != "nan" else 0 for num in Z])
    
    ax = plt.axes(projection='3d')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    zlabel = kwargs.pop('zlabel')

    ax.plot_trisurf(X, Y, Z, edgecolor='none');
    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel, labelpad=10)
    ax.zaxis.set_label_text(zlabel)
    ax.ticklabel_format(axis='both', style='plain')
    
    plt.show()