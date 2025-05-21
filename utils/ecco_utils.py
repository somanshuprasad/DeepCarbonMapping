#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division,print_function
import numpy as np
import matplotlib.pylab as plt
import xarray as xr
import xgcm

# The Proj class can convert from geographic (longitude,latitude) to native
# map projection (x,y) coordinates and vice versa, or from one map projection
# coordinate system directly to another.
# https://pypi.python.org/pypi/pyproj?
#
import pyresample as pr


def get_llc_grid(ds,domain='global'):
    """
    Define xgcm Grid object for the LLC grid
    See example usage in the xgcm documentation:
    https://xgcm.readthedocs.io/en/latest/example_eccov4.html#Spatially-Integrated-Heat-Content-Anomaly

    Parameters
    ----------
    ds : xarray Dataset
        formed from LLC90 grid, must have the basic coordinates:
        i,j,i_g,j_g,k,k_l,k_u,k_p1

    Returns
    -------
    grid : xgcm Grid object
        defines horizontal connections between LLC tiles

    """

    if 'domain' in ds.attrs:
        domain = ds.attrs['domain']

    if domain == 'global':
        # Establish grid topology
        tile_connections = {'tile':  {
                0: {'X': ((12, 'Y', False), (3, 'X', False)),
                    'Y': (None, (1, 'Y', False))},
                1: {'X': ((11, 'Y', False), (4, 'X', False)),
                    'Y': ((0, 'Y', False), (2, 'Y', False))},
                2: {'X': ((10, 'Y', False), (5, 'X', False)),
                    'Y': ((1, 'Y', False), (6, 'X', False))},
                3: {'X': ((0, 'X', False), (9, 'Y', False)),
                    'Y': (None, (4, 'Y', False))},
                4: {'X': ((1, 'X', False), (8, 'Y', False)),
                    'Y': ((3, 'Y', False), (5, 'Y', False))},
                5: {'X': ((2, 'X', False), (7, 'Y', False)),
                    'Y': ((4, 'Y', False), (6, 'Y', False))},
                6: {'X': ((2, 'Y', False), (7, 'X', False)),
                    'Y': ((5, 'Y', False), (10, 'X', False))},
                7: {'X': ((6, 'X', False), (8, 'X', False)),
                    'Y': ((5, 'X', False), (10, 'Y', False))},
                8: {'X': ((7, 'X', False), (9, 'X', False)),
                    'Y': ((4, 'X', False), (11, 'Y', False))},
                9: {'X': ((8, 'X', False), None),
                    'Y': ((3, 'X', False), (12, 'Y', False))},
                10: {'X': ((6, 'Y', False), (11, 'X', False)),
                     'Y': ((7, 'Y', False), (2, 'X', False))},
                11: {'X': ((10, 'X', False), (12, 'X', False)),
                     'Y': ((8, 'Y', False), (1, 'X', False))},
                12: {'X': ((11, 'X', False), None),
                     'Y': ((9, 'Y', False), (0, 'X', False))}
        }}

        grid = xgcm.Grid(ds,
                periodic=False,
                face_connections=tile_connections
        )
    elif domain == 'aste':
        tile_connections = {'tile':{
                    0:{'X':((5,'Y',False),None),
                       'Y':(None,(1,'Y',False))},
                    1:{'X':((4,'Y',False),None),
                       'Y':((0,'Y',False),(2,'X',False))},
                    2:{'X':((1,'Y',False),(3,'X',False)),
                       'Y':(None,(4,'X',False))},
                    3:{'X':((2,'X',False),None),
                       'Y':(None,None)},
                    4:{'X':((2,'Y',False),(5,'X',False)),
                       'Y':(None,(1,'X',False))},
                    5:{'X':((4,'X',False),None),
                       'Y':(None,(0,'X',False))}
                   }}
        grid = xgcm.Grid(ds,periodic=False,face_connections=tile_connections)
    else:
        raise TypeError(f'Domain {domain} not recognized')


    return

def UEVNfromUXVY(x_fld,y_fld, coords, grid=None):
    """Compute the zonal and meridional components of a vector field defined
    by its x and y components with respect to the model grid.

    The x and y components of the vector can be defined on model grid cell edges ('u' and 'v' points),
    or on model grid cell centers ('c' points). If the vector components are defined on the grid cell edges then the function will first interpolate them to the grid cell centers. 
    
    Once both x and y vector components are at the cell centers, they are rotated to the zonal and meridional components using the cosine and sine of the grid cell angle. The grid cell angle is defined as the angle between the Earth's parallels and the line connecting the center of a grid cell to the center of its neighbor in the x-direction. The cosine and sine of this angle are provided in the 'coords' input field.

    The function will raise an error if the grid cell angle terms (CS, SN) are not provided in the coords

    Example vector fields provided at the grid cell edges are UVEL and VVEL. Example fields provided at the grid cell centers are EXFuwind and EXFvwind. 

    Note: this routine is inspired by gcmfaces_calc/calc_UEVNfromUXVY.m

    Parameters
    ----------
    x_fld, y_fld : xarray DataArray
        x and y components of a vector field provided at the model grid cell edges or centers
    coords : xarray Dataset
        must contain CS (cosine of grid orientation) and SN (sine of grid orientation)
    grid : xgcm Grid object, optional
        see ecco_utils.get_llc_grid and xgcm.Grid
        If not provided, the function will create one using the coords dataset.
    
    Returns
    -------
    e_fld, n_fld : xarray DataArray
        zonal (positive east) and meridional (positive north) components of input vector field 
        at grid cell centers/tracer points
    """

    # Check to make sure 'CS' and 'SN' are in coords
    # before doing calculation
    required_fields = ['CS','SN']
    for var in required_fields:
        if var not in coords.variables:
            raise KeyError('Could not find %s in coords Dataset' % var)

    # If no grid, establish it
    if grid is None:
        grid = get_llc_grid(coords)

    # Determine if vector fields are at cell edges or cell centers
    # by checking the dimensions of the x and y fields
    # If the vector components are at cell edges, we need to interpolate to cell centers
    # If the vector components are at cell centers, we don't need to interpolate
    # Create a set with all of the dimensions of the x and y fields
    vector_dims = set(x_fld.dims + y_fld.dims)
    
    # if neither 'i_g' and 'j_g' are present then the vector components are at cell centers
    vector_components_at_cell_center = 'i_g' not in vector_dims and 'j_g' not in vector_dims

    if vector_components_at_cell_center:
        # If the vector components are at cell centers, we don't need to interpolate
        # vec_field is a dictionary with the x and y fields
        vec_field = {'X': x_fld, 'Y': y_fld}
    else:
        # If the vector components are at cell edges, we need to interpolate to cell centers
        # vec_field is a dictionary with the x and y fields
        vec_field = grid.interp_2d_vector({'X': x_fld, 'Y': y_fld},boundary='fill')
        
    # Compute the zonal "e" and meridional components "n" using cos(), sin()
    e_fld = vec_field['X']*coords['CS'] - vec_field['Y']*coords['SN']
    n_fld = vec_field['X']*coords['SN'] + vec_field['Y']*coords['CS']

    return e_fld, n_fld


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def resample_to_latlon(orig_lons, orig_lats, orig_field,
                       new_grid_min_lat, 
                       new_grid_max_lat, 
                       new_grid_delta_lat,
                       new_grid_min_lon, 
                       new_grid_max_lon, 
                       new_grid_delta_lon,
                       radius_of_influence = 120000,
                       fill_value = None, mapping_method = 'bin_average') :
    """Take a field from a source grid and interpolate to a target grid.

    Parameters
    ----------
    orig_lons, orig_lats, orig_field : xarray DataArray or numpy array  :
        the lons, lats, and field from the source grid

	new_grid_min_lat, new_grid_max_lat : float
		latitude limits of new lat-lon grid

    new_grid_delta_lat : float
        latitudinal extent of new lat-lon grid cells in degrees (-90..90)

    new_grid_min_lon, new_grid_max_lon : float
		longitude limits of new lat-lon grid (-180..180)

    new_grid_delta_lon : float
         longitudinal extent of new lat-lon grid cells in degrees

    radius_of_influence : float, optional.  Default 120000 m
        the radius of the circle within which the input data is search for
        when mapping to the new grid

    fill_value : float, optional. Default None
		value to use in the new lat-lon grid if there are no valid values
		from the source grid

  	mapping_method : string, optional. Default 'bin_average'
        denote the type of interpolation method to use.
        options include
            'nearest_neighbor' - Take the nearest value from the source grid
            					 to the target grid
            'bin_average'      - Use the average value from the source grid
								 to the target grid

    RETURNS:
    new_grid_lon_centers, new_grid_lat_centers : ndarrays
    	2D arrays with the lon and lat values of the new grid cell centers

    new_grid_lon_edges, new_grid_lat_edges: ndarrays
    	2D arrays with the lon and lat values of the new grid cell edges

    data_latlon_projection:
    	the source field interpolated to the new grid

    """
    if type(orig_lats) == xr.core.dataarray.DataArray:
        orig_lons_1d = orig_lons.values.ravel()
        orig_lats_1d = orig_lats.values.ravel()

    elif type(orig_lats) == np.ndarray:
        orig_lats_1d = orig_lats.ravel()
        orig_lons_1d = orig_lons.ravel()
    else:
        raise TypeError('orig_lons and orig_lats variable either a DataArray or numpy.ndarray. \n'
                'Found type(orig_lons) = %s and type(orig_lats) = %s' %
                (type(orig_lons), type(orig_lats)))
    
    if type(orig_field) == xr.core.dataarray.DataArray:
        orig_field = orig_field.values
    elif type(orig_field) != np.ndarray and \
         type(orig_field) != np.ma.core.MaskedArray :
        raise TypeError('orig_field must be a type of DataArray, ndarray, or MaskedArray. \n'
                'Found type(orig_field) = %s' % type(orig_field))

    ## Modifications to allow time and depth dimensions (DS, 2023-04-20)    
    # Collapse any non-horizontal dimensions into a single, final dimension:

    # Get shape of orig_lats, then difference with orig_field
    n_horiz_dims=len(orig_lats.shape)
    n_total_dims=len(orig_field.shape)
    n_extra_dims=n_total_dims-n_horiz_dims
    horiz_dim_shape=orig_lats.shape # e.g. [13,90,90]    
    if ( (n_extra_dims>0) & (np.prod(orig_field.shape) > np.prod(horiz_dim_shape) ) ):
        # If there are extra dimensions (and they are meaningful/have len > 1)...
        
        # Check if extra dimensions are at beginning or end of orig_field...
        if orig_field.shape[0]!=orig_lats.shape[0]:
            # ... if at the beginning, collapse and move to end
            extra_dims_at_beginning=True
            extra_dim_shape=orig_field.shape[:n_extra_dims] # e.g. [312,50]
            new_shape=np.hstack([np.prod(extra_dim_shape),\
                                 np.prod(horiz_dim_shape)])          # e.g. from [312,50,13,90,90] to [15600,105300]
            orig_field=orig_field.reshape(new_shape).transpose(1,0) # e.g. from [15600,105300] to [105300,15600]
        else:
            # ... if at the end, just collapse
            extra_dims_at_beginning=False
            extra_dim_shape=orig_field.shape[n_horiz_dims:] #e.g. [50,312]
            new_shape=np.hstack([np.prod(horiz_dim_shape),\
                                 np.prod(extra_dim_shape)]) # e.g. from [13,90,90,50,312] to [105300,15600]
            orig_field=orig_field.reshape(new_shape)
    ##


    # prepare for the nearest neighbor mapping

    # first define the lat lon points of the original data
    orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d,
                                            lats=orig_lats_1d)


   # the latitudes to which we will we interpolate
    num_lats = int((new_grid_max_lat - new_grid_min_lat) / new_grid_delta_lat + 1)
    num_lons = int((new_grid_max_lon - new_grid_min_lon) / new_grid_delta_lon + 1)

    if (num_lats > 0) and (num_lons > 0):
        # linspace is preferred when using floats!

        new_grid_lat_edges_1D =\
            np.linspace(new_grid_min_lat, new_grid_max_lat, num=int(num_lats))
        
        new_grid_lon_edges_1D =\
            np.linspace(new_grid_min_lon, new_grid_max_lon, num=int(num_lons))

        new_grid_lat_centers_1D = (new_grid_lat_edges_1D[0:-1] + new_grid_lat_edges_1D[1:])/2
        new_grid_lon_centers_1D = (new_grid_lon_edges_1D[0:-1] + new_grid_lon_edges_1D[1:])/2

        new_grid_lon_edges, new_grid_lat_edges =\
            np.meshgrid(new_grid_lon_edges_1D, new_grid_lat_edges_1D)
        
        new_grid_lon_centers, new_grid_lat_centers =\
            np.meshgrid(new_grid_lon_centers_1D, new_grid_lat_centers_1D)

        #print(np.min(new_grid_lon_centers), np.max(new_grid_lon_centers))
        #print(np.min(new_grid_lon_edges), np.max(new_grid_lon_edges))
        
        #print(np.min(new_grid_lat_centers), np.max(new_grid_lat_centers))
        #print(np.min(new_grid_lat_edges), np.max(new_grid_lat_edges)) 
        
        # define the lat lon points of the two parts.
        new_grid  = pr.geometry.GridDefinition(lons=new_grid_lon_centers,
                                               lats=new_grid_lat_centers)

        if mapping_method == 'nearest_neighbor':
            data_latlon_projection = \
                    pr.kd_tree.resample_nearest(orig_grid, orig_field, new_grid,
                                                radius_of_influence=radius_of_influence,
                                                fill_value=fill_value)
        elif mapping_method == 'bin_average':
            wf = lambda r: 1

            data_latlon_projection = \
                    pr.kd_tree.resample_custom(orig_grid, orig_field, new_grid,
                                                radius_of_influence=radius_of_influence,
                                                weight_funcs = wf,
                                                fill_value=fill_value)
        else:
            raise ValueError('mapping_method must be nearest_neighbor or bin_average. \n'
                    'Found mapping_method = %s ' % mapping_method)

        ## Modifications to allow time and depth dimensions (DS, 2023-04-20)
        if ( (n_extra_dims>0) & (np.prod(orig_field.shape) > np.prod(horiz_dim_shape) ) ):
        # If there are extra dimensions (and they are meaningful/have len > 1)
            new_horiz_shape=data_latlon_projection.shape[:2]
            if extra_dims_at_beginning:
                # If the extra dimensions were originally at the beginning, move back...
                data_latlon_projection=data_latlon_projection.transpose(2,0,1)
                # ... and unstack the additional dimensions
                final_shape=np.hstack([extra_dim_shape,new_horiz_shape])
                data_latlon_projection=data_latlon_projection.reshape(final_shape)
            else:
                # If the extra dimensions were originally at the end, just unstack
                final_shape=np.hstack([extra_dim_shape,new_horiz_shape])
                data_latlon_projection=data_latlon_projection.reshape(final_shape)
        ##
        
    else:
        raise ValueError('Number of lat and lon points to interpolate to must be > 0. \n'
                'Found num_lats = %d, num lons = %d' % (num_lats,num_lons))

    return new_grid_lon_centers, new_grid_lat_centers,\
           new_grid_lon_edges, new_grid_lat_edges,\
           data_latlon_projection
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
