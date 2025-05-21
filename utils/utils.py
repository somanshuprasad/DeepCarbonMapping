import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
from .ecco_utils import resample_to_latlon
import matplotlib as mpl
cmap_bwr = mpl.colormaps.get_cmap('bwr')
cmap_magma = mpl.colormaps.get_cmap('magma_r')
cmap_bwr.set_bad(color='gray') # sets "bad" (i.e., NaNs) to be colored in gray
cmap_magma.set_bad(color='gray') # sets "bad" (i.e., NaNs) to be colored in gray


def rotate_vector_field(x_fld, y_fld, cs, sn, manual_interp=True):
    if manual_interp:
        x_interp = 0.5 * (x_fld + np.roll(x_fld, shift=-1, axis=-1))  # shift in i (X)
        y_interp = 0.5 * (y_fld + np.roll(y_fld, shift=-1, axis=-2))  # shift in j (Y)
    else:
        x_interp = x_fld
        y_interp = y_fld

    # Broadcasting CS and SN to match dimensions if needed
    while cs.ndim < x_interp.ndim:
        cs = np.expand_dims(cs, axis=0)
        sn = np.expand_dims(sn, axis=0)

    # Rotate
    e_fld = x_interp * cs - y_interp * sn
    n_fld = x_interp * sn + y_interp * cs

    return e_fld, n_fld

def conversion(value,variable_name,dxc,dyc,drf,hfacc,time_unit="y"):
    mol_s = ['ADVxTr01','ADVyTr01','ADVrTr01','DFxETr01','DFyETr01','DFrETr01','DFrITr01']
    mol_m3_s = ['cDIC_PIC','respDIC','rDIC_DOC','rDIC_POC','dDIC_PIC']
    mol_m2_s = ['fluxCO2']
    other = ['TRAC01','gDICEpr']
    time_conversion = {"y":3600 * 24 * 365,
                       "d": 3600 * 24,
                       "h": 3600,
                       "s": 1}
    carbon_conversion = 12
    mmol_to_mol = 10**-3
    
    if variable_name in mol_s:
        #  f[x,y,z] * 10^-3 * 12 * (3600 * 24 * 365)
        # print("mol_s ran")
        output = value * mmol_to_mol * carbon_conversion * time_conversion[time_unit]
        
    elif variable_name in mol_m3_s:
        # f[x,y,z] * 10^-3 * DXC[x,y] * DYC[x,y] * DRF[z] * hFacC[x,y,z] * 12 * (3600 * 24 * 365)
        output = value * mmol_to_mol * \
                 dxc * dyc * drf[:, np.newaxis, np.newaxis] * hfacc * \
                 carbon_conversion * \
                 time_conversion[time_unit]
        
    elif variable_name in mol_m2_s:
        # f[x,y,z] * 10^-3 * DXC[x,y] * DYC[x,y] * hFacC[x,y,0] * 12 * (3600 * 24 * 365)
        output = value * mmol_to_mol * \
                 dxc * dyc * hfacc[0] * \
                 carbon_conversion * \
                 time_conversion[time_unit]
        
    elif variable_name == "gDICEpr":
        # f[x,y,z] * 10^-3 * DXC[x,y] * DYC[x,y] * DRF[z] * hFacC[x,y,0] * 12 * (3600 * 24 * 365)
        output = value * mmol_to_mol * \
                 dxc * dyc * drf[0] * hfacc[0] *\
                 carbon_conversion * \
                 time_conversion[time_unit]
        
    elif variable_name == "TRAC01":
        # f[x,y,z] * 10^-3 / 10**5 (petagram conversion)
        output = value * mmol_to_mol/10**5

    else: "not found"
        
    return output

def swap_llc270_vector_tiles(advx_array, advy_array):
    """
    Swaps tiles 8–13 between ADVxTr and ADVyTr arrays in LLC270 compact format.

    Parameters:
    - advx_array: numpy array of shape (depth, 3510, 270), raw ADVxTr data
    - advy_array: numpy array of shape (depth, 3510, 270), raw ADVyTr data

    Returns:
    - advx_fixed: array with tiles 8–13 swapped in from advy_array
    - advy_fixed: array with tiles 8–13 swapped in from advx_array
    """
    tile_size = 270
    depth, ny, nx = advx_array.shape

    assert ny == 13 * tile_size, "Input arrays must be in LLC270 compact format"

    # Split into 13 tiles along axis 1 (rows)
    advx_tiles = [advx_array[:, i*tile_size:(i+1)*tile_size, :] for i in range(13)]
    advy_tiles = [advy_array[:, i*tile_size:(i+1)*tile_size, :] for i in range(13)]

    # Swap tiles 7 to 12 (which are tiles 8 to 13, 0-indexed)
    for i in range(7, 13):
        advx_tiles[i], advy_tiles[i] = advy_tiles[i], advx_tiles[i]

    # Reassemble arrays
    advx_fixed = np.concatenate(advx_tiles, axis=1)
    advy_fixed = np.concatenate(advy_tiles, axis=1)

    return advx_fixed, advy_fixed

def interpolate_cartesian_grid(lon,lat,data,lat_res=0.5,lon_res=0.5):
    # Interpolating the curvilinear line to the cartesian plane
    new_grid_delta_lat, new_grid_delta_lon = lat_res, lon_res
    new_grid_min_lat, new_grid_max_lat = -90, 90
    new_grid_min_lon, new_grid_max_lon = -180, 180
    new_shape = (int((new_grid_max_lat - new_grid_min_lat)/new_grid_delta_lat),
                 int(((new_grid_max_lon - new_grid_min_lon)/new_grid_delta_lon)))

    interp = np.empty(new_shape)

    lon_cart,lat_cart,_,_,interp = \
                                resample_to_latlon(lon,lat,data,
                                new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,
                                new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
                                fill_value = np.NaN,
                                mapping_method = 'nearest_neighbor',
                                radius_of_influence = 120000)
    return lon_cart,lat_cart,interp

def plot_scatter(lon, lat, data,xlabel="x",ylabel="y",title="title",varname="variable",contrast=1,vmin=None,vmax=None):
    
    # Flatten everything
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    data_flat = data.flatten()
    
    if vmin is None or vmax is None:
        vmax = np.nanmax(np.abs(data_flat)) * contrast
        vmin = -vmax

    # Create mask
    nan_mask = np.isnan(data_flat)

    # Create colormap
    cmap = plt.cm.bwr

    plt.figure(figsize=(12, 6))

    # 1. Plot NaNs in gray
    plt.scatter(lon_flat[nan_mask], lat_flat[nan_mask], color='gray', s=1, label='NaN values')

    # 2. Plot valid data in color
    sc = plt.scatter(
        lon_flat[~nan_mask], lat_flat[~nan_mask],
        c=data_flat[~nan_mask], cmap=cmap,
        s=1, vmin=vmin, vmax=vmax, label=varname
    )

    # Add colorbar and labels
    plt.colorbar(sc, label=varname)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pcolor(lon,lat,values,
    ax=None,
    title=None,cbar_label=None,xlabel=None,ylabel=None,
    vmin=None,vmax=None,contrast=1):
    
    # Prepare colormaps
    cmap_bwr = mpl.colormaps.get_cmap('bwr').copy()
    cmap_magma = mpl.colormaps.get_cmap('magma_r').copy()
    cmap_bwr.set_bad(color='gray')
    cmap_magma.set_bad(color='gray')
    
    if vmax ==None or vmin == None:
        # Detect vmax/vmin and select colormap
        var_max = np.nanmax(values)
        var_min = np.nanmin(values)

        if np.isclose(var_min, 0) and (var_min >= 0):  # mostly positive data
            vmin, vmax = 0, var_max
            cmap = cmap_magma
        else:
            vmax = var_max
            vmin = -vmax
            cmap = cmap_bwr

    # Create axis if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    # Plot
    p = ax.pcolormesh(lon, lat, values, cmap=cmap, vmin=vmin*contrast, vmax=vmax*contrast)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # Colorbar
    cbar = plt.colorbar(p, ax=ax, orientation='vertical', shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)

    if created_fig:
        plt.show()

    return ax

def plot_pcolor_cartopy(ax=None, values=None, lon=None, lat=None, var_name="Variable", vmin=None, vmax=None, cmap='bwr'):
    """
    Plots a single variable on the given axis or creates a new figure if no axis is provided.
    
    Parameters:
    - ax: matplotlib Axes object where the plot will be drawn. If None, a new figure is created.
    - values: 2D numpy array of values to plot.
    - lon: 2D numpy array of longitudes.
    - lat: 2D numpy array of latitudes.
    - var_name: Name of the variable (used in title and colorbar label).
    - vmin: Minimum value for color scale (default: symmetric around 0).
    - vmax: Maximum value for color scale (default: symmetric around 0).
    - cmap: Colormap for the plot (default: 'bwr').
    
    Returns:
    - pcolormesh plot added to the provided axis or a new figure.
    """
    if vmin is None or vmax is None:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        standalone = True
    else:
        standalone = False

    # Create the pcolormesh plot
    pcolor = ax.pcolormesh(lon, lat, values, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(pcolor, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label(var_name)
    
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='gray', facecolor='gray', linewidth=0.2)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.2)
    # ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # Add labels and title
    ax.set_title(var_name, fontsize=12)
    
    if standalone:
        plt.show()
    
    return pcolor

from matplotlib.colors import ListedColormap, hsv_to_rgb

def generate_pastel_cmap(n_colors=30, seed=42):
    """
    Generate a soft pastel colormap similar to Set3 with n distinct colors for clustering cmap

    Args:
        n_colors (int): Number of colors.
        seed (int): Random seed for reproducibility.

    Returns:
        ListedColormap: Pastel-like colormap.
    """
    np.random.seed(seed)
    hues = np.linspace(0, 1, n_colors, endpoint=False)
    np.random.shuffle(hues)  # Randomize order to avoid adjacent similarity

    # Pastel: lower saturation, higher brightness
    saturation = 0.3  # soft colors
    brightness = 0.9  # bright but not glaring

    hsv = np.stack([hues, np.full_like(hues, saturation), np.full_like(hues, brightness)], axis=1)
    rgb = hsv_to_rgb(hsv)
    
    pastel_cmap = ListedColormap(rgb)

    
    pastel_cmap.set_bad(color="gray")
    return pastel_cmap

def transp_tiles(data):
    # transposes the raw imported variables for tiles 8-14
    nx = data.shape[1]
    ny = data.shape[0]
    
    tmp = data[7*nx:,::-1]
    transpo = np.concatenate((tmp[2::3,:].transpose(),tmp[1::3,:].transpose(),tmp[0::3,:].transpose()))
    data_out = np.concatenate((data[:7*nx],np.flipud(transpo[:,:nx]),np.flipud(transpo[:,nx:])))
    return data_out

def transp_tiles_3d(data_3d):
    result = np.zeros(shape=data_3d.shape)
    for depth in range(data_3d.shape[0]):
        result[depth] = transp_tiles(data_3d[depth])
    
    return result

def convert_to_tiles(compact_data, nx=270):
    faces = []
    for i in range(13):
        face = compact_data[..., i*nx:(i+1)*nx, :]
        faces.append(face)
    
    return np.array(faces)

def plot_tiles(data, tsz=270,contrast=1):
    from matplotlib.gridspec import GridSpec
    #### Initiate ####
    iid = [4,3,2,4,3,2,1,1,1,1,0,0,0]
    jid = [0,0,0,1,1,1,1,2,3,4,2,3,4]
    tid = 0
    #### plot ####
    fig = plt.figure(figsize=(10,10))
    gs = GridSpec(5, 5, wspace=.05, hspace=.05)
    vmax = np.nanmax(np.absolute(data))*contrast
    
    for i in range(len(iid)):
        ax = fig.add_subplot(gs[iid[i],jid[i]])
        if i>=7:
            ax.imshow(data[tid:tid+tsz].T,origin='lower',cmap=cmap_bwr,vmax=vmax,vmin=-vmax)
        else:
            ax.imshow(data[tid:tid+tsz],origin='lower',cmap=cmap_bwr,vmax=vmax,vmin=-vmax)
        tid += tsz
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    plt.show()

def imshow(data,contrast=0.1):
    if len(data.shape) >2: data = data[0]
    vmax = np.nanmax(np.absolute(data))*contrast
    plt.imshow(data,cmap=cmap_bwr,vmax=vmax,vmin=-vmax)
    plt.show()
    
def convert_to_faces(compact_data, nx=270):
    """
    Takes (...,3510,270) and converts to 5 faces of dimensions:
    F1: (...,810,270),
    F2: (...,810,270),
    F3: (...,270,270),
    F4: (...,810,270),
    F5: (...,810,270)
    """
    faces = []
    increments = [3,3,1,3,3] # 3rd face is the polar ice cap. rest are 2*1 strips
    i = 0

    for inc in increments:
        face = compact_data[..., i:i+(inc*nx), :] # selects inc*270 where it left off for the previous face
        faces.append(face)
        i += (inc*nx)
    
    return faces

def pad_diff_faces(array,direction="x",nx=270):
    """
    1. accepts the raw array
    2. converts to faces
    3. flips the faces into N-S orientation and stacks the non polar arrays
    4. pads the arrays on the right (x direction) and bottom (y direction)
    5. compute diff in x or y directions
    6. re-flip and re-combine the arrays
    """
    
    faces = convert_to_faces(array)
    
    f1,f2,f3,f4,f5 = faces
    f1 = np.flip(f1,axis=1)
    f2 = np.flip(f2,axis=1)
    f3 = np.flip(f3,axis=1)
    
    if direction == "x":
        # right Padding and stacking the non polar faces 
        non_polar_stack = np.concatenate([f5, f1, f2, f4,f5[...,0:1]], axis=2)
        
        # Right padding the polar face
        pad_slice = f4[:, :1, :] # Grabbing the top row of f4
        pad_slice = np.flip(pad_slice.transpose(0, 2, 1),axis=1) # Transpose to (50, 270, 1) so it can be concatenated along the last column of f3
        f3_padded = np.concatenate([f3, pad_slice], axis=2) # Concatenate to the right side of f3

        # Compute diff along the x-axis (columns)
        non_polar_diff = non_polar_stack[:, :, 1:] - non_polar_stack[:, :, :-1]
        polar_diff = f3_padded[:, :, 1:] - f3_padded[:, :, :-1]
        
    elif direction == "y":
        # right padding and stacking the non polar faces 
        non_polar_stack = np.concatenate([f5, f1, f2, f4], axis=2)
        non_polar_stack = np.concatenate([non_polar_stack,np.zeros(shape=(50,1,270*4))],axis=1)

        # bottom padding the polar face
        f3_padded = np.concatenate([f3,f2[:,0:1,:]],axis=1) # Concatenate the first row of face 2 to the last row of face 3

        # Compute diff along the y-axis (columns)
        non_polar_diff = non_polar_stack[:, 1:, :] - non_polar_stack[:, :-1, :]
        polar_diff = f3_padded[:, 1:, :] - f3_padded[:, :-1, :]
    else:
        print("Plese specify either x or y directions")
        return

    # Return the arrays back into strips
    f5_diff = non_polar_diff[..., :, 0*nx:1*nx]
    f1_diff = non_polar_diff[..., :, 1*nx:2*nx]
    f2_diff = non_polar_diff[..., :, 2*nx:3*nx]
    f4_diff = non_polar_diff[..., :, 3*nx:4*nx]
    
    # Re-flip the first 3 faces matrices
    f1_diff = np.flip(f1_diff, axis=1)
    f2_diff = np.flip(f2_diff, axis=1)
    f3_diff = np.flip(polar_diff, axis=1)
    
    # Reassmble back to the original grid
    reassembled_array = np.concatenate([f1_diff, f2_diff, f3_diff, f4_diff, f5_diff], axis=1)
    return reassembled_array