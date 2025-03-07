import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

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

def plot_pcolor(ax=None, values=None, lon=None, lat=None, var_name="Variable", vmin=None, vmax=None, cmap='bwr'):
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
    
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # Create the pcolormesh plot
    pcolor = ax.pcolormesh(lon, lat, values, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    
    # Add colorbar
    cbar = plt.colorbar(pcolor, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label(var_name)
    
    # Add labels and title
    ax.set_title(var_name, fontsize=12)
    
    if standalone:
        plt.show()
    
    return pcolor
