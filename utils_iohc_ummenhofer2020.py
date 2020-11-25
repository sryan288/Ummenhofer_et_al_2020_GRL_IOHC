"""

Specific utilities for paper: Ummenhofer et al., 2020: IOHC variability ORCA

"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from numpy.polynomial.polynomial import polyval,polyfit
import matplotlib.dates as dates
import xarray as xr
from matplotlib.patches import Rectangle


##############################################################
# adjust depending on whether run on climodes or poseidon
datapath = '/vortexfs1/share/clidex/data/'


################################ functions ############################
# derive heat content over selected depth range
def ohc(temp,mask,weights,dims,base=None,rho=1029,cp=3994):
    # multiply data with mask
    indo = temp*mask['tmaskind'].where(mask['tmaskind']!=0).values

    heat = (indo * weights).sum(dim=dims,skipna=True)*rho*cp
    return heat


# derive heat content aomaly over selected depth range
def ohc_anomaly(temp,mask,weights,dims,base=None,rho=1029,cp=3994):
    # multiply data with mask
    indo = temp*mask['tmaskind'].where(mask['tmaskind']!=0).values

    heat = (indo * weights).sum(dim=dims,skipna=True)*rho*cp
    if base:
        heat = deseason(heat,refperiod=base)
    else:
        heat = deseason(heat)
    return heat


# cut region to reduce load and fix folding line, only for ORCA data
def cut_indo(ds):
    temp1 = ds.sel(x=slice(1250,1450))
    temp2 = ds.sel(x=slice(0,600))
    dummy = xr.concat([temp1,temp2],'x')
    dummy = dummy.sel(y=slice(50,650))
    return dummy

def cut_e3t(ds):
    temp1 = ds.sel(x=slice(1250,1450))
    temp2 = ds.sel(x=slice(0,600))
    dummy = xr.concat([temp1,temp2],'x')
    dummy = dummy.sel(y=slice(300,650)).isel(deptht=slice(0,23))
    return dummy

def mask_2D_percentile(data,pval,level):
    test =np.nanpercentile(data,pval)
    mask= data.values-test
    if level=='above':
        mask[mask>0]=1
        mask[mask<0]=np.nan
    elif level=='below':
        mask[mask>0]=np.nan
        mask[mask<0]=1
    return mask

# annual average
def annual_average(ds,var):
    ds = ds.where(ds[var]>=10).groupby('time_counter.year').mean('time_counter').load()
    return ds

#
#-------------------------------------------------------------------------
# deseason data
def deseason(ds,timevar='time_counter',refperiod=None):
    dummy = timevar + '.month'
    if refperiod:
        if timevar=='time_counter':
            ds = ds.groupby(dummy)-ds.sel(time_counter=slice(*refperiod)).groupby(dummy).mean(timevar)        
        elif timevar=='time':
            ds = ds.groupby(dummy)-ds.sel(time=slice(*refperiod)).groupby(dummy).mean(timevar)
    else:
        ds = ds.groupby(dummy)-ds.groupby(dummy).mean(timevar)
    return ds


#
# ------------------------------------------------------------------
def plot_map(figsize=(10,8),c=np.arange(0,6000,500),ax=None):
    " Input: figsize"
    
#     # bounds to cut out NWA
#     x_bnds, y_bnds = [810,993], [602,758]
    
    # open bathymetry
#     bathy = xr.open_dataset('/vortex/clidex/data/ORCA/mesh_files/mesh_zgr.nc')#.sel(x=slice(*x_bnds),y=slice(*y_bnds))
#     bathy['hdept'] = bathy['hdept']#.where(bathy['hdept']!=0)

    ########## Plotting ####################
    proj = ccrs.PlateCarree()
    if not ax:
        fig,axh = plt.subplots(figsize=figsize,subplot_kw = dict(projection=proj))
    else: axh=ax
        
    # Bathymetry
#     cc= ax.pcolormesh(bathy.nav_lon,bathy.nav_lat,bathy['hdept'][0,:,:],
#                     cmap=plt.get_cmap('Blues',len(c)),vmin=np.min(c),vmax=np.max(c))
    # formatting
    axh.coastlines(color='gray')
    gl = axh.gridlines(crs=proj,draw_labels=True)
    #                             xlocs=range(-0,160,20),ylocs=range(-45,45,15))
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabels_top = False 
    axh.add_feature(cartopy.feature.LAND, color='lightgray')
    axh.set_extent([30,179,-30,30], crs=ccrs.PlateCarree())


    #     if i==0: gl.ylabels_right = False
    gl.ylabels_right = False
#     plt.colorbar(cc)
    if not ax: return fig,axh
    elif ax: return gl


#
#------------------------------------------------------------------------
# If a figure name is defined, save the figure to that file. Otherwise, display the figure on screen.
def finished_plot (fig, fig_name=None, dpi=300):

    if fig_name is not None:
        print('Saving ' + fig_name)
        fig.savefig(fig_name, dpi=dpi,bbox_inches='tight')
    else:
        fig.show()
 
#------------------------------------------------------------------------
# add ipo phase as bar
# Create rectangle x coordinates
def add_ipo_bar(ax):
    import matplotlib.dates as mdates
    indices = xr.open_dataset(datapath + 'obs/climate_indices/indices_noaa_psl_May_13_2020.nc')
    cols = ['dodgerblue','indianred','dodgerblue','indianred']
    text = ['-IPO','+IPO','-IPO','']
    years = [1956,1977,1999,2013,2020]
    for i in range(len(years)-1):
        startTime = indices.sel(Month=slice(str(years[i]) + '-01-01',str(years[i]) + '-01-31'))['Month'].values
        endTime =  indices.sel(Month=slice(str(years[i+1]-1) + '-12-01',str(years[i+1]-1) + '-12-31'))['Month'].values#startTime + timedelta(seconds = 1)
        # convert to matplotlib date representation
        start = mdates.date2num(startTime)
        end = mdates.date2num(endTime)
        width = end - start
        middle = (width/2)+start
        
        ulim = ax.get_ybound()[1]
        llim = ax.get_ybound()[1] - ax.get_ybound()[1]/6 
        rect = Rectangle((start[0], llim), width, ulim, color=cols[i],alpha=0.5)
        ax.text(middle,(ulim-llim)/2+llim,text[i],fontsize=8,fontweight='bold',verticalalignment='center',
               horizontalalignment='center')
        ax.add_patch(rect)
        

#------------------------------------------------------------------------
# Monte-Carlo simulation for significance
def monte_carlo(ds,duration,n,pval,timevar):
    """
    pval: two-tailed pval
    """
    x=0
    mc = np.empty([ds.shape[1],ds.shape[2],n])
    while x<n:
        dummy = np.random.randint(0, len(ds[timevar])-duration, size=1) # have to adjust size so total number of points is always the same
        mc[:,:,x] = ds[int(dummy):int(dummy+duration),::].mean(timevar)
        x=x+1
    # derive percentile
    perc_upper = np.nanpercentile(mc,100-pval,axis=2)
    perc_lower = np.nanpercentile(mc,pval,axis=2)
    return perc_lower,perc_upper        