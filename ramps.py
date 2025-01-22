##############################
###      FlatsimCorr       ###
###          ---           ###
###    L. Marconato 2024   ###
##############################

import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob
#from tqdm import tqdm
import datetime
#from scipy.linalg import lstsq
#from scipy.stats import linregress
#from scipy.signal import lombscargle

# Locals
from . import utils

# Some figure parameters...
import matplotlib
if os.environ["TERM"].startswith("screen"): # cluster
    matplotlib.use('Agg')
else:  # laptop
    matplotlib.rc('font', **{'family':'Arial', 'weight':'medium', 'size':8, 'style':'normal'})
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['figure.dpi'] = 300

class ramps(object):
    '''
    Args:
        * name      : Name of the FLATSIM dataset (e.g. A059SUD or D137NORD)

    Kwargs:
        * datadir         : directory where to look for the TS and AUX flatsim data (default, current dir)
        * savedir         : directory where to store computation files and outputs (default, current dir)
        * gim_list        : set the list of possible ionospheric models to read
                                (default, IGS_IGS, IGS_JPL, IGS_CODE, IGS_ESA, IGS_UPC, JPLD, RGP)
        * wavelength      : wavelength of the radar in meters (default, 0.0556 for Sentinel-1)
        * look_unw        : multilooking factor of the time-series (default, 8rlks)
        * convert_to_mmkm : if True, converts all loaded ramps from rad/px to mm/km (default, False)
                                -> to use it, check wavelength and look_unw are correctly set
        * verbose         : optional, default=True)

    Returns:
        * None
    '''

    def __init__(self, name, datadir='.', savedir='.', gim_list=None, wavelength=0.0556, look_unw=8, convert_to_mmkm=False, verbose=True):

        self.name = name
        self.verbose = verbose

        # Select GIM models to consider
        if gim_list==None:
            self.possible_iono_models = ['IGS_IGS', 'IGS_JPL', 'IGS_CODE', 'IGS_ESA', 'IGS_UPC', 'JPLD', 'RGP']
        else:
            self.possible_iono_models = gim_list

        self.iono_colors = {}
        for model in self.possible_iono_models:
            if '_IGS' in model:
                self.iono_colors[model] = 'tab:blue'
            if '_JPL' in model:
                self.iono_colors[model] = 'tab:purple'
            if '_CODE' in model:
                self.iono_colors[model] = 'tab:green'
            if '_ESA' in model:
                self.iono_colors[model] = 'tab:orange'
            if '_UPC' in model:
                self.iono_colors[model] = 'tab:olive'
            if 'JPLD' in model:
                self.iono_colors[model] = 'tab:pink'
            if 'RGP' in model:
                self.iono_colors[model] = 'tab:brown'

        if self.verbose:
            print("---------------------------------")
            print("Initialize FLATSIM ramps data set {}".format(self.name))
            print("---------------------------------")

        # Create output directory
        self.savedir = os.path.join(savedir, self.name)
        # if not os.path.isdir(self.savedir):
        #     os.makedirs(self.savedir)

        # Look for data directories
        if self.verbose:
            print("    Try to look for data directories with corresponding name...")

        self.datadir = datadir
        dirs = [ f.path for f in os.scandir(self.datadir) if f.is_dir() ]

        self.ts_dir = None
        self.aux_dir = None

        self.wavelength = wavelength
        self.look_unw = look_unw

        if convert_to_mmkm:
            self.ramp_factor = self.wavelength * 1e3/(4*np.pi) * 2150/(self.look_unw/8)/250 
            self.ramp_unit = 'mm/km'
        else:
            self.ramp_factor = 1.
            self.ramp_unit = 'rad/px'
        if self.verbose:
            print('Conversion factor applied on ramps set to', self.ramp_factor)

        for dir in dirs:
            if self.name in dir and 'TS' in dir:
                if self.verbose:
                    print(dir)
                    print("         -> Set as TS directory")
                self.ts_dir = dir

            if self.name in dir and 'DAUX' in dir:
                if self.verbose:
                    print(dir)
                    print("         -> Set as AUX directory")
                self.aux_dir = dir

        self.az_ramps = {}
        self.ra_ramps = {}
        self.sig_ramps = {}

########## FIND A WAY TO INHERIT THIS FROM FLATSIM CLASS ? ##########

    def parseMeta(self, keyword):
        '''
        Read the file 'CNES_MV-LOS_radar_8rlks.meta' to extract the metadata associated to a keyword.

        Args:
            * keyword     : information to extract in the .meta file

        Returns:
            * None
        '''

        self.file_meta = os.path.join(self.ts_dir, f'CNES_MV-LOS_radar_{self.look_unw}rlks.meta')
        if not os.path.isfile(self.file_meta):
            sys.exit("CNES_MV-LOS_radar_8rlks.meta not found")

        with open(self.file_meta) as file_object:
            for line in file_object:
                if keyword in line:
                    return line.rstrip().split(' ')[1:]

    def getAcquisitionTime(self):
        '''
        Find the min, max and mean acquisition times from the names of the master's SAFE (in .meta file)
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Load the acquisition time from the TS cube .meta file (mean of the acquisition times of the master's .SAFE)")

        # Get names of de master SAFE
        temp = self.parseMeta('Super_master_SAR_image_ID')[0]
        
        if temp == 'burst':
            print("    /!\ No SAFE name in .meta -> we use 'Reference_date' keyword")
            time_iso = self.parseMeta('Reference_date')[0]
            dt = datetime.datetime.fromisoformat(time_iso[:-1])
            time = dt.hour + dt.minute/60 + dt.second/3600
            self.min_time_utc = time
            self.max_time_utc = time
            self.mean_time_utc = time

        else:
            safes = temp.split('-')[::2]
        
            # Extract strat and stop times of acquisition from SAFE name
            times = [safe.split('T')[1:3] for safe in safes]
            times = sum(times, [])
            times = [time.split('_')[0] for time in times] 

            # Convert to decimal hours
            times = [int(time[:2])+int(time[2:4])/60 + int(time[4:])/3600 for time in times]

            # /!\ If acquisition spans midnight
            if times[0] > times[-1]:
                print('/!\ The acquisition is spanning midnight: mean_time_utc may be wrong!')
        
            self.min_time_utc = min(times)
            self.max_time_utc = max(times)
            self.mean_time_utc = np.mean(times)

        if self.verbose:
            print(f"        -> acquisition time UTC: {self.mean_time_utc:.3f}h (from {self.min_time_utc:.3f}h to {self.max_time_utc:.3f}h)")

        return
    
############################################################

    def loadDataRamps(self, remove_outliers=False, sig_thres=20., zero_median=False, plot=False):
        '''
        Read (and plot) the files containing the (az and ra) ramps inverted before the time-series inversion.

        Kwargs:
            * remove_outliers : number of standard deviations to filter outliers
            * sig_thresh      : threshold on sigma to remove bad ramps (with sigma higher than threshold)
            * zero_median     : center the ramps on zero by removing the median (after removing outliers)
            * plot            : Plot the ramps and sigma?

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Load the inverted ramps for each acquisition")

        file_ramp_az_inv = os.path.join(self.aux_dir, 'list_ramp_az_inverted_img.txt')
        if not os.path.isfile(file_ramp_az_inv):
            sys.exit("list_ramp_az_inverted_img.txt not found")
        else:
            self.az_ramps['Data'] = np.loadtxt(file_ramp_az_inv, usecols=1) * self.ramp_factor
            self.dates = np.loadtxt(file_ramp_az_inv, usecols=-1)     ### /!\ IMPROVE ###
            self.dates_decyr = np.loadtxt(file_ramp_az_inv, usecols=0)      ### /!\ IMPROVE ###
            self.Ndates = len(self.dates_decyr)                             ### /!\ IMPROVE ###

        file_ramp_ra_inv = os.path.join(self.aux_dir, 'list_ramp_ra_inverted_img.txt')
        if not os.path.isfile(file_ramp_ra_inv):
            sys.exit("list_ramp_ra_inverted_img.txt not found")
        else:
            self.ra_ramps['Data'] = np.loadtxt(file_ramp_ra_inv, usecols=1) * self.ramp_factor
            
        file_ramp_sigma_inv = os.path.join(self.aux_dir, 'list_ramp_sigma_inverted_img.txt')
        if not os.path.isfile(file_ramp_sigma_inv):
            sys.exit("list_ramp_sigma_inverted_img.txt not found")
        else:
            self.sig_ramp_inv = np.loadtxt(file_ramp_sigma_inv, usecols=1)  * self.ramp_factor

        self.az_ramps['Data'][self.az_ramps['Data'] == 0.] = np.nan
        self.ra_ramps['Data'][self.ra_ramps['Data'] == 0.] = np.nan
        self.sig_ramp_inv[self.sig_ramp_inv == 0.] = np.nan

        self.az_ramps['Data'][self.sig_ramp_inv > sig_thres * self.ramp_factor] = np.nan
        self.ra_ramps['Data'][self.sig_ramp_inv > sig_thres * self.ramp_factor] = np.nan

        if remove_outliers:
            self.az_ramps['Data'] = utils.remove_outliers(self.az_ramps['Data'], sigma=remove_outliers)
            self.ra_ramps['Data'] = utils.remove_outliers(self.ra_ramps['Data'], sigma=remove_outliers)

        if zero_median:
            self.az_ramps['Data'] -= np.nanmedian(self.az_ramps['Data'])
            self.ra_ramps['Data'] -= np.nanmedian(self.ra_ramps['Data'])

        if plot:
            fig, axs = plt.subplots(3, sharex=True)

            axs[0].plot(self.dates_decyr, self.az_ramps['Data'], c='b', label='Azimuth ramp')
            axs[1].plot(self.dates_decyr, self.ra_ramps['Data'], c='r', label='Range ramp')
            axs[2].plot(self.dates_decyr, self.sig_ramp_inv, c='grey', label='Sigma')

            #margin = 0.005
            #axs[0].set_ylim(np.nanpercentile(self.az_ramp_inv, 1)-margin, np.nanpercentile(self.az_ramp_inv, 99)+margin)
            #axs[1].set_ylim(np.nanpercentile(self.ra_ramp_inv, 1)-margin, np.nanpercentile(self.ra_ramp_inv, 99)+margin)

            [axs[i].legend(frameon=False) for i in range(3)]

            plt.suptitle(self.name+' - data', weight='bold')
            plt.show()

        return
    
    def loadDataRamps_restrict(self, remove_outliers=False, sig_thres=20., zero_median=False, plot=False):
        '''
        Read (and plot) the files containing the (az and ra) ramps inverted before the time-series inversion.

        Kwargs:
            * remove_outliers : number of standard deviations to filter outliers
            * sig_thresh      : threshold on sigma to remove bad ramps (with sigma higher than threshold)
            * zero_median     : center the ramps on zero by removing the median (after removing outliers)
            * plot            : Plot the ramps and sigma?

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Load the inverted ramps for each acquisition")

        file_ramp_az_inv = os.path.join(self.ts_dir, 'list_ramp_az_inverted_img_restrict.txt')
        if not os.path.isfile(file_ramp_az_inv):
            sys.exit("list_ramp_az_inverted_img_restrict.txt not found")
        else:
            self.az_ramps['Data'] = np.loadtxt(file_ramp_az_inv, usecols=1) * self.ramp_factor
            self.dates = np.loadtxt(file_ramp_az_inv, usecols=-1)     ### /!\ IMPROVE ###
            self.dates_decyr = np.loadtxt(file_ramp_az_inv, usecols=0)      ### /!\ IMPROVE ###
            self.Ndates = len(self.dates_decyr)                             ### /!\ IMPROVE ###

        file_ramp_ra_inv = os.path.join(self.ts_dir, 'list_ramp_ra_inverted_img_restrict.txt')
        if not os.path.isfile(file_ramp_ra_inv):
            sys.exit("list_ramp_ra_inverted_img_restrict.txt not found")
        else:
            self.ra_ramps['Data'] = np.loadtxt(file_ramp_ra_inv, usecols=1) * self.ramp_factor
            
        self.sig_ramp_inv = np.zeros_like(self.ra_ramps['Data'])

        self.az_ramps['Data'][self.az_ramps['Data'] == 0.] = np.nan
        self.ra_ramps['Data'][self.ra_ramps['Data'] == 0.] = np.nan
        self.sig_ramp_inv[self.sig_ramp_inv == 0.] = np.nan

        self.az_ramps['Data'][self.sig_ramp_inv > sig_thres] = np.nan
        self.ra_ramps['Data'][self.sig_ramp_inv > sig_thres] = np.nan

        if remove_outliers:
            self.az_ramps['Data'] = utils.remove_outliers(self.az_ramps['Data'], sigma=remove_outliers)
            self.ra_ramps['Data'] = utils.remove_outliers(self.ra_ramps['Data'], sigma=remove_outliers)

        if zero_median:
            self.az_ramps['Data'] -= np.nanmedian(self.az_ramps['Data'])
            self.ra_ramps['Data'] -= np.nanmedian(self.ra_ramps['Data'])

        if plot:
            fig, axs = plt.subplots(3, sharex=True)

            axs[0].plot(self.dates_decyr, self.az_ramps['Data'], c='b', label='Azimuth ramp')
            axs[1].plot(self.dates_decyr, self.ra_ramps['Data'], c='r', label='Range ramp')
            axs[2].plot(self.dates_decyr, self.sig_ramp_inv, c='grey', label='Sigma')

            #margin = 0.005
            #axs[0].set_ylim(np.nanpercentile(self.az_ramp_inv, 1)-margin, np.nanpercentile(self.az_ramp_inv, 99)+margin)
            #axs[1].set_ylim(np.nanpercentile(self.ra_ramp_inv, 1)-margin, np.nanpercentile(self.ra_ramp_inv, 99)+margin)

            [axs[i].legend(frameon=False) for i in range(3)]

            plt.suptitle(self.name+' - data', weight='bold')
            plt.show()

        return

    def loadSETRamps(self, remove_outliers=False, zero_median=False, plot=False):
        '''
        Read (and plot) the files containing the (az and ra) ramps computed from a SET model.

        Kwargs:
            * remove_outliers : number of standard deviations to filter outliers
            * zero_median     : center the ramps on zero by removing the median (after removing outliers)
            * plot     : Plot the ramps and sigma?

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Look for SET ramps")

        set_file = os.path.join(os.path.join(self.ts_dir, f'tilt_range_earthtide.txt'))

        if os.path.isfile(set_file):
            set_dates = list(np.loadtxt(set_file, usecols=0))
            Nmissing = self.Ndates - len(set_dates)

            az_set_temp = np.loadtxt(set_file, usecols=2) * self.ramp_factor
            ra_set_temp = np.loadtxt(set_file, usecols=1) * self.ramp_factor

            if Nmissing > 0:
                print(f'           -> /!\ {Nmissing} dates with no SET data')

                self.az_ramps['SET'] = []
                self.ra_ramps['SET'] = []

                for date in self.dates: # Put Nan where no SET data available
                    if date in set_dates:
                        idx = set_dates.index(date)
                        self.az_ramps['SET'].append(az_set_temp[idx])
                        self.ra_ramps['SET'].append(ra_set_temp[idx])
                            
                    else:
                        self.az_ramps['SET'].append(np.nan)
                        self.ra_ramps['SET'].append(np.nan)

            else:
                self.az_ramps['SET'] = az_set_temp
                self.ra_ramps['SET'] = ra_set_temp

            if remove_outliers:
                self.az_ramps['SET'] = utils.remove_outliers(self.az_ramps['SET'], sigma=remove_outliers)
                self.ra_ramps['SET'] = utils.remove_outliers(self.ra_ramps['SET'], sigma=remove_outliers)

            if zero_median:
                self.az_ramps['SET'] -= np.nanmedian(self.az_ramps['SET'])
                self.ra_ramps['SET'] -= np.nanmedian(self.ra_ramps['SET'])

            if plot:
                fig, axs = plt.subplots(2, sharex=True)

                axs[0].plot(self.dates_decyr, self.az_ramps['SET'], c='b', label='Azimuth ramp')
                axs[1].plot(self.dates_decyr, self.ra_ramps['SET'], c='r', label='Range ramp')

                #margin = 0.005
                #axs[0].set_ylim(np.nanpercentile(self.az_ramp_inv, 1)-margin, np.nanpercentile(self.az_ramp_inv, 99)+margin)
                #axs[1].set_ylim(np.nanpercentile(self.ra_ramp_inv, 1)-margin, np.nanpercentile(self.ra_ramp_inv, 99)+margin)

                [axs[i].legend(frameon=False) for i in range(2)]

                plt.suptitle(self.name+' - SET', weight='bold')
                plt.show()

        else:
            print(f'/!\ {set_file} not found')

        return
    
    def loadOTLRamps(self, remove_outliers=False, zero_median=False, plot=False):
        '''
        Read (and plot) the files containing the (az and ra) ramps computed from a OTL model.

        Kwargs:
            * remove_outliers : number of standard deviations to filter outliers
            * zero_median     : center the ramps on zero by removing the median (after removing outliers)
            * plot     : Plot the ramps and sigma?

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Look for OTL ramps")

        otl_file = os.path.join(os.path.join(self.ts_dir, f'tilt_otl_range_az.txt'))

        if os.path.isfile(otl_file):
            otl_dates = list(np.loadtxt(otl_file, usecols=0))
            Nmissing = self.Ndates - len(otl_dates)

            az_set_temp = np.loadtxt(otl_file, usecols=2) * self.ramp_factor
            ra_set_temp = np.loadtxt(otl_file, usecols=1) * self.ramp_factor

            if Nmissing > 0:
                print(f'           -> /!\ {Nmissing} dates with no OTL data')

                self.az_ramps['OTL'] = []
                self.ra_ramps['OTL'] = []

                for date in self.dates: # Put Nan where no OTL data available
                    if date in otl_dates:
                        idx = otl_dates.index(date)
                        self.az_ramps['OTL'].append(az_set_temp[idx])
                        self.ra_ramps['OTL'].append(ra_set_temp[idx])
                            
                    else:
                        self.az_ramps['OTL'].append(np.nan)
                        self.ra_ramps['OTL'].append(np.nan)

            else:
                self.az_ramps['OTL'] = az_set_temp
                self.ra_ramps['OTL'] = ra_set_temp

            if remove_outliers:
                self.az_ramps['OTL'] = utils.remove_outliers(self.az_ramps['OTL'], sigma=remove_outliers)
                self.ra_ramps['OTL'] = utils.remove_outliers(self.ra_ramps['OTL'], sigma=remove_outliers)

            if zero_median:
                self.az_ramps['OTL'] -= np.nanmedian(self.az_ramps['OTL'])
                self.ra_ramps['OTL'] -= np.nanmedian(self.ra_ramps['OTL'])

            if plot:
                fig, axs = plt.subplots(2, sharex=True)

                axs[0].plot(self.dates_decyr, self.az_ramps['OTL'], c='b', label='Azimuth ramp')
                axs[1].plot(self.dates_decyr, self.ra_ramps['OTL'], c='r', label='Range ramp')

                #margin = 0.005
                #axs[0].set_ylim(np.nanpercentile(self.az_ramp_inv, 1)-margin, np.nanpercentile(self.az_ramp_inv, 99)+margin)
                #axs[1].set_ylim(np.nanpercentile(self.ra_ramp_inv, 1)-margin, np.nanpercentile(self.ra_ramp_inv, 99)+margin)

                [axs[i].legend(frameon=False) for i in range(2)]

                plt.suptitle(self.name+' - OTL', weight='bold')
                plt.show()

        else:
            #print(f'/!\ {otl_file} not found')
            pass

        return

    def loadIonoRamps(self, models=None, remove_outliers=False, zero_median=False, plot=False):
        '''
        Read (and plot) the files containing the (az and ra) ramps computed from TEC models.

        Kwargs:
             * models : to load one ramps for one specific model, else all the models found are loaded
                       (currently supported : IGS_ESA, IGS_JPL, IGD_CODE, IGS_ESA, IGS_UPC, JPLD and RGP)
            * remove_outliers : number of standard deviations to filter outliers
            * zero_median     : center the ramps on zero by removing the median (after removing outliers)
            * plot            : Plot the ramps and sigma?

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Look for ramps computed using ionosphere models")

        if models is not None:
            if not isinstance(model, list):    # Check if one or more models are asked
                models = [models]
            if not set(models).issubset(self.possible_iono_models): 
                sys.exit('Asked ionospheric models are not supported')
        else:                              # Do all models
            models = self.possible_iono_models

        if plot:
            fig, axs = plt.subplots(3, sharex=True)

        for model in models:
            # Choose the right directory
            if model.startswith('IGS'):
                modeldir = 'iono_igs'
                modelkey = model[4:]
            else:
                modeldir = 'iono_'+model.lower()
                modelkey = model
            
            az_file = os.path.join(os.path.join(self.savedir, modeldir, f'list_ramp_az_{model}.txt'))
            ra_file = os.path.join(os.path.join(self.savedir, modeldir, f'list_ramp_ra_{model}.txt'))
            sig_file = os.path.join(os.path.join(self.savedir, modeldir, f'list_ramp_sigma_{model}.txt'))

            if os.path.isfile(az_file) and os.path.isfile(ra_file) and os.path.isfile(sig_file):
                
                self.az_ramps[modelkey] = np.loadtxt(az_file, usecols=1) * self.ramp_factor
                self.ra_ramps[modelkey] = np.loadtxt(ra_file, usecols=1) * self.ramp_factor
                self.sig_ramps[modelkey] = np.loadtxt(sig_file, usecols=1) * self.ramp_factor

                if len(self.az_ramps[modelkey]) > self.Ndates: # Remove the extra date(s) from the iono arrays
                    # Find the index
                    iono_dates = np.loadtxt(az_file, usecols=-1)
                    idx = np.where(~np.isin(iono_dates, self.dates))
                    # Remove
                    self.az_ramps[modelkey] = np.delete(self.az_ramps[modelkey], idx)
                    self.ra_ramps[modelkey] = np.delete(self.ra_ramps[modelkey], idx)
                    self.sig_ramps[modelkey] = np.delete(self.sig_ramps[modelkey], idx)
                    #print(f"   /!\ {len(idx)} dates removed from iono (doesn't exist in data)")

                if remove_outliers:
                    self.az_ramps[modelkey]= utils.remove_outliers(self.az_ramps[modelkey], sigma=remove_outliers)
                    self.ra_ramps[modelkey] = utils.remove_outliers(self.ra_ramps[modelkey], sigma=remove_outliers)
                    self.sig_ramps[modelkey] = utils.remove_outliers(self.sig_ramps[modelkey], sigma=10)

                if zero_median:
                    self.az_ramps[modelkey] -= np.nanmedian(self.az_ramps[modelkey])
                    self.ra_ramps[modelkey] -= np.nanmedian(self.ra_ramps[modelkey])

                if plot:
                    axs[0].plot(self.dates_decyr, self.az_ramps[modelkey], alpha=0.7, lw=0.8, label=model)
                    axs[1].plot(self.dates_decyr, self.ra_ramps[modelkey], alpha=0.7, lw=0.8, label=model)
                    axs[2].plot(self.dates_decyr, self.sig_ramps[modelkey], alpha=0.7, lw=0.8, label=model)

                # margin = 0.005
                # axs[0].set_ylim(np.percentile(self.ramp_az_inv, 1)-margin, np.percentile(self.ramp_az_inv, 99)+margin)
                # axs[1].set_ylim(np.percentile(self.ramp_ra_inv, 1)-margin, np.percentile(self.ramp_ra_inv, 99)+margin)

        if plot:
            [axs[i].legend(frameon=False, ncols=3) for i in range(3)]
            axs[0].set_ylabel('Azimuth ramps')
            axs[1].set_ylabel('Range ramps')
            axs[2].set_ylabel('Sigma')
            plt.suptitle(self.name, weight='bold')
            plt.show()

        return
    
    def computeIonoMedian(self, models=None):
        '''
        Compute the median of all (default) or a selection of GIM-derived ramps.

        Kwargs:
             * models : list of models to use (if None use self.possible_iono_models)

        Returns:
            * None
        '''

        if models is not None:
            if not isinstance(models, list):    # Check if one or more models are asked
                models = [models]
            if not set(models).issubset(self.possible_iono_models): 
                sys.exit(f'Asked ionospheric models are not supported, choose among: {self.possible_iono_models}')
        else:                              # Do all models
            models = self.possible_iono_models

        #print('Median computed on: ', models)

        all_ra_ramps = np.zeros((len(models), self.Ndates))
        all_az_ramps = np.zeros((len(models), self.Ndates))

        for i in range(len(models)):
            modelkey = utils.iono2key(models[i])

            if modelkey in list(self.ra_ramps.keys()):
                all_ra_ramps[i,:] = self.ra_ramps[modelkey]
            else:
                all_ra_ramps[i,:] = np.nan
            if modelkey in list(self.az_ramps.keys()):
                all_az_ramps[i,:] = self.az_ramps[modelkey]
            else:
                all_az_ramps[i,:] = np.nan

        #all_ra_ramps[all_ra_ramps == 0] = np.nan
        #all_az_ramps[all_az_ramps == 0] = np.nan

        self.ra_ramps['Iono median'] = np.nanmedian(all_ra_ramps, axis=0)
        self.az_ramps['Iono median'] = np.nanmedian(all_az_ramps, axis=0)

    def analysisIonoSET(self, plot=False, saveplot=False, models=None, itrf_ra=None, itrf_az=None, min_date=None, max_date=None):
        '''
        Computes and plot the time-series of (range ans azimuth) ramps:
            - before/after correction for Solid Earth Tides (SET)
            - before/after correction for ionosphere (with possibly several TEC models)
            - before/after a linear (and seasonal) fit in the corrected ramp time-series

        Kwargs:
            * saveplot : save the plots?
            * models : list of models to use (if None use self.possible_iono_models)
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date
            * itrf_ra : plot a ramp rate in range
            * itrf_az : plot a ramp rate in azimuth

        Returns:
            * two array containing the standard deviations after each correction for range and azimuth ramps, respectively
        '''

        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        rates = {}
        sigmas = {}
        #r_squared = {}

        modelkeys = []
        colors = []
        for model in models:
            temp_key = utils.iono2key(model)
            if temp_key in list(self.ra_ramps.keys()):
                modelkeys.append(temp_key)
                colors.append(self.iono_colors[model])
                R['Range'][temp_key] = self.ra_ramps[temp_key]
                R['Azimuth'][temp_key] = self.az_ramps[temp_key]

        for type in ['Range', 'Azimuth']:
            dicR = R[type]

            fig, axs = plt.subplots(3, 2, sharex='col', width_ratios=[3, 1], figsize=(8,6))

            ### Left: Time-series ###

            axs[0,0].plot(self.dates_decyr, dicR['Data'], c='k', linewidth=0.5, label='Data')
            axs[0,0].plot(self.dates_decyr, dicR['SET'], c='r', linewidth=1, alpha=0.8, label='SET')

            axs[1,0].plot(self.dates_decyr, dicR['Data']-dicR['SET'], c='k', linewidth=0.5, ls=':', label='Data - SET')
            filtered = utils.sliding_median(self.dates_decyr, dicR['Data']-dicR['SET'])
            axs[1,0].plot(self.dates_decyr, filtered, c='k', linewidth=1, label='Data - SET filtered')

            for i in range(len(modelkeys)):
                axs[1,0].plot(self.dates_decyr, dicR[modelkeys[i]], linewidth=1, alpha=0.8, label=modelkeys[i], color=colors[i])
            
            corrected = (dicR['Data'] - dicR['SET'] - dicR['IONO median']) # with median of iono models
            cst, vel = utils.linear_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
            cst2, vel2, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
            fit = cst+self.dates_decyr*vel
            fit2 = cst2+self.dates_decyr*vel2+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
            axs[2,0].plot(self.dates_decyr, corrected, c='k', linewidth=0.5, label='Data - SET - IONO (median)')
            axs[2,0].plot(self.dates_decyr, fit, c='tab:orange', linewidth=1, alpha=0.8, label=f'Linear fit: {vel:.1e}')
            axs[2,0].plot(self.dates_decyr, fit2, c='gold', linewidth=1, alpha=0.8, label=f'Linear and seasonal fit: {vel2:.1e}')
            if itrf_ra is not None and type=='Range':
                axs[2,0].plot(self.dates_decyr, (self.dates_decyr-self.dates_decyr[int(len(self.dates_decyr)/2)])*itrf_ra, 
                              c='k', ls='--', linewidth=2, alpha=1, label=f'ITRF: {itrf_ra:.1e}', zorder=0)
            if itrf_az is not None and type=='Azimuth':
                axs[2,0].plot(self.dates_decyr, (self.dates_decyr-self.dates_decyr[int(len(self.dates_decyr)/2)])*itrf_az, 
                              c='k', ls='--', linewidth=2, alpha=1, label=f'ITRF: {itrf_az:.1e}', zorder=0)

            ### Right: Scatter and histo ###

            axs[0,1].scatter(dicR['Data'], dicR['SET'], marker='.', linewidth=0, s=10, color='r')

            for i in range(len(modelkeys)):
                #axs[1,1].scatter(dicR['Data']-dicR['SET'], dicR[models[i]], marker='.', linewidth=0, s=10, alpha=0.4)
                axs[1,1].scatter(filtered, dicR[modelkeys[i]], marker='.', linewidths=0, s=10, alpha=0.4, color=colors[i])

            axs[2,1].hist(corrected-fit, bins=11, color='tab:orange', alpha=0.6)
            axs[2,1].hist(corrected-fit2, bins=11, color='gold', alpha=0.6)

            ### Limits ###

            xmin = min(axs[0,1].get_xlim()[0], axs[1,1].get_ylim()[0])
            xmax = max(axs[0,1].get_xlim()[1], axs[1,1].get_ylim()[1])

            for i in range(axs.shape[0]):
                axs[i,1].set_xlim(xmin, xmax)
                axs[i,0].set_ylim(xmin, xmax)
                axs[i,1].yaxis.tick_right()
                axs[i,1].yaxis.set_label_position("right")
                axs[i,0].set_ylabel('$d\phi/dx$')
                if i < axs.shape[0]-1: # Only for scatter plots
                    axs[i,1].set_ylim(xmin, xmax)
                    axs[i,1].plot([xmin, xmax], [xmin, xmax], ls='--', c='k', lw=1)
                    axs[i,0].legend(frameon=False, ncols=4)
                else: # Only for histo
                    axs[i,1].axvline(ls='--', c='k', lw=1)
                    axs[i,0].legend(frameon=False, ncols=2)

            ### Scatter and hist axes labels ###
            axs[0, 1].set_xlabel('$d\phi/dx$ data')
            axs[0, 1].set_ylabel('$d\phi/dx$ SET')
            axs[1, 1].set_xlabel('$d\phi/dx$ data - SET filtered')
            axs[1, 1].set_ylabel('$d\phi/dx$ IONO')
            axs[2, 1].set_xlabel('$d\phi/dx$ data - SET - IONO - fit')
            axs[2, 1].set_ylabel('N')

            ### Indicate R2 ###
            r_val1 = utils.R2(dicR['Data'], dicR['SET'])
            axs[0, 1].text(0.95, 0.05, f"$R^2$ = {r_val1:.2f}", transform=axs[0, 1].transAxes, va='bottom', ha='right')
            r_val2 = utils.R2(filtered, dicR['IONO median'])
            axs[1, 1].text(0.95, 0.05, f"$R^2$ = {r_val2:.2f}", transform=axs[1, 1].transAxes, va='bottom', ha='right')

            ### Indicate RMSE ###
            axs[0, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(dicR['Data']-dicR['SET']):.2e}", transform=axs[0, 1].transAxes, va='top')
            axs[1, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(filtered-dicR['IONO median']):.2e}", transform=axs[1, 1].transAxes, va='top')
            #axs[2, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(corrected-fit):.2e}", transform=axs[2, 1].transAxes, va='top')
            axs[2, 1].text(0.05, 0.95, f"RMSE =", transform=axs[2, 1].transAxes, va='top')
            axs[2, 1].text(0.05, 0.85, f"{utils.RMSE(corrected-fit):.2e}", color="tab:orange", weight='bold', transform=axs[2, 1].transAxes, va='top')
            axs[2, 1].text(0.05, 0.75, f"{utils.RMSE(corrected-fit2):.2e}", color="gold", weight='bold', transform=axs[2, 1].transAxes, va='top')

            fig.suptitle(f'{self.name} - {type} ramps', weight='bold')
            plt.tight_layout()

            if saveplot:
                if not os.path.isdir('AnalysisIonoSET'):
                    os.makedirs('AnalysisIonoSET')
                plt.savefig(f'AnalysisIonoSET/{self.name}_{type}.png')
            elif plot:
                plt.show()
            plt.close()

            # Save sigmas
            sigmas[type] = [np.nanstd(dicR['Data']), np.nanstd(dicR['Data']-dicR['SET']), np.nanstd(corrected), np.nanstd(corrected-fit), np.nanstd(corrected-fit2)]
            
            # Save R2
            #r_squared[type] = [utils.R2(self.dates_decyr, dicR['Data']), utils.R2(self.dates_decyr, dicR['Data']-dicR['SET']), utils.R2(self.dates_decyr, corrected), utils.R2(self.dates_decyr, corrected-fit)]
            
            # Save rates
            rates[type] = np.zeros(4)
            _, rates[type][0] = utils.linear_fit(self.dates_decyr, dicR['Data'], min_date=min_date, max_date=max_date)
            _, rates[type][1] = utils.linear_fit(self.dates_decyr, dicR['Data']-dicR['SET'], min_date=min_date, max_date=max_date)
            rates[type][2] = vel
            rates[type][3] = vel2

        return rates['Range'], rates['Azimuth'], sigmas['Range'], sigmas['Azimuth']
    
    def analysisIonoSETOTL(self, plot=False, saveplot=False, models=None, itrf_ra=None, itrf_az=None, min_date=None, max_date=None):
        '''
        Computes and plot the time-series of (range ans azimuth) ramps:
            - before/after correction for Solid Earth Tides (SET)
            - before/after correction for ionosphere (with possibly several TEC models)
            - before/after a linear (and seasonal) fit in the corrected ramp time-series

        NB: modified wrt analysisIonoSET (each correction is compared to the data corrected from the other corrections in order to better see the correlation)

        Kwargs:
            * saveplot : save the plots?
            * models : list of models to use (if None use self.possible_iono_models)
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date
            * itrf_ra : plot a ramp rate in range
            * itrf_az : plot a ramp rate in azimuth

        Returns:
            * two array containing the standard deviations after each correction for range and azimuth ramps, respectively
        '''

        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            otl_label = ' - OTL'
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            otl_label = ''
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        modelkeys = []
        colors = []
        for model in models:
            temp_key = utils.iono2key(model)
            if temp_key in list(self.ra_ramps.keys()):
                modelkeys.append(temp_key)
                colors.append(self.iono_colors[model])
                R['Range'][temp_key] = self.ra_ramps[temp_key]
                R['Azimuth'][temp_key] = self.az_ramps[temp_key]

        xy = {'Range': 'x', 'Azimuth':'y'}

        for type in xy:
            dicR = R[type]

            fig, axs = plt.subplots(3+do_otl, 2, sharex='col', width_ratios=[3, 1], figsize=(8,(3+do_otl)*2))

            ### Left: Time-series ###

            axs[0,0].plot(self.dates_decyr, dicR['Data']-dicR['OTL']-dicR['IONO median'], c='k', linewidth=0.5, label='Data'+otl_label+' - IONO (median)')
            axs[0,0].plot(self.dates_decyr, dicR['SET'], c='r', linewidth=1, alpha=0.8, label='SET')

            if do_otl:
                axs[do_otl,0].plot(self.dates_decyr, dicR['Data']-dicR['SET']-dicR['IONO median'], c='k', linewidth=0.5, label='Data - SET - IONO (median)')
                axs[do_otl,0].plot(self.dates_decyr, dicR['OTL'], c='dodgerblue', linewidth=1, alpha=0.8, label='OTL')
            
            axs[1+do_otl,0].plot(self.dates_decyr, dicR['Data']-dicR['SET']-dicR['OTL'], c='k', linewidth=0.5, ls=':', label='Data - SET'+otl_label+'')
            filtered = utils.sliding_median(self.dates_decyr, dicR['Data']-dicR['SET']-dicR['OTL'])
            axs[1+do_otl,0].plot(self.dates_decyr, filtered, c='k', linewidth=1, label='Data - SET'+otl_label+' filtered')

            for i in range(len(modelkeys)):
                axs[1+do_otl,0].plot(self.dates_decyr, dicR[modelkeys[i]], linewidth=1, alpha=0.8, label=modelkeys[i], color=colors[i])
            
            corrected = (dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']) # with median of iono models
            cst, vel = utils.linear_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
            cst2, vel2, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
            fit = cst+self.dates_decyr*vel
            fit2 = cst2+self.dates_decyr*vel2+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
            axs[2+do_otl,0].plot(self.dates_decyr, corrected, c='k', linewidth=0.5, label='Data - SET'+otl_label+' - IONO (median)')
            axs[2+do_otl,0].plot(self.dates_decyr, fit, c='tab:orange', linewidth=1, alpha=0.8, label=f'Linear fit: {vel:.1e}')
            axs[2+do_otl,0].plot(self.dates_decyr, fit2, c='gold', linewidth=1, alpha=0.8, label=f'Linear and seasonal fit: {vel2:.1e}')
            if itrf_ra is not None and type=='Range':
                axs[2+do_otl,0].plot(self.dates_decyr, (self.dates_decyr-self.dates_decyr[int(len(self.dates_decyr)/2)])*itrf_ra, 
                                c='k', ls='--', linewidth=2, alpha=1, label=f'ITRF: {itrf_ra:.1e}', zorder=0)
            if itrf_az is not None and type=='Azimuth':
                axs[2+do_otl,0].plot(self.dates_decyr, (self.dates_decyr-self.dates_decyr[int(len(self.dates_decyr)/2)])*itrf_az, 
                                c='k', ls='--', linewidth=2, alpha=1, label=f'ITRF: {itrf_az:.1e}', zorder=0)

            ### Right: Scatter and histo ###

            axs[0,1].scatter(dicR['Data']-dicR['OTL']-dicR['IONO median'], dicR['SET'], marker='.', linewidth=0, s=10, color='r')
            if do_otl:
                axs[do_otl,1].scatter(dicR['Data']-dicR['SET']-dicR['IONO median'], dicR['OTL'], marker='.', linewidth=0, s=10, color='dodgerblue')

            for i in range(len(modelkeys)):
                #axs[1,1].scatter(dicR['Data']-dicR['SET'], dicR[models[i]], marker='.', linewidth=0, s=10, alpha=0.4)
                axs[1+do_otl,1].scatter(filtered, dicR[modelkeys[i]], marker='.', linewidths=0, s=10, alpha=0.4, color=colors[i])

            axs[2+do_otl,1].hist(corrected-fit, bins=11, color='tab:orange', alpha=0.6)
            axs[2+do_otl,1].hist(corrected-fit2, bins=11, color='gold', alpha=0.6)

            ### Limits ###

            xmin = min(axs[0,1].get_xlim()[0], axs[1,1].get_ylim()[0])
            xmax = max(axs[0,1].get_xlim()[1], axs[1,1].get_ylim()[1])

            for i in range(axs.shape[0]):
                axs[i,1].set_xlim(xmin, xmax)
                axs[i,0].set_ylim(xmin, xmax)
                axs[i,1].yaxis.tick_right()
                axs[i,1].yaxis.set_label_position("right")
                axs[i,0].set_ylabel(f'$d\phi/d{xy[type]}$ ({self.ramp_unit})')
                if i < axs.shape[0]-1: # Only for scatter plots
                    axs[i,1].set_ylim(xmin, xmax)
                    axs[i,1].plot([xmin, xmax], [xmin, xmax], ls='--', c='k', lw=1)
                    axs[i,0].legend(frameon=False, ncols=4)
                else: # Only for histo
                    axs[i,1].axvline(ls='--', c='k', lw=1)
                    axs[i,0].legend(frameon=False, ncols=2)

            ### Scatter and hist axes labels ###
            axs[0, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data'+otl_label+f' - IONO (median) ({self.ramp_unit})')
            axs[0, 1].set_ylabel(f'$d\phi/d{xy[type]}$ SET ({self.ramp_unit})')
            if do_otl:
                axs[1, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data - SET - IONO (median) ({self.ramp_unit})')
                axs[1, 1].set_ylabel(f'$d\phi/d{xy[type]}$ OTL ({self.ramp_unit})')
            axs[1+do_otl, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data - SET'+otl_label+f' filtered ({self.ramp_unit})')
            axs[1+do_otl, 1].set_ylabel(f'$d\phi/d{xy[type]}$ IONO ({self.ramp_unit})')
            axs[2+do_otl, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data - SET'+otl_label+f' - IONO - fit ({self.ramp_unit})')
            axs[2+do_otl, 1].set_ylabel('N')

            ### Indicate R2 ###
            r_val1 = utils.R2(dicR['Data']-dicR['OTL']-dicR['IONO median'], dicR['SET'])
            axs[0, 1].text(0.95, 0.05, f"$R^2$ = {r_val1:.2f}", transform=axs[0, 1].transAxes, va='bottom', ha='right')
            if do_otl:
                r_val2 = utils.R2(dicR['Data']-dicR['SET']-dicR['IONO median'], dicR['OTL'])
                axs[1, 1].text(0.95, 0.05, f"$R^2$ = {r_val2:.2f}", transform=axs[1, 1].transAxes, va='bottom', ha='right')
            r_val3 = utils.R2(filtered, dicR['IONO median'])
            axs[1+do_otl, 1].text(0.95, 0.05, f"$R^2$ = {r_val3:.2f}", transform=axs[1+do_otl, 1].transAxes, va='bottom', ha='right')

            ### Indicate RMSE ###
            #axs[0, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(dicR['Data']-dicR['SET']):.2e}", transform=axs[0, 1].transAxes, va='top')
            #axs[1, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(filtered-dicR['IONO median']):.2e}", transform=axs[1, 1].transAxes, va='top')
            #axs[2, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(corrected-fit):.2e}", transform=axs[2, 1].transAxes, va='top')
            axs[2+do_otl, 1].text(0.05, 0.95, f"RMSE =", transform=axs[2+do_otl, 1].transAxes, va='top')
            axs[2+do_otl, 1].text(0.05, 0.85, f"{utils.RMSE(corrected-fit):.2e}", color="tab:orange", weight='bold', transform=axs[2+do_otl, 1].transAxes, va='top')
            axs[2+do_otl, 1].text(0.05, 0.75, f"{utils.RMSE(corrected-fit2):.2e}", color="gold", weight='bold', transform=axs[2+do_otl, 1].transAxes, va='top')

            fig.suptitle(f'{self.name} - {type} ramps', weight='bold')
            plt.tight_layout()

            if saveplot:
                if not os.path.isdir('AnalysisIonoSETOTL'):
                    os.makedirs('AnalysisIonoSETOTL')
                plt.savefig(f'AnalysisIonoSETOTL/{self.name}_{type}.png')
            elif plot:
                plt.show()
            plt.close()

        return

    def analysisIonoSETOTL2(self, plot=False, saveplot=False, models=None, itrf_ra=None, itrf_az=None, min_date=None, max_date=None):
        '''
        NB: new version where the trend and seasonal are also remove for data vs SET/OTL comparison.
            + linear fit only removed from the last panel

        Computes and plot the time-series of (range ans azimuth) ramps:
            - before/after correction for Solid Earth Tides (SET)
            - before/after correction for ionosphere (with possibly several TEC models)
            - before/after a linear (and seasonal) fit in the corrected ramp time-series

        NB: modified wrt analysisIonoSET (each correction is compared to the data corrected from the other corrections in order to better see the correlation)

        Kwargs:
            * saveplot : save the plots?
            * models : list of models to use (if None use self.possible_iono_models)
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date
            * itrf_ra : plot a ramp rate in range
            * itrf_az : plot a ramp rate in azimuth

        Returns:
            * two array containing the standard deviations after each correction for range and azimuth ramps, respectively
        '''

        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            otl_label = ' - OTL'
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            otl_label = ''
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        modelkeys = []
        colors = []
        for model in models:
            temp_key = utils.iono2key(model)
            if temp_key in list(self.ra_ramps.keys()):
                modelkeys.append(temp_key)
                colors.append(self.iono_colors[model])
                R['Range'][temp_key] = self.ra_ramps[temp_key]
                R['Azimuth'][temp_key] = self.az_ramps[temp_key]

        xy = {'Range': 'x', 'Azimuth':'y'}

        for type in xy:
            dicR = R[type]

            fig, axs = plt.subplots(3+do_otl, 2, sharex='col', width_ratios=[3, 1], figsize=(8,(3+do_otl)*2))

            # Compute the fit after all corrections
            corrected = (dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']) # with median of iono models
            cst, vel, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
            fit = cst+self.dates_decyr*vel+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
            
            ### Left: Time-series ###

            axs[0,0].plot(self.dates_decyr, dicR['Data']-dicR['OTL']-dicR['IONO median']-fit, c='k', linewidth=0.5, label='Data'+otl_label+' - IONO (median) - fit')
            axs[0,0].plot(self.dates_decyr, dicR['SET'], c='r', linewidth=1, alpha=0.8, label='SET')

            if do_otl:
                axs[do_otl,0].plot(self.dates_decyr, dicR['Data']-dicR['SET']-dicR['IONO median']-fit, c='k', linewidth=0.5, label='Data - SET - IONO (median) - fit')
                axs[do_otl,0].plot(self.dates_decyr, dicR['OTL'], c='dodgerblue', linewidth=1, alpha=0.8, label='OTL')
            
            axs[1+do_otl,0].plot(self.dates_decyr, dicR['Data']-dicR['SET']-dicR['OTL'], c='k', linewidth=0.5, ls=':', label='Data - SET'+otl_label+'')
            filtered = utils.sliding_median(self.dates_decyr, dicR['Data']-dicR['SET']-dicR['OTL'])
            axs[1+do_otl,0].plot(self.dates_decyr, filtered, c='k', linewidth=1, label='Data - SET'+otl_label+' filtered')

            for i in range(len(modelkeys)):
                axs[1+do_otl,0].plot(self.dates_decyr, dicR[modelkeys[i]], linewidth=1, alpha=0.8, label=modelkeys[i], color=colors[i])
            
            axs[2+do_otl,0].plot(self.dates_decyr, corrected, c='k', linewidth=0.5, label='Data - SET'+otl_label+' - IONO (median)')
            axs[2+do_otl,0].plot(self.dates_decyr, fit, c='tab:green', linewidth=2, alpha=0.8, 
                                 label='Linear + seasonal fit')
                                 #label=f'Ramp rate: {vel:.1e} {self.ramp_unit}/yr\nSeasonal amplitude: {np.sqrt(sin**2+cos**2):.1e} {self.ramp_unit}')
            if itrf_ra is not None and type=='Range':
                axs[2+do_otl,0].plot(self.dates_decyr, (self.dates_decyr-self.dates_decyr[int(len(self.dates_decyr)/2)])*itrf_ra, 
                                c='k', ls='--', linewidth=2, alpha=1, zorder=0, label=f'GNSS ramp rate')#: {itrf_ra:.1e}')
            if itrf_az is not None and type=='Azimuth':
                axs[2+do_otl,0].plot(self.dates_decyr, (self.dates_decyr-self.dates_decyr[int(len(self.dates_decyr)/2)])*itrf_az, 
                                c='k', ls='--', linewidth=2, alpha=1, zorder=0, label=f'GNSS ramp rate')#: {itrf_az:.1e}')

            ### Right: Scatter and histo ###

            axs[0,1].scatter(dicR['Data']-dicR['OTL']-dicR['IONO median']-fit, dicR['SET'], marker='.', linewidth=0, s=10, color='r')
            if do_otl:
                axs[do_otl,1].scatter(dicR['Data']-dicR['SET']-dicR['IONO median']-fit, dicR['OTL'], marker='.', linewidth=0, s=10, color='dodgerblue')

            for i in range(len(modelkeys)):
                #axs[1,1].scatter(dicR['Data']-dicR['SET'], dicR[models[i]], marker='.', linewidth=0, s=10, alpha=0.4)
                axs[1+do_otl,1].scatter(filtered, dicR[modelkeys[i]], marker='.', linewidths=0, s=10, alpha=0.4, color=colors[i])

            axs[2+do_otl,1].hist(corrected-fit, bins=21, color='tab:green', alpha=0.8)

            ### Limits ###

            xmin = min(axs[0,1].get_xlim()[0], axs[1,1].get_ylim()[0])
            xmax = max(axs[0,1].get_xlim()[1], axs[1,1].get_ylim()[1])

            for i in range(axs.shape[0]):
                axs[i,1].set_xlim(xmin, xmax)
                axs[i,0].set_ylim(xmin, xmax)
                axs[i,1].yaxis.tick_right()
                axs[i,1].yaxis.set_label_position("right")
                axs[i,0].set_ylabel(f'$d\phi/d{xy[type]}$ ({self.ramp_unit})')
                if i < axs.shape[0]-1: # Only for scatter plots
                    axs[i,1].set_ylim(xmin, xmax)
                    axs[i,1].plot([xmin, xmax], [xmin, xmax], ls='--', c='k', lw=1)
                    axs[i,0].legend(frameon=False, ncols=4)
                else: # Only for histo
                    axs[i,1].axvline(ls='--', c='k', lw=1)
                    axs[i,0].legend(frameon=False, ncols=2)

            ### Scatter and hist axes labels ###
            axs[0, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data'+otl_label+f' - IONO (median) - fit ({self.ramp_unit})')
            axs[0, 1].set_ylabel(f'$d\phi/d{xy[type]}$ SET ({self.ramp_unit})')
            if do_otl:
                axs[1, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data - SET - IONO (median) - fit ({self.ramp_unit})')
                axs[1, 1].set_ylabel(f'$d\phi/d{xy[type]}$ OTL ({self.ramp_unit})')
            axs[1+do_otl, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data - SET'+otl_label+f' filtered ({self.ramp_unit})')
            axs[1+do_otl, 1].set_ylabel(f'$d\phi/d{xy[type]}$ IONO ({self.ramp_unit})')
            axs[2+do_otl, 1].set_xlabel(f'$d\phi/d{xy[type]}$ data - SET'+otl_label+f' - IONO - fit ({self.ramp_unit})')
            axs[2+do_otl, 1].set_ylabel('N')

            ### Indicate R2 ###
            r_val1 = utils.R2(dicR['Data']-dicR['OTL']-dicR['IONO median']-fit, dicR['SET'])
            axs[0, 1].text(0.95, 0.05, f"$R^2$ = {r_val1:.2f}", transform=axs[0, 1].transAxes, va='bottom', ha='right')
            if do_otl:
                r_val2 = utils.R2(dicR['Data']-dicR['SET']-dicR['IONO median']-fit, dicR['OTL'])
                axs[1, 1].text(0.95, 0.05, f"$R^2$ = {r_val2:.2f}", transform=axs[1, 1].transAxes, va='bottom', ha='right')
            r_val3 = utils.R2(filtered, dicR['IONO median'])
            axs[1+do_otl, 1].text(0.95, 0.05, f"$R^2$ = {r_val3:.2f}", transform=axs[1+do_otl, 1].transAxes, va='bottom', ha='right')

            ### Indicate RMSE ###
            #axs[0, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(dicR['Data']-dicR['SET']):.2e}", transform=axs[0, 1].transAxes, va='top')
            #axs[1, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(filtered-dicR['IONO median']):.2e}", transform=axs[1, 1].transAxes, va='top')
            #axs[2, 1].text(0.05, 0.95, f"RMSE = \n{utils.RMSE(corrected-fit):.2e}", transform=axs[2, 1].transAxes, va='top')
            axs[2+do_otl, 1].text(0.05, 0.95, f"RMSE =", transform=axs[2+do_otl, 1].transAxes, va='top')
            axs[2+do_otl, 1].text(0.05, 0.85, f"{utils.RMSE(corrected-fit):.2e}", transform=axs[2+do_otl, 1].transAxes, va='top')
  
            fig.suptitle(f'{self.name} - {type} ramps', weight='bold')
            plt.tight_layout()

            if saveplot:
                if not os.path.isdir('AnalysisIonoSETOTL'):
                    os.makedirs('AnalysisIonoSETOTL')
                plt.savefig(f'AnalysisIonoSETOTL/{self.name}_{type}.png')
            elif plot:
                plt.show()
            plt.close()

        return

    def computeRampRates(self, models=None, seasonal=True, min_date=None, max_date=None): 
        '''
        Computes the ramp rate:
            - on raw ramp time-series
            - after correction for Solid Earth Tides (SET)
            - after correction for Ocean Tide Loading (OTL) if existing
            - after correction for ionosphere (using the median of several models)
        Ramp rates stored in the self.ra_ramprates and self.az_ramprates dicts
        Uncertainties stored in the self.ra_ramprates_sig and self.az_ramprates_sig dicts

        Kwargs:
            * models : list of models to use to compute de median model (if None use self.possible_iono_models)
            * seasonal : add a seasonal component
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date

        Returns:
            * None
        '''
        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        xy = {'Range': 'x', 'Azimuth':'y'}
        self.ra_ramprates = {}
        self.ra_ramprates_sig = {}
        self.az_ramprates = {}
        self.az_ramprates_sig = {}

        for type in xy:
            dicR = R[type]

            # Fill an array with all the datasets to fit

            # Raw data and SET
            data = {'Data': dicR['Data'],
                    'Data - SET': dicR['Data'] - dicR['SET']
                   }
            
            # OTL if exists
            if do_otl:
                data['Data - OTL'] = dicR['Data'] - dicR['OTL']

            # Iono
            data['Data - IONO'] = dicR['Data'] - dicR['IONO median']
            
            # All corrections
            if do_otl:
                data['Data - ALL'] = dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']
            else:
                data['Data - ALL'] = dicR['Data'] - dicR['SET'] - dicR['IONO median']

            for d in data:

                # perform fit
                if seasonal:
                    _, rate, _, _, rate_sig = utils.linear_seasonal_fit_sigma(self.dates_decyr, data[d], min_date=min_date, max_date=max_date)
                else:
                    _, rate, rate_sig = utils.linear_fit_sigma(self.dates_decyr, data[d], min_date=min_date, max_date=max_date)

                # Store results
                if type == 'Range':
                    self.ra_ramprates[d], self.ra_ramprates_sig[d] = rate, rate_sig
                if type == 'Azimuth':
                    self.az_ramprates[d], self.az_ramprates_sig[d] = rate, rate_sig

        return
    
    def computeSeasAmp(self, models=None, min_date=None, max_date=None):
        '''
        Computes the amplitude of the seasonal term fitted:
            - on raw ramp time-series
            - after correction for Solid Earth Tides (SET)
            - after correction for Ocean Tide Loading (OTL) if existing
            - after correction for ionosphere (using the median of several models)
        Amplitudes stored in the self.ra_seasamp and self.az_seasamp dicts

        Kwargs:
            * models : list of models to use to compute de median model (if None use self.possible_iono_models)
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date

        Returns:
            * None
        '''
        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        xy = {'Range': 'x', 'Azimuth':'y'}
        self.ra_seasamp = {}
        self.az_seasamp = {}

        for type in xy:
            dicR = R[type]

            # Fill an array with all the datasets to fit

            # Raw data and SET
            data = {'Data': dicR['Data'],
                    'Data - SET': dicR['Data'] - dicR['SET']
                   }
            
            # OTL if exists
            if do_otl:
                data['Data - OTL'] = dicR['Data'] - dicR['OTL']

            # Iono
            data['Data - IONO'] = dicR['Data'] - dicR['IONO median']
            
            # All corrections
            if do_otl:
                data['Data - ALL'] = dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']
            else:
                data['Data - ALL'] = dicR['Data'] - dicR['SET'] - dicR['IONO median']

            for d in data:
                # Fit linear + seasonal
                _, _, sin_amp, cos_amp, _ = utils.linear_seasonal_fit_sigma(self.dates_decyr, data[d], min_date=min_date, max_date=max_date)

                # Store results
                if type == 'Range':
                    self.ra_seasamp[d] = np.sqrt(sin_amp**2+cos_amp**2)
                if type == 'Azimuth':
                    self.az_seasamp[d] = np.sqrt(sin_amp**2+cos_amp**2)

        return

    def computeStds(self, models=None):
        '''
        Computes the standard deviation of ramps:
            - on raw ramp time-series
            - after correction for Solid Earth Tides (SET)
            - after correction for Ocean Tide Loading (OTL) if existing
            - after correction for ionosphere (using the median of several models)
        Results stored in the self.ra_std and self.az_std dicts

        Kwargs:
            * models : list of models to use to compute de median model (if None use self.possible_iono_models)

        Returns:
            * None
        '''
        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        xy = {'Range': 'x', 'Azimuth':'y'}
        self.ra_std = {}
        self.az_std = {}

        for type in xy:
            dicR = R[type]

            # Fill an array with all the datasets to fit

            # Raw data and SET
            data = {'Data': dicR['Data'],
                    'Data - SET': dicR['Data'] - dicR['SET']
                   }
            
            # OTL if exists
            if do_otl:
                data['Data - OTL'] = dicR['Data'] - dicR['OTL']

            # Iono
            data['Data - IONO'] = dicR['Data'] - dicR['IONO median']
            
            # All corrections
            if do_otl:
                data['Data - ALL'] = dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']
            else:
                data['Data - ALL'] = dicR['Data'] - dicR['SET'] - dicR['IONO median']

            for d in data:
                # Store results
                if type == 'Range':
                    self.ra_std[d] = np.nanstd(data[d])
                if type == 'Azimuth':
                    self.az_std[d] = np.nanstd(data[d])

        return

    def computeStdReductions(self, models=None, min_date=None, max_date=None):
        '''
        Computes the reduction of standard deviation of ramps when applying:
            - the correction for Solid Earth Tides (SET)
            - the correction for Ocean Tide Loading (OTL) if existing
            - the correction for ionosphere (using the median of several models)
        Results stored in the self.ra_stdred and self.az_stdred dicts

        Kwargs:
            * models : list of models to use to compute de median model (if None use self.possible_iono_models)
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date

        Returns:
            * None
        '''
        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        xy = {'Range': 'x', 'Azimuth':'y'}
        self.ra_stdred = {}
        self.az_stdred = {}

        for type in xy:
            dicR = R[type]

            # Fill an array with all the datasets to fit

            # OTL if exists
            if do_otl:
                # Compute the fit after all corrections
                corrected = dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']
                cst, vel, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
                fit = cst+self.dates_decyr*vel+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
                # Set all the time-series
                data = {}
                data['SET'] = {'ref': dicR['Data'] - dicR['OTL'] - dicR['IONO median'] - fit,
                               'corr':  dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median'] - fit}
                data['OTL'] = {'ref': dicR['Data'] - dicR['SET'] - dicR['IONO median'] - fit,
                               'corr':  dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median'] - fit}
                data['IONO'] = {'ref': dicR['Data'] - dicR['SET'] - dicR['OTL'] - fit,
                                'corr':  dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median'] - fit}
                
            else: # TO DO
                # Compute the fit after all corrections
                corrected = dicR['Data'] - dicR['SET'] - dicR['IONO median']
                cst, vel, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
                fit = cst+self.dates_decyr*vel+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
                # Set all the time-series
                data = {}
                data['SET'] = {'ref': dicR['Data'] - dicR['IONO median'] - fit,
                               'corr':  dicR['Data'] - dicR['SET'] - dicR['IONO median'] - fit}
                data['IONO'] = {'ref': dicR['Data'] - dicR['SET'] - fit,
                                'corr':  dicR['Data'] - dicR['SET'] - dicR['IONO median'] - fit}
                  
            for d in data:
                # Store results
                if type == 'Range':
                    self.ra_stdred[d] = (np.nanstd(data[d]['corr'])-np.nanstd(data[d]['ref']))/np.nanstd(data[d]['ref'])
                if type == 'Azimuth':
                    self.az_stdred[d] = (np.nanstd(data[d]['corr'])-np.nanstd(data[d]['ref']))/np.nanstd(data[d]['ref'])

        return

    def computeStds2(self, relative=False, models=None, min_date=None, max_date=None):
        '''
        NB: modified to remove one correction while keeping the others (like analysisIonoSETOTL2).
        Computes the standard deviation of ramps:
            - on raw ramp time-series
            - after correction for Solid Earth Tides (SET) if existing
            - after correction for Ocean Tide Loadgin (OTL) if existing
            - after correction for ionosphere (using the median of several models)
        Results stored in the self.ra_std and self.az_std dicts

        Kwargs:
            * relative: if True, compute the variation of standard deviation w.r.t.raw data: (std_corr-std_raw)/std_raw
            * models : list of models to use to compute de median model (if None use self.possible_iono_models)
            * min_date  : only fit data after this date
            * max_date  : only fit data before this date

        Returns:
            * None
        '''
        if models is None: # Find all the ionospheric models
            models = self.possible_iono_models
        elif not isinstance(models, list):
            sys.exit('Problem with model list')

        # Define dicts for range et azimuth ramps
        R = {}
        R['Range'] = {}
        R['Azimuth'] = {}
        R['Range']['Data'] = self.ra_ramps['Data']
        R['Azimuth']['Data'] = self.az_ramps['Data']
        R['Range']['SET'] = self.ra_ramps['SET']
        R['Azimuth']['SET'] = self.az_ramps['SET']
        
        if 'OTL' in self.ra_ramps:
            do_otl = 1
            R['Range']['OTL'] = self.ra_ramps['OTL']
            R['Azimuth']['OTL'] = self.az_ramps['OTL']
        else:
            do_otl = 0
            R['Range']['OTL'] = np.zeros_like(self.ra_ramps['Data'])
            R['Azimuth']['OTL'] = np.zeros_like(self.az_ramps['Data'])

        self.computeIonoMedian()
        R['Range']['IONO median'] = self.ra_ramps['Iono median']
        R['Azimuth']['IONO median'] = self.az_ramps['Iono median']

        xy = {'Range': 'x', 'Azimuth':'y'}
        self.ra_std = {}
        self.az_std = {}

        for type in xy:
            dicR = R[type]

            # Fill an array with all the datasets to fit

            # Raw data and SET
            data = {'Data': dicR['Data'],
                    'Data - SET': dicR['Data'] - dicR['SET']
                   }
            
            # OTL if exists
            if do_otl:
                # Compute the fit after all corrections
                corrected = dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median']
                cst, vel, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
                fit = cst+self.dates_decyr*vel+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
                # Set all the time-series
                data = {'Data': dicR['Data'],                                                               # Raw
                        'Data - OTL - IONO - fit': dicR['Data'] - dicR['OTL'] - dicR['IONO median'] - fit,  # for SET
                        'Data - SET - IONO - fit': dicR['Data'] - dicR['SET'] - dicR['IONO median'] - fit,  # for OTL
                        'Data - SET - OTL': dicR['Data'] - dicR['SET'] - dicR['OTL'],                       # for IONO
                        'Data - SET - OTL - IONO - fit': dicR['Data'] - dicR['SET'] - dicR['OTL'] - dicR['IONO median'] - fit,
                        }                                                                                   # All corrections
            else:
                # Compute the fit after all corrections
                corrected = dicR['Data'] - dicR['SET'] - dicR['IONO median']
                cst, vel, sin, cos = utils.linear_seasonal_fit(self.dates_decyr, corrected, min_date=min_date, max_date=max_date)
                fit = cst+self.dates_decyr*vel+sin*np.sin(2*np.pi*self.dates_decyr)+cos*np.cos(2*np.pi*self.dates_decyr)
                # Set all the time-series
                data = {'Data': dicR['Data'],                                                               # Raw
                        'Data - IONO - fit': dicR['Data'] - dicR['IONO median'] - fit,                      # for SET
                        'Data - SET': dicR['Data'] - dicR['SET'],                                           # for IONO
                        'Data - SET - IONO - fit': dicR['Data'] - dicR['SET'] - dicR['IONO median'] - fit,
                        }                                                                                   # All corrections

            for d in data:
                # Store results
                if type == 'Range':
                    if relative:
                        self.ra_std[d] = (np.nanstd(data[d])-np.nanstd(data['Data']))/np.nanstd(data['Data'])
                    else:
                        self.ra_std[d] = np.nanstd(data[d])
                if type == 'Azimuth':
                    if relative:
                        self.az_std[d] = (np.nanstd(data[d])-np.nanstd(data['Data']))/np.nanstd(data['Data'])
                    else:
                        self.az_std[d] = np.nanstd(data[d])

        return

    #######################################################################
    ################################ UTILS ################################
    #######################################################################

    # def iono2key(model):
    #     '''
    #     Basically remove the IGS suffix if it exists.

    #     Args:
    #         * model : full name of model (e.g. 'IGS_JPL' or 'JPLD')
        
    #     Returns:
    #         * The key to access ra_ramps and az_ramps dicts (e.g. 'JPL' or 'JPLD')
    #     '''
    #     if model.startswith('IGS'):
    #         return model[4:]
    #     else:
    #         return model

    # def linear_fit(self, dates, data0, min_date=None, max_date=None):
    #     '''
    #     Fit a linear trend in time.

    #     Args:
    #         * dates : list or 1d array of time positions
    #         * data  : list or 1d array of data to fit (same size as dates)
        
    #     Kwargs:
    #         * min_date  : only fit data after this date
    #         * max_date  : only fit data before this date

    #     Returns:
    #         * constant, slope
    #     '''
    #     data = np.copy(data0)
    #     if min_date is not None:
    #         data[dates<min_date] = np.nan
    #     if max_date is not None:
    #         data[dates>max_date] = np.nan

    #     valid = ~np.isnan(data)
    #     vel,cst,_,_,_ = linregress(dates[valid], data[valid])
    #     return cst, vel

    # def linear_seasonal_fit(self, dates, data0, min_date=None, max_date=None):
    #     '''
    #     Fit a linear trend in time plus a sinusoidal signal of period 1 (year).

    #     Args:
    #         * dates : list or 1d array of time positions
    #         * data  : list or 1d array of data to fit (same size as dates)
        
    #     Kwargs:
    #         * min_date  : only fit data after this date
    #         * max_date  : only fit data before this date

    #     Returns:
    #         * constant, slope, sin amplitude, cos amplitude
    #     '''
    #     data = np.copy(data0)
    #     if min_date is not None:
    #         data[dates<min_date] = np.nan
    #     if max_date is not None:
    #         data[dates>max_date] = np.nan
            
    #     valid = ~np.isnan(data)
    #     A = np.c_[np.ones(np.count_nonzero(valid)), dates[valid], np.sin(2*np.pi*dates[valid]), np.cos(2*np.pi*dates[valid])]
    #     C,_,_,_ = lstsq(A, data[valid])

    #     return C[0], C[1], C[2], C[3]

    # def R2(self, data1, data2):
    #     '''
    #     Computes the r-squared coefficient (determination coefficient) between two datasets.

    #     Args:
    #         * data1 and data2 : two lists or 1d-array with the same size

    #     Returns:
    #         * R^2
    #     '''
    #     valid = ~np.logical_or(np.isnan(data1), np.isnan(data2))
    #     data1_sel = data1[valid]
    #     data2_sel = data2[valid]
    #     _,_,r_val,_,_ = linregress(data1_sel, data2_sel)
    #     return r_val**2

    # def RMSE(self, dif):
    #     '''
    #     Computes the root-mean-square of the a series.

    #     Args:
    #         * dif : array (e.g., difference between two datasets to compute a proper RMSE)

    #     Returns:
    #         * RMSE
    #     '''
    #     return np.sqrt(np.nanmean(np.square(dif)))
    
    # def detrend(x, y):
    #     a = np.c_[x, y]
    #     a = a[~np.isnan(a).any(axis=1)]
    #     lin, cst = np.polyfit(a[:,0], a[:,1], 1)
    #     return y - (cst + np.array(x)*lin)

    # def correlate(x1, y1, x2, y2, window=30/365):

    #     # Remove NaNs
    #     x1 = x1[~np.isnan(y1)]
    #     y1 = y1[~np.isnan(y1)]
    #     x2 = x2[~np.isnan(y2)]
    #     y2 = y2[~np.isnan(y2)]

    #     # Keep only common values
    #     common_values = np.intersect1d(x1, x2)
    #     indices_in_x1 = np.where(np.isin(x1, common_values))[0]
    #     indices_in_x2 = np.where(np.isin(x2, common_values))[0]

    #     x1 = x1[indices_in_x1]
    #     y1 = y1[indices_in_x1]
    #     y2 = y2[indices_in_x2]

    #     N = len(x1)
    #     rho = np.zeros(N)
    #     slope = np.zeros(N)

    #     for i in range(N):
    #         xmin = x1[i]-window/2
    #         xmax = x1[i]+window/2
    #         sel = np.logical_and(x1 > xmin, x1 < xmax)

    #         if np.count_nonzero(sel) >= 5:
    #             rho[i] = np.corrcoef(y1[sel], y2[sel])[0,1]
    #             slope[i],_,_,_,_ = linregress(y2[sel], y1[sel])

    #     return x1, np.where(rho==0., np.nan, rho), np.where(slope==0., np.nan, slope)

    # def dif_ramps(dict_x, dict_y, data1, data2):

        x1 = dict_x[data1]
        y1 = dict_y[data1]
        x2 = dict_x[data2]
        y2 = dict_y[data2]

        # Keep only common values
        common_values = np.intersect1d(x1, x2)
        indices_in_x1 = np.where(np.isin(x1, common_values))[0]
        indices_in_x2 = np.where(np.isin(x2, common_values))[0]

        x1 = x1[indices_in_x1]
        y1 = y1[indices_in_x1]
        y2 = y2[indices_in_x2]

        return x1, y1-y2