##############################
###      FlatsimCorr       ###
###          ---           ###
###    L. Marconato 2024   ###
##############################

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from osgeo import gdal
from tqdm import tqdm
import datetime
from scipy.linalg import lstsq
from scipy.interpolate import griddata
import cartopy

import string
import ftplib, urllib
from spacepy import pycdf

# Some figure parameters...
import matplotlib
matplotlib.use('Agg')
plt.rcParams['figure.figsize'] = [8,5]
plt.rcParams['figure.dpi'] = 300

matplotlib.rc('font', **{'family':'Arial', 'weight':'medium', 'size':8, 'style':'normal'})

class flatsim(object):

    '''
    Args:
        * name      : Name of the FLATSIM dataset (e.g. A059SUD or D137NORD)

    Kwargs:
        * datadir   : directory where to look for the TS and AUX flatsim data (default, current dir)
        * savedir   : directory where to store computation files and outputs (default, current dir)
        * utmzone   : UTM zone (optional)
        * lon0      : Longitude of the utmzone (optional)
        * lat0      : Latitude of the utmzone (optional)
        * ellps     : ellipsoid (optional, default='WGS84')
        * verbose   : optional, default=True)

    Returns:
        * None
    '''

    def __init__(self, name, datadir='.', savedir='.', utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        self.name = name
        self.verbose = verbose
        self.utmzone = utmzone
        self.ellps = ellps
        self.lon0 = lon0
        self.lat0 = lat0

        if self.verbose:
            print("---------------------------------")
            print("Initialize FLATSIM time-series data set {}".format(self.name))
            print("---------------------------------")

        # Create output directory
        self.savedir = os.path.join(savedir, self.name)
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

        # Look for data directories
            print("    Try to look for data directories with corresponding name...")

        self.datadir = datadir
        dirs = [ f.path for f in os.scandir(self.datadir) if f.is_dir() ]

        self.ts_dir = None
        self.aux_dir = None

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

        # Exit if data dirs not found
        if self.ts_dir is None or self.aux_dir is None:
                sys.exit("Data directories not found")

        # Store usefull file names, and check they are here

        self.file_meta = os.path.join(self.ts_dir, 'CNES_MV-LOS_radar_8rlks.meta')
        if not os.path.isfile(self.file_meta):
            sys.exit("CNES_MV-LOS_radar_8rlks.meta not found")

        self.file_lut_radar = os.path.join(self.aux_dir, 'CNES_Lut_radar_8rlks.tiff')
        if not os.path.isfile(self.file_lut_radar):
            sys.exit("CNES_Lut_radar_8rlks.tiff not found")

        self.file_los_radar = os.path.join(self.aux_dir, 'CNES_CosENU_radar_8rlks.tiff')
        if not os.path.isfile(self.file_los_radar):
            sys.exit("CNES_CosENU_radar_8rlks.tiff not found")

        # Load image list 
        if self.verbose:
            print("    Load the list of acquisitions")

        self.file_baselines = os.path.join(self.aux_dir, 'baseline.rsc')
        if not os.path.isfile(self.file_baselines):
            sys.exit("baseline.rsc not found")
        else:
            self.dates = np.loadtxt(self.file_baselines, usecols=0, dtype=str)
            self.dates_decyr = np.loadtxt(self.file_baselines, usecols=4)
            self.Ndates = len(self.dates)
            if self.verbose:
                print(f'    {self.Ndates} dates found (from {self.dates_decyr[0]:.2f} to {self.dates_decyr[-1]:.2f})')

        self.file_baseline_tb = os.path.join(self.aux_dir, 'baseline_top_bot.rsc')
        if not os.path.isfile(self.file_baseline_tb):
            sys.exit("baseline_top_bot.rsc not found")
        else:
            self.bperp = np.loadtxt(self.file_baseline_tb, usecols=1)

        return
    
    def plotBaselines(self):
        '''
        Plot the parpendicular baselines as a function of time for all acquisitions.
        '''

        plt.scatter(self.dates_decyr, self.bperp)
        plt.title(self.name, weight='bold')

        return
    
    def loadDateRamps(self, plot=True):
        '''
        Read (and plot) the files containing the (az and ra) ramps inverted before the time-series inversion.

        Kwargs:
            * plot     : Plot the ramps and sigma?

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Load the inverted ramps for each acquisition")

        self.file_ramp_az_inv = os.path.join(self.aux_dir, 'list_ramp_az_inverted_img.txt')
        if not os.path.isfile(self.file_ramp_az_inv):
            sys.exit("list_ramp_az_inverted_img.txt not found")
        else:
            self.ramp_az_inv = np.loadtxt(self.file_ramp_az_inv, usecols=1)

        self.file_ramp_ra_inv = os.path.join(self.aux_dir, 'list_ramp_ra_inverted_img.txt')
        if not os.path.isfile(self.file_ramp_ra_inv):
            sys.exit("list_ramp_ra_inverted_img.txt not found")
        else:
            self.ramp_ra_inv = np.loadtxt(self.file_ramp_ra_inv, usecols=1)
            
        self.file_ramp_sigma_inv = os.path.join(self.aux_dir, 'list_ramp_sigma_inverted_img.txt')
        if not os.path.isfile(self.file_ramp_sigma_inv):
            sys.exit("list_ramp_sigma_inverted_img.txt not found")
        else:
            self.ramp_sig_inv = np.loadtxt(self.file_ramp_sigma_inv, usecols=1)
            
        if plot:
            fig, axs = plt.subplots(3, sharex=True)

            axs[0].plot(self.dates_decyr, self.ramp_az_inv, c='b', label='Azimuth ramp')
            axs[1].plot(self.dates_decyr, self.ramp_ra_inv, c='r', label='Range ramp')
            axs[2].plot(self.dates_decyr, self.ramp_sig_inv, c='grey', label='Sigma')

            margin = 0.005
            axs[0].set_ylim(np.percentile(self.ramp_az_inv, 1)-margin, np.percentile(self.ramp_az_inv, 99)+margin)
            axs[1].set_ylim(np.percentile(self.ramp_ra_inv, 1)-margin, np.percentile(self.ramp_ra_inv, 99)+margin)

            [axs[i].legend(frameon=False) for i in range(3)]

            plt.suptitle(self.name, weight='bold')

        return
    
    def parseMeta(self, keyword):
        '''
        Read the file 'CNES_MV-LOS_radar_8rlks.meta' to extract the metadata associated to a keyword.

        Args:
            * keyword     : information to extract in the .meta file

        Returns:
            * None
        '''

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
        safes = self.parseMeta('Super_master_SAR_image_ID')[0].split('-')[::2]
    
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
    
    def getLatLon(self, plot=False):
        '''
        Read (and plot) the LUT containing lat and lon information in radar geometry.

        Kwargs:
            * plot     : Plot the lat and lon arrays

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Load the Lat-Lon in radar from LUT")
        
        ds = gdal.Open(self.file_lut_radar, gdal.GA_ReadOnly)
        self.width_radar = ds.RasterXSize
        self.length_radar = ds.RasterYSize
        self.lon_radar = ds.GetRasterBand(1).ReadAsArray()
        self.lat_radar = ds.GetRasterBand(2).ReadAsArray()

        if plot:
            fig, axs = plt.subplots(1, 2, sharey=True)#, figsize=(8,4))
            pcm0 = axs[0].imshow(self.lon_radar)
            pcm1 = axs[1].imshow(self.lat_radar)
            plt.colorbar(pcm0, ax=axs[0], label='Longitude (°)', shrink=0.5)
            plt.colorbar(pcm1, ax=axs[1], label='Latitude (°)', shrink=0.5)
            fig.supxlabel('Range')
            fig.supylabel('Azimuth')
            fig.suptitle(self.name, weight='bold')

        return
    
    def getIncidence(self, plot=False, plotENU=False):
        '''
        Read (and plot) the LOS information in radar geometry.

        Kwargs:
            * plot     : Plot the incidence and look angle arrays
            * plot     : Plot the LOS E, N and U arrays

        Returns:
            * None
        '''

        if self.verbose:
            print("\n---------------------------------")
            print("    Load the incidence and heading from losENU file")
        
        ds = gdal.Open(self.file_los_radar, gdal.GA_ReadOnly)
        self.losE = ds.GetRasterBand(1).ReadAsArray()
        self.losN = ds.GetRasterBand(2).ReadAsArray()
        self.losU = ds.GetRasterBand(3).ReadAsArray()

        self.losE[self.losE==0.] = np.nan
        self.losN[self.losN==0.] = np.nan
        self.losU[self.losU==0.] = np.nan

        self.look_angle = np.degrees(np.arctan2(self.losE, self.losN))
        self.incidence = np.degrees(np.arctan2(np.sqrt(self.losE**2+self.losN**2), -self.losU))

        if plotENU:
            fig, axs = plt.subplots(1, 3, sharey=True)
            pcm0 = axs[0].imshow(self.losE)
            pcm1 = axs[1].imshow(self.losN)
            pcm1 = axs[2].imshow(self.losU)
            plt.colorbar(pcm0, ax=axs[0], shrink=0.5)
            plt.colorbar(pcm1, ax=axs[1], shrink=0.5)
            plt.colorbar(pcm1, ax=axs[2], shrink=0.5)
            axs[0].set_title('E')
            axs[1].set_title('N')
            axs[2].set_title('U')
            fig.supxlabel('Range')
            fig.supylabel('Azimuth')
            fig.suptitle(self.name, weight='bold')

        if plot:
            fig, axs = plt.subplots(1, 2, sharey=True)
            pcm0 = axs[0].imshow(self.look_angle)
            pcm1 = axs[1].imshow(self.incidence)
            plt.colorbar(pcm0, ax=axs[0], label='Look angle (°)', shrink=0.5)
            plt.colorbar(pcm1, ax=axs[1], label='Incidence (°)', shrink=0.5)
            fig.supxlabel('Range')
            fig.supylabel('Azimuth')
            fig.suptitle(self.name, weight='bold')

        return
    
    def wrap(self, phase, w=2*np.pi):
        '''
        Wrap the phase.

        Args:
            * phase   : Phase array

        Kwargs:
            * w       : wrapping factor (default: 2*pi)

        Returns:
            * Whrapped phase array, with the same size as phase
        '''
        w/=2
        wrapped_phi = (phase + w) % (2 * w) - w
        return(wrapped_phi)

    def downloadTecRGP(self, date, replace=False):
        '''
        Download the IONEX files from rgpdata.ign.fr, corresponding to one date (for previous and next round hours).
        Requires mean_time_utc variable.
            -> run after getAcquisitionTime

        Args:
            * date    : date to consider (YYYMMDD)

        Kwargs:
            * replace : re-download the .ION file if existing (default, False)

        Returns:
            * None
        '''
        # Create dict to store local paths to ionest files, if not existing alread
        if not hasattr(self, 'rgp_files'):
            self.rgp_files = {}

        # Local directory to store files
        self.local_rgp_dir = os.path.join(self.savedir, 'iono_rgp')
        if not os.path.isdir(self.local_rgp_dir):
            os.makedirs(self.local_rgp_dir)

        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        day_of_year = datetime.datetime(year, month, day).timetuple().tm_yday

        model_time1 = int(np.floor(self.mean_time_utc))
        model_time2 = int(np.ceil(self.mean_time_utc))
        time_session1 = string.ascii_uppercase[model_time1]
        time_session2 = string.ascii_uppercase[model_time2]

        rgp_dir1 = f'/pub/products/ionosphere/hourly_1/{year}/{day_of_year:>03}/'

        if time_session1 == 'X':
            if date[4:] == '1231':
                sys.exit('/!\ Acquisition around 23h on 31st of December => download next year not implemented!')
            rgp_dir1 = f'/pub/products/ionosphere/hourly_1/{year}/{day_of_year:>03}/'
            rgp_dir2 = f'/pub/products/ionosphere/hourly_1/{year}/{day_of_year+1:>03}/'
            rgp_file1 = f'HCAL_{date[2:4]}{day_of_year:>03}{time_session1}.ION'
            rgp_file2 = f'HCAL_{date[2:4]}{day_of_year+1:>03}A.ION'

        else:
            rgp_dir1 = f'/pub/products/ionosphere/hourly_1/{year}/{day_of_year:>03}/'
            rgp_dir2 = rgp_dir1
            rgp_file1 = f'HCAL_{date[2:4]}{day_of_year:>03}{time_session1}.ION'
            rgp_file2 = f'HCAL_{date[2:4]}{day_of_year:>03}{time_session2}.ION'

        self.rgp_files[date] = []

        # DOWNLOAD previous hour
        full_file = os.path.join(rgp_dir1, rgp_file1)
        local_file = os.path.join(self.local_rgp_dir, rgp_file1)

        if os.path.isfile(local_file) and not replace:
            if os.stat(local_file).st_size>0:
                self.rgp_files[date].append(local_file)
        else:
            # Open FTP connexion
            ftp = ftplib.FTP("rgpdata.ign.fr")
            try:
                ftp.login()
            except:
                print('FTP server rgpdata.ign.fr not available')
            # Download file
            try:
                ftp.retrbinary("RETR " + full_file, open(local_file, 'wb').write)
                self.rgp_files[date].append(local_file)
            except:
                #os.remove(local_file)
                print(f'{full_file} not found on server')

        # DOWNLOAD next hour
        full_file = os.path.join(rgp_dir2, rgp_file2)
        local_file = os.path.join(self.local_rgp_dir, rgp_file2)

        if os.path.isfile(local_file) and not replace:
            if os.stat(local_file).st_size>0:
                self.rgp_files[date].append(local_file)
        else:
            # Open FTP connexion
            if not ftp:
                ftp = ftplib.FTP("rgpdata.ign.fr")
                try:
                    ftp.login()
                except:
                    print('FTP server rgpdata.ign.fr not available')
            # Download file
            try:
                ftp.retrbinary("RETR " + full_file, open(local_file, 'wb').write)
                self.rgp_files[date].append(local_file)
            except:
                #os.remove(local_file)
                print(f'{full_file} not found on server')
        
        return
    
    def downloadTecIGS(self, date, dir=None, replace=False):
        '''
        Download the CDF files from cdaweb.gsfc.nasa.gov corresponding to one date (all IGS TEC computations for one day contained in one file)
        Requires max_time_utc variable.
            -> run after getAcquisitionTime

        Args:
            * date    : date to consider (YYYMMDD)

        Kwargs:
            * dir     : relative path to store iono files in a 'iono_igs' folder (default, self.savedir)
            * replace : re-download the .cdf file if existing (default, False)

        Returns:
            * None
        '''
        # Create dict to store local paths to ionest files, if not existing alread
        if not hasattr(self, 'igs_files'):
            self.igs_files = {}

        # Local directory to store files
        self.local_igs_dir = os.path.join(self.savedir, 'iono_igs')
        if not os.path.isdir(self.local_igs_dir):
            os.makedirs(self.local_igs_dir)

        year = int(date[:4])
        self.igs_time_before = int(np.floor(self.mean_time_utc/2))
        self.igs_time_after = int(np.ceil(self.mean_time_utc/2))

        # DOWNLOAD actual day
        igs_file=f'gps_tec2hr_igs_{date}_v01.cdf'
        local_file = os.path.join(self.local_igs_dir, igs_file)

        self.igs_files[date] = []

        if os.path.isfile(local_file) and not replace:
            self.igs_files[date].append(local_file)
        else:
            try:
                url = f'https://cdaweb.gsfc.nasa.gov/pub/data/gps/tec2hr_igs/{year}/{igs_file}'
                urllib.request.urlretrieve(url, filename=local_file)
                self.igs_files[date].append(local_file)
            except:
                os.remove(local_file)
                print(f'{igs_file} not found on server')

        if self.igs_time_after == 12:
            if date[4:] == '1231':
                sys.exit('/!\ Acquisition around 23h on 31st of December => download next year not implemented!')
            # DOWNLOAD next day
            next_date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:8])) + datetime.timedelta(1)
            next_date = next_date.strftime("%Y%m%d")
            igs_file=f'gps_tec2hr_igs_{next_date}_v01.cdf'
            local_file = os.path.join(self.local_igs_dir, igs_file)

            if os.path.isfile(local_file) and not replace:
                self.igs_files[date].append(local_file)
            else:
                try:
                    url = f'https://cdaweb.gsfc.nasa.gov/pub/data/gps/tec2hr_igs/{year}/{igs_file}'
                    urllib.request.urlretrieve(url, filename=local_file)
                    self.igs_files[date].append(local_file)
                except:
                    os.remove(local_file)
                    print(f'{igs_file} not found on server')

        return

    def ion2TecRGP(self, ionfile, skip_res):
        '''
        Compute a TEC map from a IONEX file download from RGP.
        Requires the mean_time_utc, lon_iono lat_iono variables.
            -> run after getAcquisitionTime, getLatLon, ground2IPP, and downloadTecRGP

        Args:
            * ionfile    : name of the IONEX file (e.g., HCAL_YYDDD.ION)
            * skip_res   : decimation factor in range and azimuth

        Returns:
            * TEC array in radar geometry, decimated by skip_res**2, in TECU
        '''

        # Parse ION file
        meta = np.loadtxt(ionfile, skiprows=4, max_rows=10, delimiter=':', usecols=-1, dtype='str')

        t0          = float(meta[0].split()[-1])
        lat0        = float(meta[1])
        lon0        = float(meta[2])
        height      = float(meta[3])
        mmax        = int(meta[4])+1
        nmax        = int(meta[5])+1
        norm_lat    = float(meta[7])
        norm_t      = float(meta[8])
        norm_tec    = float(meta[9].replace('D', 'e'))

        coef = np.loadtxt(ionfile, skiprows=18, max_rows=nmax*mmax, usecols=2).reshape(nmax, mmax)
        #rms = np.loadtxt(rgp_file, skiprows=18, max_rows=nmax*mmax, usecols=3).reshape(nmax, mmax)

        # Convert longitude to s
        s_arr = self.mean_time_utc + self.lon_iono[::skip_res,::skip_res]*24/360 - 12
        s0 = t0 + lon0*24/360 - 12 

        s_dif = s_arr-s0

        s_dif[s_dif < -12] += 24
        s_dif[s_dif >= 12] -= 24

        E = np.zeros(np.shape(self.lat_iono[::skip_res,::skip_res]))

        # Taylor development
        for m in range(mmax):
            for n in range(nmax):
                E += norm_tec * coef[n,m] * ((self.lat_iono[::skip_res,::skip_res]-lat0)/norm_lat)**n * ((s_dif)/norm_t)**m

        # Convert to TECU
        return E * 1e-16
        
    def computeTecRGP(self, date, skip_res=100, plot=False, saveplot=True):
        '''
        Compute the TEC map in radar geometry for one date of the TS, 
        as a weighted average of the predictions of the previous and next hour RGP models.
        Requires the rgp_files, local_ion_dir, mean_time_utc, local_rgp_dir, lon_iono lat_iono variables.
            -> run after getAcquisitionTime, getLatLon, downloadTecIGS and ground2IPP

        Args:
            * date    : date to consider (YYYMMDD)

        Kwargs:
            * skip_res   : decimation factor in range and azimuth (default, 50)
            * plot       : plot the TEC map
            * saveplot   : save the plot of the TEC

        Returns:
            * TEC array in radar geometry, decimated by skip_res**2, in TECU
        '''

        # Check what is available
        
        if not hasattr(self, 'rgp_files'):
            sys.exit("You need to download the TEC data first")

        elif date not in self.rgp_files.keys():
            sys.exit("TEC data has not been downloaded for "+date)

        elif len(self.rgp_files[date]) == 0:
            print(f"No TEC data found for {date}: TEC map not computed")

        elif len(self.rgp_files[date]) == 1: # Compute from just one model

            E = self.ion2TecRGP(self.rgp_files[date][0], skip_res)

            # PLOT
            if plot or saveplot:
                plt.figure()
                plt.imshow(E)
                plt.colorbar(label='E (TECU)', shrink=0.5)
                plt.title(f'{date} - {round(self.mean_time_utc, 1)}h')
                if plot:
                    plt.show()
                if saveplot:
                    plt.savefig(os.path.join(self.local_rgp_dir, 'TEC_'+date+'.jpg'))
                plt.close()

        elif len(self.rgp_files[date]) == 2: # Compute from the weighted average of the previous/next hours

            E1 = self.ion2TecRGP(self.rgp_files[date][0], skip_res)
            E2 = self.ion2TecRGP(self.rgp_files[date][1], skip_res)

            coef1 = np.ceil(self.mean_time_utc) - self.mean_time_utc
            coef2 = self.mean_time_utc - np.floor(self.mean_time_utc)

            E = E1*coef1 + E2*coef2

            # PLOT
            if plot or saveplot:
                fig, axs = plt.subplots(1,3, sharey=True)

                vmin = np.nanmin(np.minimum(E1, E2))
                vmax = np.nanmax(np.maximum(E1, E2))

                axs[0].imshow(E1, vmin=vmin, vmax=vmax)
                axs[1].imshow(E, vmin=vmin, vmax=vmax)
                pcm = axs[2].imshow(E2, vmin=vmin, vmax=vmax)

                plt.colorbar(pcm, ax=axs, label='E (TECU)', shrink=0.5)

                axs[0].set_title(f'{np.floor(self.mean_time_utc)}h model')
                axs[1].set_title(f'Weighted average')
                axs[2].set_title(f'{np.ceil(self.mean_time_utc)}h model')

                fig.suptitle(f'{date} - {round(self.mean_time_utc, 1)}h')

                if plot:
                    plt.show()
                if saveplot:
                    plt.savefig(os.path.join(self.local_rgp_dir, 'TEC_'+date+'.jpg'))
                plt.close()

        return E
    
    def computeTecIGS(self, date, model='IGS', skip_res=100, plot=False, saveplot=True):
        '''
        Compute the TEC map in radar geometry for one date of the TS, 
        using temporal interpolation of IGS models for previous and next steps (2h sampling).
        Requires the igs_files, local_ion_dir, mean_time_utc, local_igs_dir, lon_iono_before, lon_iono_after and lat_iono variables.
            -> run after getAcquisitionTime, getLatLon, downloadTecIGS and ground2IPP

        Args:
            * date    : date to consider (YYYMMDD)

        Kwargs:
            * model      : model to used: IGS, JPL, CODE, ESA, UPC (default, IGS)
            * skip_res   : decimation factor in range and azimuth (default, 50)
            * plot       : plot the TEC map
            * saveplot   : save the plot of the TEC

        Returns:
            * TEC array in radar geometry, decimated by skip_res**2, in TECU
        '''

        # Check what is available
        
        if not hasattr(self, 'igs_files'):
            sys.exit("You need to download the TEC data first")

        elif date not in self.igs_files.keys():
            sys.exit("TEC data has not been downloaded for "+date)

        elif len(self.igs_files[date]) == 0:
            print(f"No TEC data found for {date}: TEC map not computed")

        else:
            # Read file
            cdf = pycdf.CDF(self.igs_files[date][0])
            lat_grid = np.array(cdf['lat'])
            lon_grid = np.array(cdf['lon'])
            TEC_before = cdf['tec'+model[:3]][self.igs_time_before,:,:]   # lat[71] x lon[73]

            if len(self.igs_files[date]) == 1: # Normal case: both time steps are in the same daily file
                TEC_after  = cdf['tec'+model[:3]][self.igs_time_after,:,:]

            elif self.igs_time_after == 12 and len(self.igs_files[date]) == 2: # If next time step is in another file (next day)
                cdf = pycdf.CDF(self.igs_files[date][1])
                TEC_after = cdf['tec'+model[:3]][0,:,:]
            
            else:
                sys.exit('Problem with IGS files or times...')

            # Interpolate the TEC map at lat-lon of the radar image projected on ionosphere layer
            lon_arr, lat_arr = np.meshgrid(lon_grid, lat_grid)
            TEC_interp_before = griddata((lat_arr.ravel(), lon_arr.ravel()), TEC_before.ravel(), (self.lat_iono[::skip_res,::skip_res], self.lon_iono_before[::skip_res,::skip_res]), method='cubic')
            TEC_interp_after = griddata((lat_arr.ravel(), lon_arr.ravel()), TEC_after.ravel(), (self.lat_iono[::skip_res,::skip_res], self.lon_iono_after[::skip_res,::skip_res]), method='cubic')

            # Interpolate in time
            before_coef = (self.igs_time_after*2-self.mean_time_utc)/(self.igs_time_after*2-self.igs_time_before*2)
            after_coef = (self.mean_time_utc-self.igs_time_before*2)/(self.igs_time_after*2-self.igs_time_before*2)
            E = before_coef * TEC_interp_before + after_coef * TEC_interp_after

            # PLOT
            if plot or saveplot:
                fig, axs = plt.subplots(1,3, sharey=True)

                vmin = np.nanmin(np.minimum(TEC_interp_before, TEC_interp_after))
                vmax = np.nanmax(np.maximum(TEC_interp_before, TEC_interp_after))

                axs[0].imshow(TEC_interp_before, vmin=vmin, vmax=vmax)
                axs[1].imshow(E, vmin=vmin, vmax=vmax)
                pcm = axs[2].imshow(TEC_interp_after, vmin=vmin, vmax=vmax)

                plt.colorbar(pcm, ax=axs, label='E (TECU)', shrink=0.5)

                axs[0].set_title(f'{self.igs_time_before*2}h model')
                axs[1].set_title(f'Weighted average')
                axs[2].set_title(f'{self.igs_time_after*2}h model')

                fig.suptitle(f'{date} - {round(self.mean_time_utc, 1)}h')

                if plot:
                    plt.show()
                if saveplot:
                    plt.savefig(os.path.join(self.local_igs_dir, 'TEC_'+date+'.jpg'))
                plt.close()

        return E

    def ground2IPP(self, lon=None, lat=None, H=400, pad=200, time_shifts=None, plot=True, saveplot=False):
        '''
        Convert lat-lon on the ground to lat-lon at the ionospheric piercing point (IPP).
            -> run after getIncidence and getLatLon

        Kwargs:
            * lon         : longitude value or array (default, self.lon_radar)
            * lat         : latitude value or array (default, self.lat_radar)
            * H           : height of the IPP (default, 400 km)
            * pad         : padding in the array to select corner value (helps to avoid the NaNs)
            * time_shifts : shift the longitude to take into account Earth's rotation with respect to the model time
                                -> should be a list : [t_acquisition-t_model_before, t_acquisition-t_model_after] in decimal hours
                                -> if set to 'igs', computes directly the shifts from the acquisition and IGS model times (run downloadTecIGS before)
            * plot        : plot the ground and IPP coverage
            * saveplot    : save this plot

        Returns:
            * None
        '''     

        earth_radius = 6371     # km
        omega = 360/24          # °/h

        # Approximate degree length between ground and ionosphere
        degree_length = (earth_radius+H/2)*2*np.pi/360

        if lon is None:
            lon = self.lon_radar
        if lat is None:
            lat = self.lat_radar

        # Without time_shift
        self.lat_iono = lat + H/degree_length * np.tan(np.radians(self.incidence)) * np.sin(np.radians(self.look_angle-90))
        self.lon_iono = lon - H/degree_length * np.tan(np.radians(self.incidence)) * np.cos(np.radians(self.look_angle-90))

        corner_lats = np.array((lat[pad][pad], lat[pad][-pad], lat[-pad][-pad], lat[-pad][pad], lat[pad][pad]))
        corner_lons = np.array((lon[pad][pad], lon[pad][-pad], lon[-pad][-pad], lon[-pad][pad], lon[pad][pad]))
        corner_lons_iono  = np.array((self.lon_iono[pad][pad], self.lon_iono[pad][-pad], self.lon_iono[-pad][-pad], self.lon_iono[-pad][pad], self.lon_iono[pad][pad]))
        corner_lats_iono  = np.array((self.lat_iono[pad][pad], self.lat_iono[pad][-pad], self.lat_iono[-pad][-pad], self.lat_iono[-pad][pad], self.lat_iono[pad][pad]))

        # With earth's rotation
        if time_shifts is not None:

            if time_shifts == 'igs':
                time_shifts = [self.mean_time_utc-self.igs_time_before*2, self.mean_time_utc-self.igs_time_after*2]

            self.lon_iono_before = self.lon_iono + omega * time_shifts[0]
            self.lon_iono_after  = self.lon_iono + omega * time_shifts[1]

            corner_lons_iono_before = np.array((self.lon_iono_before[pad][pad], self.lon_iono_before[pad][-pad], self.lon_iono_before[-pad][-pad], self.lon_iono_before[-pad][pad], self.lon_iono_before[pad][pad]))
            corner_lons_iono_after  = np.array((self.lon_iono_after[pad][pad],  self.lon_iono_after[pad][-pad],  self.lon_iono_after[-pad][-pad],  self.lon_iono_after[-pad][pad],  self.lon_iono_after[pad][pad] ))

        if plot or saveplot:
            plt.figure()
            extent = 20 # °
            ax = plt.axes(projection=cartopy.crs.PlateCarree())
            ax.coastlines(linewidth=0.5)

            gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True,
                            linewidth=0.5, linestyle=':', color="grey", alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            ax.add_feature(cartopy.feature.LAND, color='lightgrey')
            ax.add_feature(cartopy.feature.BORDERS, linestyle='--', linewidth=0.5)

            ax.plot(corner_lons, corner_lats, color='r', ls='-', lw=2, label='Ground coverage')
            ax.plot(corner_lons_iono, corner_lats_iono, color='r', ls='--', lw=1, label=f'Ionosphere intersect (H = {H} km)')

            if time_shifts is not None:
                extent = 30 # °
                ax.plot(corner_lons_iono_before, corner_lats_iono, color='g', ls='--', lw=1, label=f'   with time_shift = {time_shifts[0]:.2f}h (before)')
                ax.plot(corner_lons_iono_after, corner_lats_iono, color='b', ls='--', lw=1, label=f'   with time_shift = {time_shifts[1]:.2f}h (after)')

            ax.set_extent([np.nanmean(self.lon_iono)-extent, np.nanmean(self.lon_iono)+extent, np.nanmean(self.lat_iono)-extent/2, np.nanmean(self.lat_iono)+extent/2])
            ax.legend(fancybox=0, edgecolor='w')
            ax.set_title(f'{self.name}', weight='bold')

            if plot:
                plt.show()
            if saveplot:
                if time_shifts is not None:
                    outfile = f'IPP_projection_{self.name}_earth_rotation.jpg'
                else:
                    outfile = f'IPP_projection_{self.name}.jpg'
                plt.savefig(os.path.join(self.savedir, outfile))

        return

    def Tec2Ips(self, E, wavelength=0.0556, top_iono=0.9, plot=False):
        '''
        Compute the Ionospheric Phase Screen from a TEC map.

        Args:
            * E         : TEC array (in TECU)

        Kwargs:
            * wavelength : radar wavelength (default, Sentinel-1)
            * top_iono   : factor to compensate for the top-side part of the ionosphere
            * plot       : plot the TEC map

        Returns:
            * Phase array of the size of E, contining the ionospheric phase
        '''

        K = 40.28                 # Appelton–Hartree equation constant
        c = 299792458             # Speed of light
        f = c/wavelength

        skip_res = round(self.incidence.shape[0]/E.shape[0]+1)
        ips = - top_iono * 4 * np.pi * K / (c*f) * E * 10**(16) * 1/np.cos(np.radians(self.incidence[::skip_res,::skip_res]))

        if plot:
            plt.figure()
            plt.imshow(self.wrap(ips), cmap='jet', vmin=-np.pi, vmax=np.pi)
            plt.colorbar(label='IPS (rad)')

        return ips

    def fitPhaseRamp(self, phase, skip_res=1, plot=False):
        '''
        Fit a ramp in a (unwrapped) phase array

        Args:
            * phase      : phase array

        Kwargs:
            * skip_res   : decimation factor of the phase to convert the coefficients in full resolution
            * plot       : plot the phase, ramp, and detrended phase

        Returns:
            * azimuth ramps coefficient, range ramp coefficient, sigma of the detrended phase
        '''

        X, Y = np.meshgrid(np.arange(phase.shape[1]), np.arange(phase.shape[0]))
        XX = X[~np.isnan(phase)].flatten()
        YY = Y[~np.isnan(phase)].flatten()
        data = phase[~np.isnan(phase)].flatten()
        A = np.c_[XX, YY, np.ones(data.shape[0])]
        C,_,_,_ = lstsq(A, data)    # coefficients
            
        # evaluate it on grid
        ramp = C[0]*X + C[1]*Y + C[2]
        phase_detrend = phase - ramp
        sigma = np.nanstd(phase_detrend)

        if plot:
            fig, axs = plt.subplots(1, 3, sharey=True)
            axs[0].imshow(self.wrap(phase), cmap='jet', vmin=-np.pi, vmax=np.pi)
            axs[1].imshow(self.wrap(ramp), cmap='jet', vmin=-np.pi, vmax=np.pi)
            pcm = axs[2].imshow(self.wrap(phase_detrend), cmap='jet', vmin=-np.pi, vmax=np.pi)
            plt.colorbar(pcm, ax=axs, shrink=0.5, label='Phase (rad)')
            axs[0].set_title('Phase')
            axs[1].set_title('Ramp')
            axs[2].set_title('Detrended phase')
            fig.supxlabel('Range')
            fig.supylabel('Azimuth')

        return C[1]/skip_res, C[0]/skip_res, sigma

###############################################

    def computeTecRampsRGP(self, skip_res=100):
        '''
        Compute ramps due to TEC from the RGP IONEX model (centered on France)

        Kwargs:
            * skip_res   : decimation factor in range and azimuth (default, 100)

        Returns:
            * None
        '''

        self.getAcquisitionTime()
        self.getLatLon()
        self.getIncidence()

        if self.verbose:
            print("\n---------------------------------")
            print(f"    Computing ionospheric ramps with RGP TEC model")
            
        if self.verbose:
            print(f"\n        Fetch RGP TEC data for all dates...")

        num_prod = []
        for date in tqdm(self.dates):
            
            self.downloadTecRGP(date)
            num_prod.append(len(self.rgp_files[date]))

        num_prod = np.array(num_prod)
        
        if np.count_nonzero(num_prod) < self.Ndates:
            print(f'            -> {self.Ndates-np.count_nonzero(num_prod)} out of {self.Ndates} have no TEC data on the RGP repository')

        if np.count_nonzero(num_prod==1) > 0:
            print(f'            -> {np.count_nonzero(num_prod==1)} out of {self.Ndates} have data for only one time step on the RGP repository')
    
        if np.count_nonzero(num_prod) == self.Ndates:
            print(f'            -> All dates have data for both times steps on the RGP repository')

        if self.verbose:
            print(f"\n        Compute IPP coordinates...")

        self.ground2IPP(plot=False, saveplot=True)

        if self.verbose:
            print(f"\n        Compute TEC maps and fit ramps for all dates...")

        self.ramp_az_rgp = np.zeros(self.Ndates)
        self.ramp_ra_rgp = np.zeros(self.Ndates)
        self.ramp_sig_rgp = np.zeros(self.Ndates)

        for i in tqdm(range(self.Ndates)):
            date = self.dates[i]

            if num_prod[i] > 0:

                # Get the TEC 
                tec = self.computeTecRGP(date, skip_res=skip_res, plot=False, saveplot=False)

                # Then convert to phase
                ips = self.Tec2Ips(tec)

                # Then fit ramps
                self.ramp_az_rgp[i], self.ramp_ra_rgp[i], self.ramp_sig_rgp[i] = self.fitPhaseRamp(ips, skip_res=skip_res)

        # Put zeros to nan
        self.ramp_az_rgp[self.ramp_az_rgp == 0.] = np.nan
        self.ramp_ra_rgp[self.ramp_ra_rgp == 0.] = np.nan
        self.ramp_sig_rgp[self.ramp_sig_rgp == 0.] = np.nan
        
        # Reference with first date
        if self.ramp_az_rgp[0] != np.nan:
            self.ramp_az_rgp -= self.ramp_az_rgp[0]
            self.ramp_ra_rgp -= self.ramp_ra_rgp[0]
        else:
            sys.exit('Ramp of first date is NaN, reference in another way...')

        # Save AZIMUTH ramps
        header = f'date_decyr az_ramp / number_of_products date'
        file = os.path.join(self.local_rgp_dir, f'list_ramp_az_RGP.txt')
        arr = np.c_[ self.dates_decyr, self.ramp_az_rgp, np.zeros(self.Ndates), num_prod, self.dates ].astype(float)
        np.savetxt(file, arr, fmt='%.6f % .7e % .7e %  2i % 6d', header=header)

        # Save RANGE ramps
        header = f'date_decyr ra_ramp / number_of_products date'
        file = os.path.join(self.local_rgp_dir, f'list_ramp_ra_RGP.txt')
        arr = np.c_[ self.dates_decyr, self.ramp_ra_rgp, np.zeros(self.Ndates), num_prod, self.dates ].astype(float)
        np.savetxt(file, arr, fmt='%.6f % .7e % .7e %  2i % 6d', header=header)

        # Save SIGMA
        header = f'date_decyr sigma date'
        file = os.path.join(self.local_rgp_dir, f'list_ramp_sigma_RGP.txt')
        arr = np.c_[ self.dates_decyr, self.ramp_sig_rgp, self.dates ].astype(float)
        np.savetxt(file, arr, fmt='%.6f % .7e % 6d')

        return
    
    def computeTecRampsIGS(self, skip_res=100, model='IGS'):
        '''
        Compute ramps due to TEC from different GIMs compiled by IGS.

        Kwargs:
            * model      : model to used: IGS, JPL, CODE, ESA, UPC (default, IGS)
            * skip_res   : decimation factor in range and azimuth (default, 100)

        Returns:
            * None
        '''

        self.getAcquisitionTime()
        self.getLatLon()
        self.getIncidence()

        if self.verbose:
            print("\n---------------------------------")
            print(f"    Computing ionospheric ramps with IGS TEC model")
            
        if self.verbose:
            print(f"\n        Fetch IGS TEC data for all dates...")

        num_prod = []
        for date in tqdm(self.dates):
            
            self.downloadTecIGS(date)
            num_prod.append(len(self.igs_files[date]))

        num_prod = np.array(num_prod)
        
        if np.count_nonzero(num_prod) < self.Ndates:
            print(f'            -> {self.Ndates-np.count_nonzero(num_prod)} out of {self.Ndates} have no TEC data on the IGS repository')
        else:
            print(f'            -> All dates have TEC data on the IGS repository')

        if self.verbose:
            print(f"\n        Compute IPP coordinates taking in to account the Earth's rotation...")

        self.ground2IPP(time_shifts='igs', plot=False, saveplot=True)

        if self.verbose:
            print(f"\n        Compute TEC maps and fit ramps for all dates...")

        if not hasattr(self, 'ramp_az_igs'):
            self.ramp_az_igs = {}
        if not hasattr(self, 'ramp_ra_igs'):
            self.ramp_ra_igs = {}
        if not hasattr(self, 'ramp_sig_igs'):
            self.ramp_sig_igs = {}
        
        self.ramp_az_igs[model] = np.zeros(self.Ndates)
        self.ramp_ra_igs[model] = np.zeros(self.Ndates)
        self.ramp_sig_igs[model] = np.zeros(self.Ndates)

        for i in tqdm(range(self.Ndates)):
            date = self.dates[i]

            if num_prod[i] > 0:

                # Get the TEC 
                tec = self.computeTecIGS(date, model=model, skip_res=skip_res, plot=False, saveplot=False)

                # Then convert to phase
                ips = self.Tec2Ips(tec)

                # Then fit ramps
                self.ramp_az_igs[model][i], self.ramp_ra_igs[model][i], self.ramp_sig_igs[model][i] = self.fitPhaseRamp(ips, skip_res=skip_res)

        # Put zeros to nan
        self.ramp_az_igs[model][self.ramp_az_igs == 0.] = np.nan
        self.ramp_ra_igs[model][self.ramp_ra_igs == 0.] = np.nan
        self.ramp_sig_igs[model][self.ramp_sig_igs == 0.] = np.nan
        
        # Reference with first date
        if self.ramp_az_igs[model][0] != np.nan:
            self.ramp_az_igs[model] -= self.ramp_az_igs[model][0]
            self.ramp_ra_igs[model] -= self.ramp_ra_igs[model][0]
        else:
            sys.exit('Ramp of first date is NaN, reference in another way...')

        # Save AZIMUTH ramps
        header = f'date_decyr az_ramp / number_of_products date'
        file = os.path.join(self.local_igs_dir, f'list_ramp_az_IGS_{model}.txt')
        arr = np.c_[ self.dates_decyr, self.ramp_az_igs[model], np.zeros(self.Ndates), num_prod, self.dates ].astype(float)
        np.savetxt(file, arr, fmt='%.6f % .7e % .7e %  2i % 6d', header=header)

        # Save RANGE ramps
        header = f'date_decyr ra_ramp / number_of_products date'
        file = os.path.join(self.local_igs_dir, f'list_ramp_ra_IGS_{model}.txt')
        arr = np.c_[ self.dates_decyr, self.ramp_ra_igs[model], np.zeros(self.Ndates), num_prod, self.dates ].astype(float)
        np.savetxt(file, arr, fmt='%.6f % .7e % .7e %  2i % 6d', header=header)

        # Save SIGMA
        header = f'date_decyr sigma date'
        file = os.path.join(self.local_igs_dir, f'list_ramp_sigma_IGS_{model}.txt')
        arr = np.c_[ self.dates_decyr, self.ramp_sig_igs[model], self.dates ].astype(float)
        np.savetxt(file, arr, fmt='%.6f % .7e % 6d')

        return
