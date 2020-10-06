#!/usr/bin/env python
# coding: utf-8

# In[2]:


from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Column, Table, vstack
import tarfile
import os
import glob
from pathlib import Path
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline


# In[33]:


#extract all files in 2016 tar into jupyter-- success
#spec=tarfile.open('hi-manga_gbt_spectra_2016.tar')
#spec.extractall(path='/Users/Frank/KNAC_Internship/2016-spectra')

#extract all files in 2017-2019 tar into jupyter-- success
#spec=tarfile.open('hi-manga_gbt_spectra_2017_2019.tar')
#spec.extractall(path='/Users/Frank/KNAC_Internship/2017-2019-spectra')

#extract all files in alfalfa tar into jupyter
#spec=tarfile.open('mangaHI_alfalfa_spec.tar')
#spec.extractall(path='/Users/Frank/KNAC_Internship/ALFALFA-spectra')


# In[16]:


#test single alfalfa spectra file cause they looking weird af
hdul = fits.open('mangaHI-8338-3701-Copy1.fits')
hdul.info()

hdu = hdul[1]
hdr0 = hdul[0].header 
hdr = hdul[1].header
data = hdu.data
print(hdr)
hdul.close()


# In[17]:


#test single spectra plot
t=Table(data)
x=t['VHI']-hdr['OBJ_VEL']
y=t['FHI']

plt.figure(figsize=(5,5))
plt.plot(x, y, color='red')
plt.xlabel("Velocity [km/s]")
plt.ylabel("Flux [Jy km/s]")
plt.title('Original Spectra')
#plt.savefig('Test Plot.png')


# In[7]:


#iterate over file in mac
#directory= "/Users/Frank/KNAC_Internship/fits"
#path, dirs, files = next(os.walk("/Users/Frank/KNAC_Internship/fits"))
#file_count = len(files)
#print(file_count)

#for filename in os.listdir(directory):
    #if filename.endswith('.fits'):
        #print(os.path.join(filename))
   # if filename.endswith('.fits'):
#filelist=os.path.join(filename)
#print(filelist)


# In[47]:


#pin down jupyter location of 2016 spectra for loop 
fil_location = os.path.join('/Users/Frank/KNAC_Internship/all_spectra/2016-spectra', '*.fits')

filnames = glob.glob(fil_location)
print(len(filnames))


# In[45]:


#pin down jupyter location of 2017-2019 spectra for loop 
fill_location = os.path.join('/Users/Frank/KNAC_Internship/all_spectra/2017-2019-spectra', '*.fits')

fillnames = glob.glob(fill_location)
print(len(fillnames))


# In[46]:


#pin down jupyter location of alfalfa spectra for loop 
filll_location = os.path.join('/Users/Frank/KNAC_Internship/all_spectra/ALFALFA-spectra', '*.fits')

filllnames = glob.glob(filll_location)
print(len(filllnames))

curious_loc = os.path.join('/Users/Frank/KNAC_Internship/all_spectra/ALFALFA-spectra', 'mangaHI-AGC*')
curious_names = glob.glob(curious_loc)
print(len(curious_names), 'of ALFALFA files use AGC naming convention.')


# In[48]:


feel_location = os.path.join('/Users/Frank/KNAC_Internship/all_spectra', '**/*.fits')
feel_names = glob.glob(feel_location)
print(len(feel_names))
print('Only 4592 of these spectra files use SDSS plate-ifu naming convention.')


# In[4]:


file2='all-finall.fits'

hdul8 = fits.open(file2)
hdul8.info()

hdu8 = hdul8[1]
hdr08 = hdul8[0].header 
hdr8 = hdul8[1].header
alldata = hdu8.data


# In[56]:


def title(y): #this isolates the plateifu in the spectra file name
    title0 = (os.path.splitext(y)[0])
    base=os.path.basename(y)
    title1 = os.path.splitext(base)[0]
    titlep = title1.replace('mangaHI-', '') #titlep is plateifu
    return titlep

#want to go through all spectra files, find the ones that exist in the samples (alldata), and move those to a new folder (samples_spectra)
for f in feel_names:
    name = title(f)
    #sel=np.where(alldata['plateifu_1']==name)
    #if np.sum(alldata['plateifu_1']==name) > 0:
    if name in alldata['plateifu_1']:
    #if name in alldata['plateifu_1']: #"in" statement checks if item is in another list
        #dst = '/Users/Frank/KNAC_Internship/samples_spectra'
        #just running linux copy file command
        #os.system('cp '+f+' '+dst)
        #copyfile(f, dst)
        #print('This', name)


# In[51]:


drip_location = os.path.join('/Users/Frank/KNAC_Internship/samples_spectra', '*.fits')
drip_names = glob.glob(drip_location)
print(len(drip_names))


# In[66]:


check = []
for r in drip_names:
    namer = title(r)
    check.append(namer)
#print(check)
for t in alldata['plateifu_1']:
    if t in check:
        pass
    else:
        print(t)
print('The first eight are part of the control sample, and the last is a red geyser.')


# In[1]:


# MOST SUCCESSFUL loop to pull galaxy velocity out of header and isolate velocity and flux columns
def read_spec(filnam): 
    sp = fits.open(filnam)
    s = sp[1]
    datat = s.data
    header = sp[1].header #; can use to pull out header data later if I want
    
    galv = header['OBJ_VEL']
    
    flux = datat['FHI']
    
    vel = datat['VHI']
    
    corrected_vel = vel-galv
    
    return vel, flux, galv, corrected_vel
    
    sp.close()

#for f in feel_names: #goes through and pulls out VHI and FHI and galaxy velocity
    #vel, flux, galv, corrected_vel = read_spec(f)
    #corrected_vel = vel-galv
    #print(flux)


# In[102]:


#second test of single spectra plot, but with velocity corrected for galaxy velocity
x1=corrected_vel[0]
y1=flux[0]
plt.figure(figsize=(5,5))
plt.scatter(x1, y1, color='red', marker='.')
plt.xlabel("Velocity")
plt.ylabel("Flux")
#plt.savefig('Red_Geyser_Mass_plot.png')
plt.title("Test2")


# In[7]:


# call all plateifu of spectra
def read_name(toop): 
    sp1 = fits.open(toop)
    s1 = sp1[1]
    datat1 = s1.data
    header1 = sp1[1].header 
    
    mangaifu = header1['OBJECT']
    
    return mangaifu

#myfile = open('all_spec_id2.txt', 'w')
#yah = []
#for f in feel_names:
    mangaifu = read_name(f)
    #yah.append(mangaifu)
    #myfile.write("%s\n" % mangaifu)
    #print(mangaifu)
#myfile.close()
#print(len(yah))


# In[9]:


#drpall file with only selected spectra info
hdul7 = fits.open('all-spec-info-final.fits')
hdul7.info()
hdu7 = hdul7[1]
hdr7 = hdul7[0].header 
hdr7 = hdul7[1].header
data7 = hdu7.data
hdul7.close()


# In[10]:


#applying z < 0.05 restriction onto all of the spectra from the original three files
mask = data7['Z'] < 0.05
newsamples = data7[mask]
print(len(data7))
print('Only', len(newsamples), 'galaxies out of all three original files (not including AGC"s from ALFALFA survey) are z < 0.05.')


# In[11]:


#make table of necessary data for scaling constant

c = 3e5 #km/s
H = 70 #km/s/Mpc
tebl = Table()
tebl['MANGAID'] = data7['plate-ifu']
tebl['M*'] = data7['nsa_elpetro_mass']
tebl['Distance [Mpc]'] = (data7['nsa_zdist'])*((3e5)/(70))
p=(data7['nsa_zdist'])*((3e5)/(70))
tebl['Scale for Flux'] = (2.36e5*((p)**2))/(data7['nsa_elpetro_mass'])
#tebl.show_in_notebook()

np.set_printoptions(threshold=np.inf)
dat = np.array(tebl['Scale for Flux'])
print(len(dat))
print(dat)


# for t in dat:
#     for f in feel_names:
#         vel, flux, galv, corrected_vel = read_spec(f)
#     #print(flux)
#     #for t in dat:
#         for s in flux:
#             print(s)
#     break
# print(flux)

# #scaling flux
# for f in feel_names:
#     vel, flux, galv, corrected_vel = read_spec(f)
#     #print(flux)
# b = []
# for t in dat:
#     for s in flux:
#         scaled_flux = t * s
#         print(scaled_flux)
#         b.append(scaled_flux)
#         #print(b)
# 
# print(len(b)) 
# #print(b)

# In[28]:


#plotting all ORIGINAL data, without interpolation

for f in feel_names: #goes through and pulls out VHI and FHI and galaxy velocity
    vel, flux, galv, corrected_vel = read_spec(f)
    corrected_vel = vel-galv #corrects galaxy velocity for velocity of the galaxy itself
    x = corrected_vel
    y = flux #FIGURE OUT HOW TO GET SCALED FLUX IN HERE
    for i in range(len(x)):
        plt.figure()
        plt.plot(x[i],y[i])
        fname = os.path.basename(f)
        plt.title(fname)
        plt.xlabel('VHI [km/s]')
        plt.ylabel('FHI')
    # Figures not saved yet
        plt.show()
        plt.close()
    break


# In[65]:


#interpolating all data onto same coordinate system, takes ages

for f in feel_names: #goes through and pulls out VHI and FHI and galaxy velocity
    vel, flux, galv = read_spec(f)
    corrected_vel = vel-galv
    #print(corrected_vel, flux, galv)
    xe = corrected_vel
    ye = flux #FIGURE OUT HOW TO GET SCALED FLUX IN HERE
    print(xe)
    #interpolation_function = InterpolatedUnivariateSpline(xe,ye)
    #new_x = np.arange(-1000,1000,5)
    #new_y = interpolation_function(new_x)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(new_x,new_y, color = 'green')
    #ax1.plot(xe,ye, color = 'red')
    plt.xlim(-1000,1000)
    fname = os.path.basename(f)
    plt.title(fname)
    plt.xlabel('VHI [km/s]')
    plt.ylabel('FHI')
    # Figures not saved yet
    plt.show()
    plt.close()
    print(xe)
    break


# In[61]:


#interpolating all data onto same coordinate system but with different interpolate command
#ALSO OVERPLOTS ORIGINAL AND INTERPOLATED DATA

for f in feel_names: #goes through and pulls out VHI and FHI and galaxy velocity
    vel, flux, galv = read_spec(f)
    corrected_vel2 = vel-galv
    yeet=flux
    for p in corrected_vel2:
        for q in yeet:
            int_func = interpolate.interp1d(p,q,fill_value="extrapolate")
            new_x2 = np.arange(-1000,1000,5)
            new_y2 = int_func(new_x2)
            figg = plt.figure()
            ax1 = figg.add_subplot(111)
            ax1.plot(new_x2,new_y2, color = 'red')
            ax1.plot(p,q, color = 'blue')
            plt.xlim(-1000,1000)
            fname = os.path.basename(f)
            plt.title(fname)
            plt.xlabel('VHI [km/s]')
            plt.ylabel('FHI')
            plt.show()
            plt.close()
            
            #print(new_y2)
            #a, b = zip(*new_y2)
            #print(a, b)
    #break


# In[66]:


listy = []
for f in feel_names: #goes through and pulls out VHI and FHI and galaxy velocity
    vel, flux, galv = read_spec(f)
    corrected_vel2 = vel-galv
    #for p in corrected_vel2:
        #for q in flux:
            int_func = interpolate.interp1d(p,q,fill_value="extrapolate")
            new_x2 = np.arange(-1000,1000,5)
            new_y2 = int_func(new_x2)
            
            listy.append(new_y2)
            #print(listy)
            #col_totals_f = [ sum(x) for x in zip(listy) ]
            #print(col_totals_f)
            
#col_totals_f = [ sum(x) for x in zip(listy) ]
            #print(col_totals_f)

list_y = new_y2
#print(list_y)
print(np.shape(list_y))
col_totals_f = [ sum(x) for x in zip(list_y) ]
print(col_totals_f)


# In[66]:


for f in filnames: #goes through and pulls out VHI and FHI and galaxy velocity
    vel, flux, galv = read_spec(f)
    corrected_vel = vel-galv
    #print(corrected_vel, flux, galv)
    xe = corrected_vel
    ye = flux #FIGURE OUT HOW TO GET SCALED FLUX IN HERE
    interpolation_function = InterpolatedUnivariateSpline(xe,ye)
    new_x = np.arange(-1000,1000,5)
    new_y = interpolation_function(new_x)
    listicle = new_y.tolist()
    coltotalsf = [ sum(x) for x in zip(listicle) ]
    #print(len(coltotalsf))
    #break
    
coltotalsf = [ sum(x) for x in zip(new_y) ]
#print(coltotalsf)


# # BELOW ARE FAILED CELLS

# In[ ]:


# Dictionary to store NxN data matrices of spectra
new = {}

# Interate over all images ('j'), which contain the current object, indexed by 'i'
for i in range(0, len(files)):
   # for j in range(0, len(containingImages[containedObj[i]])):

        countImages += 1

        # Path to the current image: 'mnt/...'
        current_image_path = ImagePaths[int(containingImages[containedObj[i]][j])]

        # Open .fits images
        with fits.open(current_image_path, memmap=False) as hdul:
            # Collect image data
            image_data = fits.getdata(current_image_path)

            # Collect WCS data from the current .fits's header
            ImageWCS = wcs.WCS(hdul[1].header)

            # Cropping parameters:
            # 1. Sky-coordinates of the croppable object
            # 2. Size of the crop, already defined above
            Coordinates = coordinates.SkyCoord(finalObjects[i][1]*u.deg,finalObjects[i][2]*u.deg, frame='fk5')
            size = (cropSize*u.pixel, cropSize*u.pixel)

            try:
                # Cut out the image tile
                cutout = Cutout2D(image_data, position=Coordinates, size=size, wcs=ImageWCS, mode='strict')

                # Write the cutout to a new FITS file
                cutout_filename = "Cropped_Images_Sorted/Cropped_" + str(containedObj[i]) + current_image_path[-23:]

                # Sava data to dictionary
                CroppedObjects[cutout_filename] = cutout.data

                foundImages += 1

            except:
                pass

            else:
                del image_data
                continue

        # Memory maintainance                
        gc.collect()

        # Progress bar
        sys.stdout.write("\rProgress: [{0}{1}] {2:.3f}%\tElapsed: {3}\tRemaining: {4}  {5}".format(u'\u2588' * int(countImages/allCrops * progressbar_width),
                                                                                                   u'\u2591' * (progressbar_width - int(countImages/allCrops * progressbar_width)),
                                                                                                   countImages/allCrops * 100,
                                                                                                   datetime.now()-starttime,
                                                                                                   (datetime.now()-starttime)/countImages * (allCrops - countImages),
                                                                                                   foundImages))

        sys.stdout.flush()


# In[151]:


#see all files in jupyter fits folder
ls fits


# In[94]:


testfile=os.path.join('fits', 'mangaHI-8329-1902.fits')
outfil = open(testfile,"r")
data = outfil.readlines()


# In[81]:


for filename in os.listdir(directory):
    lines = open(filename).readlines()
    print(lines)


# In[66]:


hdulist2 = [fits.open('/Users/Frank/KNAC_Internship/fits' % i) for i in range(332)]


# In[57]:


destdir = Path('/Users/Frank/KNAC_Internship/fits')
files = [p for p in destdir.iterdir() if p.is_file()]
for p in files:
    with p.open() as f:
        t=Table(f)
        print(t)


# In[72]:


from astropy.io import fits
for filename in os.listdir(directory):
    if filename.endswith('.fits'):
        with open('filename', 'w') as f:
            data2 = 'some data to be written to the file'
            f.write(data2)
for filename in 'fits':
    hdul = fits.open(filename)
    for hdu in hdul:
        hdu_data = hdul.data
         # Do some stuff with the data
         # ...
         # Don't need the data anymore; delete all references to it
         # so that it can be garbage collected
        del hdu_data
        del hdu.data
hdul.close()


# In[51]:


for file in directory:
    f = open(file).readlines()
    f = [i.strip('\n') for i in f]

    final_email = f[:f.index("Original message")] #this list slicing will remove the part containing "Original message" and below it

    final_message = '\n'.join(final_email)

    f = open(file, 'w')

    f.write(final_message)

    f.close()


# In[42]:


cwd = os.getcwd()
#Load the images from images folder.
for f in glob.glob('fits'):   
    dir_name = get_dir_name(f)
    #To print the file name with path (path will be in string)
    print (dir_name)


# In[48]:


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io.fits import getdata
from astropy.io.fits import getheader

#headernote = 'Data and wavelength scale modified w/2nd order telluric correction.'
#wavestart = 3550.
#dwave = 0.0455
#infiles = "infiles_9246715.txt"
f1 = open(directory) # open a text file containing a list of FITS filenames

# NOTE brandnewspeclist is a previously-defined list of numpy arrays.
# Each array is ~155000 values long, and contains floats between -0.17 and 2.0.
# I tried adding 1.0 to all values so everything was > 0, but there was no change.

i = 0
# Loop over each file, this time to create new versions in new files.
for line in f1:
    infile = line.rstrip()
    outfile = 'new_' + infile

    # Read in the original FITS header
    head = getheader(infile)

    # Make a new FITS file with the new data and old header
    n = np.arange(100.0)
    #hdu = n #works with this
    hdu = brandnewspeclist[i] #doesn't work with this
    #print type(n), type(hdu) # I checked, and both are numpy.ndarrays
    fout = open('test' + str(i) + '.txt', 'w')
    for j in range(len(brandnewspeclist[i])):
        fout.write('%f \n' % brandnewspeclist[i][j]) # outfile doesn't have any nans

    # This is where it breaks and gives me a KeyError: 'object'
    # (unless I've set hdu = n instead of brandnewspeclist[i])
    fits.writeto(outfile, hdu, header=head)

    # I tried this approach instead, but got the same error
    #hdu = fits.PrimaryHDU(brandnewspeclist[i])
    #hdu.writeto(outfile, header=head)

    # Next step: create a new header object
    newhead = getheader(outfile)
    # Update the two header values to change the wavelength solution
    # Note: this isn't working either. The code ran when I used 'n' instead of
    # 'brandnewspeclist[i]' as hdu, but these header values didn't change.
    # At the moment, however, this is a secondary concern.
    newhead['cdelt1'] = (dwave, headernote)
    newhead['crval1'] = (wavestart, headernote)

    i = i + 1
f1.close()


# In[ ]:


from astropy.io import fits as pyfits
from astropy.table import Table, Column

# where is your data?
dir = "/Users/Frank/KNAC_Internship/fits"

# pick the header keys you want to dump to a table.
keys = ['NAXIS', 'RA', 'DEC', 'FILTER']
# pick the HDU you want to pull them from. It might be that your data are spectra, or FITS tables, or multi-extension "mosaics". 
hdu = 0

# get header keyword values
# http://docs.astropy.org/en/stable/io/fits/index.html#working-with-a-fits-header
values = []
fitsNames = []
for fitsName in glob.glob(dir+'*.fits'):
    # opening the file is unnecessary. just pull the (right) header
    header = pyfits.getheader(fitsName, hdu)
    values.append([header.get(key) for key in keys])
    fitsNames.append(fitsName)
    # if you want the fits file name only without the full path then
    # fitsNames.append(os.path.split(fitsName)[1])

# Create a table container. 
# http://docs.astropy.org/en/stable/table/construct_table.html
# One trick is to use the data types in the first "values" to let astropy guess datatypes.
# to use this trick, you need to specify the column names in the table
row0 = [dict(zip(keys, values[0]))]
t = Table(row0, names=keys)

# now add all the other rows. again, because dict didn't preserve column order, you have to repeat
# the dict here.
for i in range(1, len(values)):
    t.add_row(values[i])

# add the filenames column
#t.add_column
new_column = Column(name='fitsName', data=fitsNames)
t.add_column(new_column, 0)

# save the file
# http://docs.astropy.org/en/stable/table/io.html
t.write('table.dat', format='ascii.ipac')


# In[ ]:




