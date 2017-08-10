#!/usr/bin/env python

import numpy, scipy
from numpy import *
import pylab
import spimage
import os
from matplotlib.colors import LogNorm
import sys
from eke import image_manipulation
import cPickle as pickle
import time
from scipy.special import cbrt
from scipy.signal import argrelextrema
import h5py

def show_img_and_mask(input_file):
    im=spimage.sp_image_read(input_file,0)
    if len(shape(im.image))==2:
        fig=pylab.figure('image and mask',figsize=(20,10))
        fig.clear()
        fig.add_subplot(1,2,1)
        pylab.imshow(numpy.absolute(im.image), norm=LogNorm())
        fig.add_subplot(1,2,2)
        pylab.imshow(numpy.absolute(im.mask))
    if len(shape(im.image))==3:
        fig=pylab.figure('image and mask',figsize=(10,10))
        fig.clear()
        fig.add_subplot(2,2,1)
        pylab.imshow(numpy.absolute(im.image[0,:,:]), norm=LogNorm())
        fig.add_subplot(2,2,2)
        pylab.imshow(numpy.absolute(im.mask[0,:,:]))
        fig.add_subplot(2,2,3)
        pylab.imshow(numpy.absolute(im.image[shape(im.image)[0]/2,:,:]), norm=LogNorm())
        fig.add_subplot(2,2,4)
        pylab.imshow(numpy.absolute(im.mask[shape(im.image)[0]/2,:,:]))

def show_img_times_mask(input_file,s='x'):
    im=spimage.sp_image_read(input_file,0)
    if s=='x':
        sl=(shape(im.image)[0]/2,slice(None),slice(None))
    if s=='y':
        sl=(slice(None),shape(im.image)[0]/2,slice(None))
    if s=='z':
        sl=(slice(None),slice(None),shape(im.image)[0]/2)
    if len(shape(im.image))==2:
        fig=pylab.figure('image and mask',figsize=(20,10))
        fig.clear()
        pylab.imshow(numpy.absolute(im.image)*im.mask, norm=LogNorm())
    if len(shape(im.image))==3:
        fig=pylab.figure('image and mask',figsize=(10,10))
        fig.clear()
        fig.add_subplot(2,1,1)
        pylab.imshow(numpy.absolute(im.image[0,:,:])*im.mask[0,:,:], norm=LogNorm())
        fig.add_subplot(2,1,2)
        pylab.imshow(numpy.absolute(im.image[sl])*im.mask[sl], norm=LogNorm())

    
        
def show_slice(input_file):
    im=spimage.sp_image_read(input_file,0)
    fig=pylab.figure(1, figsize=(10,10))
    fig.clear()
    pylab.imshow(numpy.absolute(im.image)[32,:,:])

def show_img_and_phase(input_file, shift=True):
    im=spimage.sp_image_read(input_file,0)
    if shift:
        im=spimage.sp_image_shift(im)
    s=shape(im.image)[0]/2
    fig=pylab.figure(1,figsize=(20,10))
    fig.clear()
    fig.add_subplot(1,2,1)
    pylab.imshow(numpy.absolute(im.image[s,:,:]))
    fig.add_subplot(1,2,2)
    pylab.imshow(numpy.angle(im.image[s,:,:]),cmap='PiYG')
    
def add_mask(path, output_file):
    imgs=[path+i for i in os.listdir(path) if i.endswith('.h5')]
    for img_file in imgs:
        img=spimage.sp_image_read(img_file,0)
        try:
            msk_array=msk_array+img.mask
        except NameError:
            msk_array=img.mask
    msk_array[msk_array<msk_array.max()]=0
    msk_array[msk_array!=0]=1
    new=spimage.sp_image_alloc(numpy.shape(img.image)[0], numpy.shape(img.image)[1],1)
    new.mask[:,:]=msk_array
    new.image[:,:]=msk_array
    spimage.sp_image_write(new,output_file,0)

def add_mask_center(img_path, pref_mask_img, output_file, FINAL_SIZE=276):
    imgs=[img_path+i for i in os.listdir(img_path) if i.endswith('.h5')]
    mask_img=spimage.sp_image_read(pref_mask_img,0)
    orig_center=mask_img.detector.image_center
    for img_file in imgs:
        img=spimage.sp_image_read(img_file,0)
        center=img.detector.image_center
        cropped_mask = image_manipulation.crop_and_pad(mask_img.mask, (orig_center[1], center[1], orig_center[0]+center[0]), FINAL_SIZE)
        try:
            msk_array=msk_array+cropped_mask
        except NameError:
            msk_array=cropped_mask
    msk_array[msk_array<msk_array.max()]=0
    msk_array[msk_array!=0]=1
    new=spimage.sp_image_alloc(FINAL_SIZE,FINAL_SIZE,1)
    new.mask[:,:]=msk_array
    new.image[:,:]=msk_array
    spimage.sp_image_write(new,output_file,0)

def prep_emc_for_phasing(input_file, output_file):
    img=spimage.sp_image_read(input_file,0)
    #set all negative values to 0:
    img.image[:,:,:]=img.image.clip(0)
    #set corners of mask to 1:
    z_array = arange(img.image.shape[0]) - img.image.shape[0]/2. + 0.5
    y_array = arange(img.image.shape[1]) - img.image.shape[1]/2. + 0.5
    x_array = arange(img.image.shape[2]) - img.image.shape[2]/2. + 0.5
    r_array = sqrt(x_array[:]**2 + y_array[:, newaxis]**2 + z_array[:, newaxis, newaxis]**2)
    img.mask[r_array>img.image.shape[0]/2. + 0.5]=1
    spimage.sp_image_write(img,output_file,0)

def slice_3D(fn, output_dir=''):
    #Takes 3 slices through the center along x,y and z from a 3D image and saves it as three new 2D images.
    img_3D=spimage.sp_image_read(fn, 0)
    dim=shape(img_3D.image)[0]
    img_2D=spimage.sp_image_alloc(dim,dim,1)
    img_2D.shifted=img_3D.shifted
    img_2D.scaled=img_3D.scaled
    if img_3D.shifted==0:
        s=dim/2
    else:
        s=0

    img_2D.image[:,:]=img_3D.image[s,:,:]
    img_2D.mask[:,:]=img_3D.mask[s,:,:]
    spimage.sp_image_write(img_2D, output_dir+fn.split('.')[0]+'_x_slice.h5',0)

    img_2D.image[:,:]=img_3D.image[:,s,:]
    img_2D.mask[:,:]=img_3D.mask[:,s,:]
    spimage.sp_image_write(img_2D, output_dir+fn.split('.')[0]+'_y_slice.h5',0)
    
    img_2D.image[:,:]=img_3D.image[:,:,s]
    img_2D.mask[:,:]=img_3D.mask[:,:,s]
    spimage.sp_image_write(img_2D, output_dir+fn.split('.')[0]+'_z_slice.h5',0)


def plot_img_with_circle(image_file_name, r=20.):
    #Shows an image with a circle of given radius around the image center
    import matplotlib.pyplot as plt
    img=spimage.sp_image_read(image_file_name,0)
    print shape(img.image)
    circle1=plt.Circle(img.detector.image_center,r,color='r', fill=False)
    fig = plt.gcf()
    negatives=real(img.image)
    print shape(negatives)
    negatives[real(img.image)>0.]=0.
    pylab.imshow(absolute(img.image))
    #pylab.imshow(absolute(negatives), norm=LogNorm())
    fig.gca().add_artist(circle1)

def put_neg_to_zero(image_file_name,output_file_name):
    img=spimage.sp_image_read(image_file_name,0)
    new_img=img.image
    new_img[real(img.image)<0.]=0.
    img.image[:,:]=new_img
    spimage.sp_image_write(img, output_file_name, 0)

def radial_average(data, center):
    y,x = numpy.indices((data.shape)) # first determine radii of all pixels
    r = numpy.sqrt((x-center[0])**2+(y-center[1])**2)
    ind = numpy.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(numpy.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = numpy.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = numpy.cumsum(sim, dtype=numpy.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return radialprofile

def add_new_mask(img_file_name, new_mask_file_name, output_file_name, imgtimesmask=False):
    img=spimage.sp_image_read(img_file_name,0)
    new_mask=spimage.sp_image_read(new_mask_file_name,0)
    new_img=spimage.sp_image_alloc(*shape(img.image))
    new_img.image[:,:,:]=img.image
    new_img.shifted=img.shifted
    new_img.scaled=img.scaled
    new_img.detector=img.detector
    new_img.mask[:,:,:]=new_mask.mask
    spimage.sp_image_write(new_img, output_file_name,0)
    if imgtimesmask:
        new_img.image[:,:,:]=img.image*new_mask.mask
        spimage.sp_image_write(new_img, 'imgtimesmask.h5',0)

def mask_center(img_file_name, radius, output_file_name, save_file=True):
    """Create a new mask around the center with given radius"""
    img=spimage.sp_image_read(img_file_name,0)
    z_array = arange(img.image.shape[0]) - img.image.shape[0]/2. + 0.5
    y_array = arange(img.image.shape[1]) - img.image.shape[1]/2. + 0.5
    x_array = arange(img.image.shape[2]) - img.image.shape[2]/2. + 0.5
    r_array = sqrt(x_array[:]**2 + y_array[:, newaxis]**2 + z_array[:, newaxis, newaxis]**2)
    img.mask[r_array<radius]=0
    img.mask[r_array>radius]=1
    new_mask=spimage.sp_image_alloc(*shape(img.mask))
    new_mask.mask[:,:,:]=img.mask
    new_mask.image[:,:,:]=img.image
    new_mask.shifted=img.shifted
    new_mask.scaled=img.scaled
    new_mask.detector=img.detector
    if save_file:
        spimage.sp_image_write(new_mask, output_file_name, 0)
    else:
        return(new_mask)

def mask_center_and_negatives(img_file_name, radius, output_file_name, save_file=True):
    """Creates a new mask around the center with given radius as well as masking out regions with negative values in the image."""
    msk=mask_center(img_file_name, radius, output_file_name, save_file=False)
    img=spimage.sp_image_read(img_file_name,0)
    msk.mask[real(img.image)<0.]=0
    if save_file:
        spimage.sp_image_write(msk, output_file_name, 0)
    else:
        return(msk)

def plot_errors(dir='.'):
    """ Plots the errors saved in efourier.data and ereal.data files in given directory """
    efourier=numpy.genfromtxt(dir+'/efourier.data')
    ereal=numpy.genfromtxt(dir+'/ereal.data')
    fig=pylab.figure('Phasing errors')
    pylab.plot(range(len(efourier)), efourier, lw=2., c='r', label='Fourier error')
    pylab.plot(range(len(ereal)), ereal, lw=2., c='k', label='Real error')
    pylab.legend()
    pylab.show()

def extract_final_errors(path=os.getcwd(), pickle_output=True):
    print path
    errors=[]
    rundirs=[d for d in os.listdir(path) if d.startswith('run_')]
    rundirs.sort()
    for d in rundirs:
        try:
            errors.append([numpy.genfromtxt(d+'/efourier.data')[-1], numpy.genfromtxt(d+'/ereal.data')[-1], d])
        except:
            print d + ' is not a directory or does not contain error files'
    Structured_errors=numpy.core.records.fromarrays(array(errors).transpose(),names='fourier_error, real_error, run', formats = 'f8, f8, S8')
    if pickle_output:
        pickle.dump(Structured_errors, open(path+'/final_errors.p', 'wb'))
    return(Structured_errors)
    
def plot_final_error_hist(b=10):
    try:
        errors=pickle.load(open('final_errors.p', 'wb'))
    except:
        errors=extract_final_errors()
    fig=pylab.figure()
    fig.clear()
    fig.add_subplot(2,1,1)
    pylab.hist(errors['fourier_error'], bins=b, label='Fourier error')
    pylab.legend()
    fig.add_subplot(2,1,2)
    pylab.hist(errors['real_error'], bins=b, label='Real error')
    pylab.legend()
    
def plot_final_errors():
    try:
        errors=pickle.load(open('final_errors.p', 'rb'))
    except:
        errors=extract_final_errors()
    fig=pylab.figure('Errors')
    #fig.clear()
    fig.add_subplot(2,1,1)
    errors.sort(order='fourier_error')
    pylab.plot(range(len(errors['fourier_error'])), errors['fourier_error'], label='Fourier error')
    pylab.legend()
    fig.add_subplot(2,1,2)
    errors.sort(order='real_error')
    pylab.plot(range(len(errors['real_error'])), errors['real_error'], label='Real error')
    pylab.legend()

def select_by_error_cutoff(cutoff):
    try:
        errors=pickle.load(open(path+'/final_errors.p', 'rb'))
    except:
        errors=extract_final_errors()
    return errors[where(errors['real_error']<cutoff)]

def calc_prtf_subset(cutoff):
    errors=select_by_error_cutoff(cutoff)
    l=[f+'/output_net/model_4009.h5' for f in errors['run']]
    output_dir='prtf_output_{c}_{t}'.format(c=str(cutoff),t=time.strftime('%Y%m%d'))
    os.mkdir(output_dir)
    command='prtf {output_dir}/PRTF {list_of_files}'.format(output_dir=output_dir, list_of_files=' '.join(l))
    print command
    os.system(command)

def calc_prtf_all():
    try:
        errors=pickle.load(open('final_errors.p', 'rb'))
    except:
        errors=extract_final_errors()
    l=[f+'/output_net/model_4009.h5' for f in errors['run']]
    output_dir='prtf_output_all_{t}'.format(t=time.strftime('%Y%m%d'))
    os.mkdir(output_dir)
    command='prtf {output_dir}/PRTF {list_of_files}'.format(output_dir=output_dir, list_of_files=' '.join(l))
    print command
    os.system(command)

def plot_prtf(d):
    p=genfromtxt(os.path.join(d,'PRTF'))
    pylab.figure('PRTF')
    pylab.plot(p[:,0], p[:,1])
    pylab.axhline(1/e)      

def plot_from_file(f):
    d=genfromtxt(f)
    fig=pylab.figure()
    pylab.plot(len(d),d, lw=2.)

def calc_average_img(cutoff):
    run_folders=select_by_error_cutoff(cutoff)['run']
    for r in run_folders:
        img=spimage.sp_image_read(r+'/output_net/model_4009.h5',0)
        try:
            added_imgs=added_imgs+img.image
        except NameError:
            added_imgs=img.image
    avg_img=added_imgs/float(len(run_folders))
    new=spimage.sp_image_alloc(*shape(img.image))
    new.image[:,:,:]=avg_img
    spimage.sp_image_write(new,'average_final_model_{c}_{t}.h5'.format(c=cutoff,t=time.strftime('%Y%m%d')),0)

def calc_average_img_2(files, output_file_name):
    for f in files:
        img=spimage.sp_image_read(f,0)
        try:
            added_imgs=added_imgs+img.image
        except NameError:
            added_imgs=img.image
    avg_img=added_imgs/float(len(files))
    new=spimage.sp_image_alloc(*shape(img.image))
    new.image[:,:,:]=avg_img
    spimage.sp_image_write(new,output_file_name,0)

# for i,r in enumerate(errors['run']):
#     img=spimage.sp_image_read(r+'/output/support_4009.h5',0)
#     if i%10==0:
#         print i, r
#         if i>0:
#             new=spimage.sp_image_alloc(*shape(img.image))
#             new.image[:,:,:]=added_img
#             spimage.sp_image_write(new,'cumulative_support_{index}.h5'.format(index=i/10),0)
#         added_img=img.image
#     else:
#         added_img=added_img+img.image

def create_local_output_symlink(local_folder):
    dirs=[d for d in os.listdir(local_folder) if d.startswith('run_')]
    for d in dirs:
        try:
            scratch_folder='/net/davinci'+os.readlink(d+'/output')
        except OSError:
            print d+ ' has no symlinked output folder'
        try:
            os.symlink(scratch_folder, os.path.join(d, 'output_net'))
        except OSError:
            os.unlink(os.path.join(d, 'output_net'))
            os.symlink(scratch_folder, os.path.join(d, 'output_net'))

#def convert_condor_cxi_to_spimage(input_cxi_file_name, mask=None, output_file):
#    input_cxi=h5py.File(input_cxi_file_name, 'r')
#    data=input_cxi['data']
#    spimage.sp_image_alloc()
#    input_cxi.close()

def radial_average(data, center):
    y,x = numpy.indices((data.shape)) # first determine radii of all pixels
    r = numpy.sqrt((x-center[0])**2+(y-center[1])**2)
    ind = numpy.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(numpy.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = numpy.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = numpy.cumsum(sim, dtype=numpy.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer
    return radialprofile

def plot_radial_average(image_name, return_r=True):
    if type(image_name)==str:
        img=spimage.sp_image_read(image_name,0).image
    elif type(image_name)==numpy.array:
        img=image_name
    else:
        img=image_name.image
    dim=shape(img)[0]
    if len(shape(img))==3:
        img_slice=img[dim/2,:,:]
    else:
        img_slice=img

    radavg=radial_average(img_slice, (dim/2,dim/2))
    pylab.figure('radavgs')
    pylab.plot(log10(radavg))
    if return_r:
        return(radavg)

def michelson_contrast(radial_average_profile, norm=True,plot=True):
    max_positions=argrelextrema(radial_average_profile, numpy.greater)
    min_positions=argrelextrema(radial_average_profile, numpy.less)
    n=min(len(max_positions[0]), len(min_positions[0]))
    max_positions2=max_positions[0][:n]
    min_positions2=min_positions[0][:n]
    max_vals=radial_average_profile[max_positions2]
    min_vals=radial_average_profile[min_positions2]
    if norm:
        min_vals=min_vals/max_vals[0]
        ravg=radial_average_profile/max_vals[0]        
        max_vals=max_vals/max_vals[0]
    else:
        ravg=radial_average_profile
    print max_vals
    if plot:
        pylab.figure('Contrast')
        pylab.plot(ravg)
        pylab.plot(max_positions2,max_vals, 'ro')
        pylab.plot(min_positions2,min_vals, 'ko')
    return (max_vals-min_vals)/(max_vals+min_vals)

def percentage_contrast(radial_average_profile, plot=True):
    '''Returns the difference of maxima minus minima divided by fringe maxima'''
    max_positions=argrelextrema(radial_average_profile, numpy.greater)[0]
    min_positions=argrelextrema(radial_average_profile, numpy.less)[0]
    if radial_average_profile[0]==0.:
        max_positions=max_positions[1:]
    n=min(len(max_positions), len(min_positions))
    max_positions=max_positions[:n]
    min_positions=min_positions[:n]
    max_vals=radial_average_profile[max_positions]
    min_vals=radial_average_profile[min_positions]
    if plot:
        pylab.figure('Contrast')
        pylab.plot(radial_average_profile)
        pylab.plot(max_positions,max_vals, 'ro')
        pylab.plot(min_positions,min_vals, 'ko')
    return (max_vals-min_vals)/(max_vals)

        
    
def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    
    For example:
    from matplotlib import pyplot as pp
    from scipy.misc import lena
    
    matrix = lena()
    mask = sector_mask(matrix.shape,(200,100),300,(0,50))
    matrix[~mask] = 0
    pp.imshow(matrix)
    pp.show()
    """
    x,y = numpy.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = numpy.deg2rad(angle_range)
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*numpy.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = numpy.arctan2(x-cx,y-cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2*numpy.pi)
    # circular mask
    circmask = r2 <= radius*radius
    # angular mask
    anglemask = theta <= (tmax-tmin)
    return circmask*anglemask

def image_radial_section(image, center, radius, angle_range, show_slice=True):
    x,y=numpy.ogrid[:shape(image)[0],:shape(image)[1]]
    cx, cy = center
    r=numpy.sqrt((x-cx)**2+(y-cy)**2)
    tmin, tmax = numpy.deg2rad(angle_range)
    theta=numpy.arctan2(x-cx, y-cy) - tmin
    theta %= (2*numpy.pi)
    circmask = r <= radius
    anglemask = theta <= (tmax-tmin)
    radval=r[where(circmask*anglemask)]
    #radval=(r*(circmask*anglemask)).flatten()
    imgval=image[where(circmask*anglemask)]
    #imgval=(image*(circmask*anglemask)).flatten()
    if show_slice:
        pylab.figure('Current slice')
        pylab.imshow(image*(circmask*anglemask))
    ind=argsort(radval)
    imgval_sorted=imgval[ind]
    a,b=histogram(radval[ind], radius)
    i=numpy.zeros(radius+1)
    i[1:]=cumsum(a)
    binned_arr=numpy.zeros(radius)
    for j in range(len(i)-1):
        binned_arr[j]=sum(imgval_sorted[i[j]:i[j+1]])/a[j]
    return(binned_arr)

def plot_radial_section_averages(image, center, radius, angle_range_full=360, steps=36):
    for a in linspace(0,angle_range_full,steps):
        rad_sect=image_radial_section(image, center, radius, (a, a+angle_range_full/steps), show_slice=False)
        print len(rad_sect)
        pylab.plot(log10(rad_sect))
        
#gauss = lambda x, height, mu, sig: height*(1/(sig*sqrt(2*pi)))*exp(-(x-mu)**2/2*sig**2)
gauss = lambda x, height, mu, sig: height*exp(-(x-mu)**2/2*sig**2)
def gaussian2D(side, height, mu1, sig1, mu2=None, sig2=None):
    #gauss = lambda x, mu, sig: (1/(sig*sqrt(2*pi)))*exp(-(x-mu)**2/2*sig**2)
    if mu2==None:
        mu2=mu1
    if sig2==None:
        sig2=sig1
    X,Y=meshgrid(gauss(arange(side), height, mu1, sig1), gauss(arange(side), height, mu2, sig2))
    return sqrt(X*Y)


def gaussian3D(side, height, mu1, sig1):
    X,Y,Z=meshgrid(gauss(arange(side), height, mu1, sig1),gauss(arange(side), height, mu1, sig1),gauss(arange(side), height, mu1, sig1))
    return cbrt(X*Y*Z)

def deconvolve_gaussian(img_array, gaussian_r, epsilon=0., dim=3):
    if dim==3:
        kernel=spimage.sp_gaussian_kernel(gaussian_r, *img_array.shape).image[:]
    if dim==2:
        kernel=spimage.sp_gaussian_kernel(gaussian_r, img_array.shape[0], img_array.shape[1], 1).image[:]
    img_fft=fft.fftn(fft.fftshift(img_array))
    kernel_fft=fft.fftn(fft.fftshift(kernel))+epsilon
    deconvolved_img=fft.fftshift(fft.ifftn(img_fft/kernel_fft))
    return absolute(deconvolved_img).clip(0.1), kernel
    
def centrosymmeterize(centro, x, y):
    centro.image[x:,:y]=flipud(fliplr(centro.image[:x,y:]))
    centro.mask[x:,:y]=flipud(fliplr(centro.mask[:x,y:]))
    return centro

def downsample_by_average(image, file_type, sf):
    """image can be either spimage object or numpy array, file_type = spimage_object or numpy_array, sf is the downsampling factor"""
    if file_type == 'spimage_object':
        model_side=shape(image.image)[0]
    if file_type == 'numpy_array':
        model_side=shape(image)[0]
    downsampled=spimage.sp_image_alloc(model_side/sf,model_side/sf,1)
    for i, x in enumerate(range(0, model_side, sf)):
        for j, y in enumerate(range(0, model_side, sf)):
            if file_type == 'spimage_object':
                downsampled.image[i,j]=average(image.image[x:x+sf,y:y+sf])
                downsampled.mask[i,j]=int((image.mask[x:x+sf,y:y+sf]).all())
            if file_type == 'numpy_array':
                downsampled.image[i,j]=average(image[x:x+sf,y:y+sf])
    return downsampled

def smooth(y, box_pts):
    box = numpy.ones(box_pts)/box_pts
    y_smooth = numpy.convolve(y, box, mode='same')
    return y_smooth

def calc_resolution_at_pix(pix):
    return 1.035e-09*(0.7317/(pix/2*0.0003))

def calc_support_trinity(pix, pixsize):
    R=1.035e-09*(0.7317/(pix/2*pixsize))
    return 220e-09/R

def calc_support(particle_size, wl, dd, Npix, pixsize):
    R=wl*(dd/(Npix/2*pixsize))
    return particle_size/R

def r_array_3D(dim, center=None):
    if center==None:
        z_array = arange(dim) - dim/2. + 0.5
        y_array = arange(dim) - dim/2. + 0.5
        x_array = arange(dim) - dim/2. + 0.5
    else:
        z_array = arange(dim) - center[0]
        y_array = arange(dim) - center[1]
        x_array = arange(dim) - center[2]
    return sqrt(x_array[:]**2 + y_array[:, newaxis]**2 + z_array[:, newaxis, newaxis]**2)

def r_array_2D(dim):
    y_array = arange(dim) - dim/2. + 0.5
    x_array = arange(dim) - dim/2. + 0.5
    return sqrt(x_array[:]**2 + y_array[:, newaxis]**2)

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = meshgrid(arange(nx), arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    r = sqrt(x**2 + y**2)
    theta = arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * cos(theta)
    y = r * sin(theta)
    return x, y

def reproject_2Dimage_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    #r_i = linspace(r.min(), r.max(), nx)
    #theta_i = linspace(theta.min(), theta.max(), ny)

    r_i = linspace(r.min(), r.max(), int(r.max()))
    theta_i = linspace(theta.min(), theta.max(), 360)
    
    theta_grid, r_grid = meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = vstack((xi, yi)) # (map_coordinates requires a 2xn array)
    
    zi = scipy.ndimage.map_coordinates(data.T, coords, order=1)
    #zi.reshape((nx, ny))
    zi2=zi.reshape((int(r.max()), 360))
    return zi2, r_i, theta_i

def Fourier_shell_correlation(F1, F2):
    return sum(dot(F1, conjugate(F2))) / sqrt(dot(sum(abs(F1)**2), sum(abs(F2)**2)))

def real_space_correlation_coefficient(Rho_ref, Rho_rec, support=None):
    '''The real-space correlation coefficient, RSCC, is a measure 
    of the similarity between an electron-density map calculated 
    directly from a structural model and one calculated from 
    experimental data. An advantage of techniques for evaluating 
    goodness of fit in real space is that they can be performed 
    for arbitrary sets of atoms. They are therefore used most often 
    in the refinement of biological macromolecular structures to 
    improve the model fit on a per-residue basis. This metric is 
    similar to the real-space residual RSR, but does not require 
    that the two densities be scaled against each other.'''
    if support==None:
        R1=Rho_ref*ones_like(Rho_ref)
        R2=Rho_rec*ones_like(Rho_rec)

    else:
        R1=Rho_ref[support==1]
        R2=Rho_rec[support==1]
        print '{} values used'.format(len(R1))
        
    return (sum(abs(R2-mean(R2)))*sum(abs(R1-mean(R1))))/sqrt(sum(abs(R2-mean(R2)))**2 * sum(abs(R1-mean(R1)))**2)

def real_space_residual(Rho_ref, Rho_rec, support=None, normalize=True):
    ''' The real-space residual, RSR, is a measure of the 
    similarity between an electron-density map calculated directly 
    from a structural model and one calculated from experimental 
    data. An advantage of techniques for evaluating goodness of 
    fit in real space is that they can be performed for arbitrary 
    sets of atoms. They are therefore used most often in the refinement 
    of biological macromolecular structures to improve the model fit on 
    a per-residue basis.
    The measure of similarity is often provided in the form of a graph 
    of RSR values against residue number, showing clearly which residues 
    give best and worst agreement with the experimental electron-density map. 
    For nucleic acid structures, RSR may also be calculated separately for base, 
    sugar and phosphate moieties of the nucleic acid monomer. RSR is generally 
    considered an excellent model-validation tool.'''
    if support==None:
        R1=Rho_ref*ones_like(Rho_ref)
        R2=Rho_rec*ones_like(Rho_rec)
    else:
        R1=Rho_ref[support==1]
        R2=Rho_rec[support==1]
    if normalize:
        R1/=R1.max()
        R2/=R2.max()
    return sum(abs(R2-R1))/sum(abs(R2+R1))




'''
def cxi_to_spimage(cxi_file, spimage_file_name, key_path='entry_1/data_1/data', mask=None):
    with h5py.File(cxi_file) as cxi:
        data=cxi[key_path][:]
    s=shape(data)
    print s
    if len(s)>3:
        sm=max(s)
        im=spimage.sp_image_alloc(sm,sm,sm)
        data=[0,:,:,:]
    if len(s)==3:
        im=spimage.sp_image_alloc(s)
    elif len(s)==2:
        im=spimage.sp_image_alloc(s+(1,))
    im.image[:]=data
    if mask!=None:
        im.mask[:]=mask
    spimage.sp_image_write(im, spimage_file_name,0)

'''



    

#-----------STOLEN FROM REDFLAMINGO (sizing_convexhull_ball_refine.py)
def high_pass_filter(image_size, sigma):
    #image_size = 1024
    x_array = arange(image_size) - image_size/2
    y_array = arange(image_size) - image_size/2
    X_array, Y_array = meshgrid(x_array, y_array)
    r = sqrt(X_array**2 + Y_array**2)
   
    kernel = (r/2.0/sigma)**4*exp(2.0-r**2/2.0/sigma**2)
    kernel[r > 2.0*sigma] = ones(shape(kernel))[r > 2.0*sigma]

    return kernel, r

def autocorrelation(I, sigma):
  
  image_size = int(min(I.shape)/2)*2


  #ensure square image with even number of pix
  d0 = (I.shape[0]-image_size)/2
  d1 = (I.shape[1]-image_size)/2
  I  =  I[d0:image_size+d0, d1:image_size+d1]  

  Ifilter = high_pass_filter( image_size,sigma )[0]
  AC_real = fft.fft2(I*Ifilter)
  a = fft.fftshift(abs(AC_real))
  
  return a

#----------------------------------------------------------
