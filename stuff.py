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


def mask_center_and_negatives(img_file_name, radius, output_file_name, save_file=True):
    """Creates a new mask around the center with given radius as well as masking out regions with negative values in the image."""
    msk=mask_center(img_file_name, radius, output_file_name, save_file=False)
    img=spimage.sp_image_read(img_file_name,0)
    msk.mask[real(img.image)<0.]=0
    if save_file:
        spimage.sp_image_write(msk, output_file_name, 0)
    else:
        return(msk)

def plot_from_file(f):
    d=genfromtxt(f)
    fig=pylab.figure()
    pylab.plot(len(d),d, lw=2.)

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
    

#-----------FROM REDFLAMINGO (sizing_convexhull_ball_refine.py)
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
