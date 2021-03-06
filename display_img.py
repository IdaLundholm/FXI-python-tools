#!/usr/bin/env python

import numpy as np
#from numpy import *
import spimage
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import sys
from eke import image_manipulation
import cPickle as pickle
import time
import FXI_python_tools.phasing_tools as ida_phasing
import FXI_python_tools.stuff as stuff

def show_img_and_mask(input_file):
    im=spimage.sp_image_read(input_file,0)
    if len(shape(im.image))==2:
        fig=plt.figure('image and mask',figsize=(20,10))
        fig.clear()
        fig.add_subplot(1,2,1)
        plt.imshow(np.absolute(im.image), norm=LogNorm())
        fig.add_subplot(1,2,2)
        plt.imshow(np.absolute(im.mask))
        plt.colorbar()
    if len(shape(im.image))==3:
        fig=plt.figure('image and mask',figsize=(10,10))
        fig.clear()
        fig.add_subplot(2,2,1)
        plt.imshow(np.absolute(im.image[0,:,:]), norm=LogNorm())
        fig.add_subplot(2,2,2)
        plt.imshow(np.absolute(im.mask[0,:,:]))
        fig.add_subplot(2,2,3)
        plt.imshow(np.absolute(im.image[shape(im.image)[0]/2,:,:]), norm=LogNorm())
        fig.add_subplot(2,2,4)
        plt.imshow(np.absolute(im.mask[shape(im.image)[0]/2,:,:]))

def show_img_times_mask(input_file,s='x', save_file=True, output_file_name=None):
    im=spimage.sp_image_read(input_file,0)
    if len(shape(im.image))==2:
        fig=plt.figure('image and mask',figsize=(10,10))
        fig.clear()
        plt.imshow(np.absolute(im.image)*im.mask, norm=LogNorm())
        plt.colorbar()
    if len(shape(im.image))==3:
        if s=='x':
            sl=(np.shape(im.image)[0]/2,slice(None),slice(None))
        if s=='y':
            sl=(slice(None),shape(im.image)[0]/2,slice(None))
        if s=='z':
            sl=(slice(None),slice(None),shape(im.image)[0]/2)
        fig=plt.figure('image and mask',figsize=(10,10))
        fig.clear()
        fig.add_subplot(2,1,1)
        plt.imshow(np.absolute(im.image[0,:,:])*im.mask[0,:,:], norm=LogNorm())
        fig.add_subplot(2,1,2)
        plt.imshow(np.absolute(im.image[sl])*im.mask[sl], norm=LogNorm())
    if save_file:
        if output_file_name==None:
            plt.savefig(input_file.replace('h5', 'png'))
        else:
            plt.savefig(output_file_name)

def show_x_y_z_slice_emc_output(input_file, prefix='slice fig', save_file=False):
    im=spimage.sp_image_read(input_file,0)
    fig=plt.figure(prefix, figsize=(30,10))
    c=np.shape(im.image)[0]/2
    fig.add_subplot(1,3,1)
    plt.title('X slice')
    plt.imshow(np.absolute(im.image[c,:,:])*im.mask[c,:,:], norm=LogNorm())
    fig.add_subplot(1,3,2)
    plt.title('Y slice')
    plt.imshow(np.absolute(im.image[:,c,:])*im.mask[:,c,:], norm=LogNorm())
    fig.add_subplot(1,3,3)
    plt.title('Z slice')
    plt.imshow(np.absolute(im.image[:,:,c])*im.mask[:,:,c], norm=LogNorm())
    if save_file:
        plt.savefig(prefix)

def show_x_y_z_slice(input_file, shift=True, mode='absolute', added_slice=False, one_slice=False, prefix='slice fig', save_file=False, mask=False, mask_array=None, mask_center=False, pix=13.5, log_scale=False):
    "Plot x, y and z slices of image. Choices: shift image, true or false, mode: absolute, angle, real or imag. added_slice true or false, one_slice true or false, shows only x slice if true, prefix sets image name, save_file true or false, mask true or false, if true masks image with mask, log_scale true (norm=LogNorm()) or false"
    if mode=='absolute': f=np.absolute
    elif mode=='angle': f=np.angle
    elif mode=='real': f=np.real
    elif mode=='imag': f=np.imag
    print f
    n=None
    if log_scale:
        n=LogNorm()
        
    if type(input_file)==str:
        print 'loading file with spimage'
        im=spimage.sp_image_read(input_file,0)

    if type(input_file)!=str:
        print 'image is not read'
        im=input_file

    cm='viridis'
    if mode!='absolute' and log_scale==False:
        cm='RdYlBu'
        cm='seismic'
        
    if mask:
        im.image[:]=im.image*im.mask

    if mask_array!=None:
        im.image[:]=im.image*mask_array

    if shift:
        im=spimage.sp_image_shift(im)
        
    if mask_center:
        r_array=stuff.r_array_3D(im.image.shape[0])
        mask=np.ones_like(im.image)
        mask[r_array<pix]=0
        im.image[:]=im.image*mask

    if im.image.min()<0. and (mode=='real' or mode=='imag'):
        vmax=np.absolute(im.image).max()
        vmin=-1*vmax
    else:
        vmax=None
        vmin=None
    c=np.shape(im.image)[0]/2
    z=int((c*2-np.count_nonzero(im.image[c, c, :]))/2.2)
    
    if one_slice:
        fig=plt.figure(prefix, figsize=(10,10))
        if added_slice:
            plt.imshow(f(np.sum(im.image, axis=0))[z:-z,z:-z], vmax=vmax, vmin=vmin, norm=n, cmap=cm)
        else:
            plt.imshow(f(im.image[c,z:-z,z:-z]), vmax=vmax, vmin=vmin, norm=n, cmap=cm)
    else:    
        fig=plt.figure(prefix, figsize=(30,10))
        fig.add_subplot(1,3,1)
        plt.title('X slice')
        if added_slice:
            plt.imshow(f(np.sum(im.image, axis=0))[z:-z,z:-z], vmax=vmax, vmin=vmin, norm=n, cmap=cm)
        else:
            plt.imshow(f(im.image[c,z:-z,z:-z]),vmax=vmax, vmin=vmin, norm=n, cmap=cm)
        fig.add_subplot(1,3,2)
        plt.title('Y slice')
        if added_slice:
            plt.imshow(f(np.sum(im.image, axis=1))[z:-z,z:-z], vmax=vmax, vmin=vmin, norm=n, cmap=cm)
        else:
            plt.imshow(f(im.image[z:-z,c,z:-z]), vmax=vmax, vmin=vmin, norm=n, cmap=cm)
        fig.add_subplot(1,3,3)
        plt.title('Z slice')
        if added_slice:
            plt.imshow(f(np.sum(im.image, axis=2))[z:-z,z:-z], vmax=vmax, vmin=vmin, norm=n, cmap=cm)
        else:
            plt.imshow(f(im.image[z:-z,z:-z,c]), vmax=vmax, vmin=vmin, norm=n, cmap=cm)
    if save_file:
        plt.savefig(prefix)

def show_center_speckle_mask(input_file, radius_array, prefix='center_speckle_mask', save_img=False):
    img=spimage.sp_image_read(input_file,0)
    n=len(radius_array)
    c=np.shape(img.image)[0]/2
    img_dim=np.max(radius_array)*1.5
    fig, ax = plt.subplots(figsize=(9, 3*n), nrows=n, ncols=3)
    for i,r in enumerate(radius_array):
        print i,r
        ax[i][0].set_title('Mask radius = '+str(r))
        mask=ida_phasing.mask_center(input_file, r, save_file=False)
        ax[i][0].imshow(np.absolute(img.image[c,c-img_dim:c+img_dim,c-img_dim:c+img_dim])*mask.mask[c,c-img_dim:c+img_dim,c-img_dim:c+img_dim], interpolation='nearest')
        ax[i][1].imshow(np.absolute(img.image[c-img_dim:c+img_dim,c,c-img_dim:c+img_dim])*mask.mask[c-img_dim:c+img_dim,c, c-img_dim:c+img_dim], interpolation='nearest')
        ax[i][2].imshow(np.absolute(img.image[c-img_dim:c+img_dim,c-img_dim:c+img_dim, c])*mask.mask[c-img_dim:c+img_dim,c-img_dim:c+img_dim, c], interpolation='nearest')
        
    plt.tight_layout()
    if save_img:
        plt.savefig(prefix)


def show_slice(input_file, shift=True):
    im=spimage.sp_image_read(input_file,0)
    if shift:
        im=spimage.sp_image_shift(im)
    s=np.shape(im.image)[0]/2
    fig=plt.figure(1, figsize=(10,10))
    fig.clear()
    plt.imshow(np.absolute(im.image)[s,:,:])

def show_img_and_phase(input_file, save_fig=False, output_file=None, shift=True):
    im=spimage.sp_image_read(input_file,0)
    if shift:
        im=spimage.sp_image_shift(im)
    s=np.shape(im.image)[0]/2
    fig=plt.figure(1,figsize=(20,10))
    fig.clear()
    fig.add_subplot(1,2,1)
    plt.imshow(np.absolute(im.image[s,:,:]))
    fig.add_subplot(1,2,2)
    plt.imshow(np.angle(im.image[s,:,:]),cmap='PiYG')
    if save_fig:
        plt.savefig(output_file)

def show_support_fmodel_model_slice(run_dir, iteration=None, save_imgs=False, output_folder='pngs'):
    if iteration==None:
        iteration=ida_phasing.return_last_iteration_integer(run_dir+'/output_mnt/')
    print os.getcwd(), iteration
    model=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/model_{i}.h5'.format(r=run_dir, i=iteration),0))
    fmodel=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration),0))
    support=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/support_{i}.h5'.format(r=run_dir, i=iteration),0))
    s=np.shape(model.image)[0]/2
    #matplotlib.rcParams.update({'font.size': 16})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))
    ax1.set_title('Model')
    ax1.imshow(np.absolute(model.image[s,:,:]))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(np.absolute(log10(fmodel.image[s,:,:])))
    ax3.set_title('Support')
    ax3.imshow(np.absolute(support.image[s,:,:]))
    ax4.set_title('Phase')
    ax4.imshow(np.angle(model.image[s,:,:]))
    if save_imgs:
        plt.savefig(os.path.join(output_folder,run_dir+'.png'))
        plt.close(f)

def show_absolute_phase_real(input_file, shift=True, save_img=False):
    im=spimage.sp_image_read(input_file,0)
    if shift:
        im=spimage.sp_image_shift(im)
    s=np.shape(im.image)[0]/2
    fig=plt.figure(1,figsize=(25,10))
    fig.clear()
    fig.add_subplot(1,3,1)
    plt.title('Absolute')
    plt.imshow(np.absolute(im.image[s,:,:]))
    fig.add_subplot(1,3,2)
    plt.title('Phase')
    plt.imshow(np.angle(im.image[s,:,:]),cmap='PiYG')
    fig.add_subplot(1,3,3)
    plt.title('Real part')
    m=array(np.absolute(np.real(im.image[64,:,:]).min()), np.absolute(np.real(im.image[s,:,:]).max())).max()
    plt.imshow(np.real(im.image[s,:,:]),vmin=-m, vmax=m, cmap='coolwarm')
    plt.tight_layout()
    if save_img:
        path='/'.join(input_file.split('/')[:-1])
        plt.savefig(os.path.join(path,'absolute_phase_real_x_slice.png'))

def plot_img_with_circle(image_file_name, r=20.):
    #Shows an image with a circle of given radius around the image center
    img=spimage.sp_image_read(image_file_name,0)
    print np.shape(img.image)
    circle1=plt.Circle(img.detector.image_center,r,color='r', fill=False)
    fig = plt.gcf()
    negatives=np.real(img.image)
    print shape(negatives)
    negatives[np.real(img.image)>0.]=0.
    plt.imshow(np.absolute(img.image))
    fig.gca().add_artist(circle1)

def show_support_fmodel_model_slice(run_dir, iteration=None, save_imgs=False, output_folder='pngs', s='x'):
    if iteration==None:
        iteration=ida_phasing.return_last_iteration_integer(run_dir+'/output_mnt/')
    print os.getcwd(), iteration
    model=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/model_{i}.h5'.format(r=run_dir, i=iteration),0))
    fmodel=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration),0))
    support=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/support_{i}.h5'.format(r=run_dir, i=iteration),0))
    sh=np.shape(model.image)[0]/2
    #matplotlib.rcParams.update({'font.size': 16})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))
    print s
    if s=='x':
        sl=(sh,slice(None),slice(None))
        print 'x'
    if s=='y':
        sl=(slice(None),sh,slice(None))
        print 'y'
    if s=='z':
        sl=(slice(None),slice(None),sh)
        print 'z'
    ax1.set_title('Model')
    ax1.imshow(np.absolute(model.image[sl]))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(log10(np.absolute(fmodel.image[sl])))
    ax3.set_title('Support')
    ax3.imshow(np.absolute(support.image[sl]))
    ax4.set_title('Phase')
    ax4.imshow(np.angle(model.image[sl]))
    if save_imgs:
        plt.savefig(os.path.join(output_folder,run_dir+'.png'))
        plt.close(f)

def show_support_fmodel_model_slice2(iteration, save_imgs=False, output_folder='pngs'):
    model=spimage.sp_image_shift(spimage.sp_image_read('model_{i}.h5'.format(i=iteration),0))
    fmodel=spimage.sp_image_shift(spimage.sp_image_read('fmodel_{i}.h5'.format(i=iteration),0))
    support=spimage.sp_image_shift(spimage.sp_image_read('support_{i}.h5'.format(i=iteration),0))
    s=np.shape(model.image)[0]/2
    #matplotlib.rcParams.update({'font.size': 16})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))
    ax1.set_title('Model')
    ax1.imshow(np.absolute(model.image[s,:,:]))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(log10(np.absolute(fmodel.image[s,:,:])))
    ax3.set_title('Support')
    ax3.imshow(np.absolute(support.image[s,:,:]))
    ax4.set_title('Phase')
    ax4.imshow(np.angle(model.image[s,:,:]))
    if save_imgs:
        plt.savefig(os.path.join(output_folder,'{i}.png'.format(i=iteration)))
        plt.close(f)

def read_phasing_params(path):
    f=open(os.path.join(path,'phasing_params.py'),'r')
    lines=f.readlines()
    f.close()
    phase_dict={}
    for l in lines:
        l_split=l.split('=')
        try:
            val=l_split[1].replace('/n','')
            phase_dict[l_split[0].strip()]=val.strip()
        except IndexError:
            None
    return phase_dict

def plot_errors(dir_list):
    phasing_params=read_phasing_params('.')
    static_it=int(phasing_params['NUMBER_OF_STATIC_ITERATIONS'])
    support_it=static_it+int(phasing_params['NUMBER_OF_SUPPORT_ITERATIONS'])
    static_it2=support_it+int(phasing_params['NUMBER_OF_STATIC_ITERATIONS_2'])
    refine_it=static_it2+int(phasing_params['NUMBER_OF_REFINE_ITERATIONS'])
    op=int(phasing_params['OUTPUT_PERIOD'])
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    for d in dir_list:
        fourier_errors=np.genfromtxt(os.path.join(d,'efourier.data'))
        real_errors=np.genfromtxt(os.path.join(d,'ereal.data'))
        ax1.plot(linspace(0,len(fourier_errors)*op,len(fourier_errors)),fourier_errors, label=d)
        ax2.plot(linspace(0,len(real_errors)*op,len(real_errors)),real_errors, label=d)
    ax1.set_title('Fourier Errors')
    ax1.fill_between([static_it, support_it], 1, facecolor='0.8', edgecolor='0.8')
    ax1.fill_between([static_it2,refine_it], 1, facecolor='0.8', edgecolor='0.8')
    ax1.text(op,0.8,'Static')
    ax1.text(static_it+op,0.8,'Area support update')
    ax1.text(support_it+op,0.8,'Static')
    ax1.text(static_it2+op,0.8,'ER')
    ax2.set_title('Real Errors')
    ax2.fill_between([static_it, support_it], 1, facecolor='0.8', edgecolor='0.8')
    ax2.fill_between([static_it2, refine_it], 1, facecolor='0.8', edgecolor='0.8')
    ax2.text(op,0.8,'Static')
    ax2.text(static_it+op,0.8,'Area support update')
    ax2.text(support_it+op,0.8,'Static')
    ax2.text(static_it2+op,0.8,'ER')

def show_2Dhawk_result_real_fourier(iteration):
    re=spimage.sp_image_shift(spimage.sp_image_read('real_space-{0:07}.h5'.format(iteration),0))
    fo=spimage.sp_image_shift(spimage.sp_image_read('fourier_space-{0:07}.h5'.format(iteration),0))
    f, (ax1,ax2) = plt.subplots(1,2, figsize=(15,8))
    ax1.imshow(np.absolute(re.image))
    ax2.imshow(log10(absolute(fo.image)))
    plt.tight_layout()
    plt.savefig('real_and_fourier_{i}.png'.format(i=str(iteration)))

def plot_real_space(n, png_output='real_space_pngs', iteration='4009'):
    for j in range(n):
        fig, axes = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)
        real_space=np.real(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/model_%s.h5'%(j,iteration),0)).image)
        dim=np.shape(real_space)[0]
        nonzero=np.shape(real_space[real_space>0.])[0]
        new_dim=(dim/2-nonzero**(1/3))/2
        vmax=real_space.max()
        vmin=real_space.min()
        axes[0].imshow(real_space[dim/2,new_dim:-new_dim,new_dim:-new_dim], vmax=vmax, vmin=vmin)
        axes[1].imshow(real_space[new_dim:-new_dim,dim/2,new_dim:-new_dim], vmax=vmax, vmin=vmin)
        im=axes[2].imshow(real_space[new_dim:-new_dim,new_dim:-new_dim,dim/2], vmax=vmax, vmin=vmin)
        fig.colorbar(im, ax=axes.ravel().tolist())
        print png_output+'/%04d.png'%j
        plt.savefig(png_output+'/%04d.png'%j)
        plt.close(fig)

def plot_individual_phasing_results(n, png_output, iteration, n_start=0, sort_by_error=True, error='default'):
    try:
        os.mkdir(png_output)
    except OSError:
        None
    if sort_by_error:
        if error=='default':
            error=pickle.load(open('final_errors.p'))['fourier_error']
        sorting_order=error.argsort()
    for j in range(n_start,n):
        fig, axes = plt.subplots(figsize=(9, 9), nrows=3, ncols=3)
        real_space=np.real(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/model_%s.h5'%(j,iteration),0)).image)
        real_phase=np.angle(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/model_%s.h5'%(j,iteration),0)).image)
        fourier_phase=np.angle(ida_phasing.phase_shift(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/fmodel_%s.h5'%(j,iteration),0))).image[:])
        fourier_amplitude=absolute(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/fmodel_%s.h5'%(j,iteration),0)).image)
        print fourier_amplitude.max()
        support=np.real(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/support_%s.h5'%(j,iteration),0)).image)
        dim=np.shape(real_space)[0]
        nonzero=np.shape(real_space[real_space>0.])[0]
        new_dim=(dim-nonzero**(1/3.))/2.4
        vmax=real_space.max()
        vmin=real_space.min()
        axes[0][0].imshow(real_space[dim/2,new_dim:-new_dim,new_dim:-new_dim], vmax=vmax, vmin=vmin)
        axes[0][1].imshow(real_space[new_dim:-new_dim,dim/2,new_dim:-new_dim], vmax=vmax, vmin=vmin)
        im=axes[0][2].imshow(real_space[new_dim:-new_dim,new_dim:-new_dim,dim/2], vmax=vmax, vmin=vmin)
        fig.colorbar(im, ax=axes[0][2])
        axes[1][0].imshow(fourier_amplitude[dim/2,:,:], norm=LogNorm()) #cmap='hsv')
        axes[1][1].imshow(fourier_amplitude[:,dim/2,:], norm=LogNorm()) #cmap='hsv')
        im2=axes[1][2].imshow(fourier_amplitude[:,:,dim/2], norm=LogNorm())#, cmap='hsv')
        fig.colorbar(im2, ax=axes[1][2])
        axes[2][0].imshow(real_phase[dim/2,new_dim:-new_dim,new_dim:-new_dim], cmap='RdBu')
        axes[2][1].imshow(real_phase[new_dim:-new_dim,dim/2,new_dim:-new_dim], cmap='RdBu')
        axes[2][2].imshow(real_phase[new_dim:-new_dim,new_dim:-new_dim,dim/2], cmap='RdBu')
        print str(sorting_order[j])
        plt.savefig(png_output+'/%s_%04d.png'%(str(sorting_order[j]), j))
        plt.close(fig)

def plot_model_fdiff(j, png_output, iteration):
    try:
        os.mkdir(png_output)
    except OSError:
        None                
    error=pickle.load(open('final_errors.p'))['fourier_error']
    sorting_order=error.argsort()
    fig, axes = plt.subplots(figsize=(9, 6), nrows=2, ncols=3)
    real_space=np.real(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/model_%s.h5'%(j,iteration),0)).image)
    fourier_phase=np.angle(ida_phasing.phase_shift(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/fmodel_%s.h5'%(j,iteration),0))).image[:])
    fourier_amplitude=spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/fmodel_%s.h5'%(j,iteration),0)).image
    input_amplitudes=spimage.sp_image_shift(spimage.sp_image_read('run_%04d/input_amplitudes.h5'%j,0)).image
    mask=spimage.sp_image_shift(spimage.sp_image_read('run_%04d/input_amplitudes.h5'%j,0)).mask
    support=np.real(spimage.sp_image_shift(spimage.sp_image_read('run_%04d/output_mnt/support_%s.h5'%(j,iteration),0)).image)
    dim=np.shape(real_space)[0]
    nonzero=np.shape(real_space[real_space>0.])[0]
    new_dim=(dim-nonzero**(1/3.))/2.4
    vmax=real_space.max()
    vmin=real_space.min()
    axes[0][0].imshow(real_space[dim/2,new_dim:-new_dim,new_dim:-new_dim], vmax=vmax, vmin=vmin)
    axes[0][1].imshow(real_space[new_dim:-new_dim,dim/2,new_dim:-new_dim], vmax=vmax, vmin=vmin)
    im=axes[0][2].imshow(real_space[new_dim:-new_dim,new_dim:-new_dim,dim/2], vmax=vmax, vmin=vmin)
    fig.colorbar(im, ax=axes[0][2])
    diff_amp=(np.absolute(fourier_amplitude-input_amplitudes))*mask
    axes[1][0].imshow(diff_amp[dim/2,:,:], vmax=30.)
    axes[1][1].imshow(diff_amp[:,dim/2,:], vmax=30.)
    axes[1][2].imshow(diff_amp[:,:,dim/2], vmax=30.)
    #fig.tight_layout()
    print str(sorting_order[j])
    plt.savefig(png_output+'/%s_%04d.png'%(str(sorting_order[j]), j))
    plt.close(fig)

def plot_fft_of_realspace_artifact(input_real_space, real_space_cutoff, tr=0., save_fig=False, fname='artefact_fig.png'):
    fig, ((ax1,ax2),(ax3, ax4)) = plt.subplots(figsize=(9,9), nrows=2, ncols=2)
    dim=np.shape(input_real_space)[0]
    ax1.imshow(real(input_real_space)[dim/2,tr:-tr,tr:-tr])
    ax2.imshow(absolute(fft.fftshift(fft.fftn(fft.fftshift(input_real_space))))[80,:,:])
    artefact=input_real_space.copy()
    artefact[input_real_space<real_space_cutoff]=complex(0.)
    ax3.imshow(real(artefact[dim/2,tr:-tr,tr:-tr]))
    artefact_fft=np.fft.fftshift(fft.fftn(fft.fftshift(artefact)))
    ax4.imshow(absolute(artefact_fft)[80,:,:])
    if save_fig:
        plt.savefig(fname)

def radavg_from_file(file_name, mode='absolute', mask=False, log_scale=False, shift=False, s='x', plot=False, do_return=False):
    im=spimage.sp_image_read(file_name,0)
    if shift:
        im=spimage.sp_image_shift(im)
    dim=len(np.shape(im.image))
    if dim==3 and s=='x':
        sl=(np.shape(im.image)[0]/2, slice(None), slice(None))
    elif dim==3 and s=='y':
        sl=(slice(None), np.shape(im.image)[1]/2, slice(None))
    elif dim==3 and s=='z':
        sl=(slice(None), slice(None), np.shape(im.image)[2]/2)
    elif dim==2:
        sl=(slice(None),slice(None))
    
    if mask:
        msk=im.mask
    if not mask:
        msk=np.ones_like(im.image)

    if mode=='absolute': f=np.absolute
    elif mode=='angle': f=np.angle
    elif mode=='real': f=np.real
    elif mode=='imag': f=np.imag

    radavg=spimage.radialMeanImage(f(im.image[sl]), msk=msk[sl]) #cx=im.detector.image_center[0], cy=im.detector.image_center[1])
    if log_scale:
        radavg=log10(radavg)
    if plot:
        plt.plot(radavg)
    if do_return:
        return radavg
