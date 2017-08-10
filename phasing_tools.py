#!/usr/bin/env python

import numpy
from numpy import *
import pylab
import spimage
import os
from matplotlib.colors import LogNorm
import matplotlib
import sys
from eke import image_manipulation
import cPickle as pickle
import time
import math
import Ida_python_tools.stuff as stuff
import h5py
from IPython import embed
#pylab.rc('axes', linewidth=1.5)
#pylab.rc('xtick', labelsize = 12.0)
#pylab.rc('ytick', labelsize = 12.0)


def prep_emc_output_for_phasing(input_file, output_file):
    "Sets all negative values in image to 0 and create a rectangular mask"
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

def mask_center(img_file_name, radius, output_file_name=None, save_file=True):
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


def slice_3D(fn, output_fn):
    #Takes 3 slices through the center along x,y and z from a 3D image and saves it as three new 2D images.
    img_3D=spimage.sp_image_read(fn, 0)
    dim=shape(img_3D.image)[0]
    img_2D=spimage.sp_image_alloc(dim,dim,1)
    img_2D.shifted=img_3D.shifted
    img_2D.scaled=img_3D.scaled
    if img_3D.shifted==0:
        s=dim/2
        
def plot_error_per_iteration(dir='.', with_legend=True):
    """ Plots the errors saved in efourier.data and ereal.data files in given directory """
    efourier=numpy.genfromtxt(dir+'/efourier.data')
    ereal=numpy.genfromtxt(dir+'/ereal.data')
    fig=pylab.figure('Phasing errors')
    pylab.plot(range(len(efourier)), efourier, lw=2., c='r', label='Fourier error')
    pylab.plot(range(len(ereal)), ereal, lw=2., c='k', label='Real error')
    if with_legend:
        pylab.legend()
    pylab.show()

def extract_final_errors(path, pickle_output=True):
    """Reads fourier and real errors for last phasing iteration and returns structured array"""
    try:
        Structured_errors=pickle.load(open(os.path.join(path,'/final_errors.p'), 'rb'))
    except:
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
    
def plot_final_error_hist(path, b=10):
    errors=extract_final_errors(path)
    fig=pylab.figure('Final Errors Histogram')
    fig.clear()
    fig.add_subplot(2,1,1)
    pylab.hist(errors['fourier_error'], bins=b, label='Fourier error')
    pylab.legend()
    fig.add_subplot(2,1,2)
    pylab.hist(errors['real_error'], bins=b, label='Real error')
    pylab.legend()
    
def plot_final_errors(path, sort=True):
    errors=extract_final_errors(path)
    fig=pylab.figure('Errors')
    #fig.clear()
    fig.add_subplot(2,1,1)
    if sort:
        errors.sort(order='fourier_error')
    pylab.title('Fourier error')
    pylab.plot(range(len(errors['fourier_error'])), errors['fourier_error'])
    fig.add_subplot(2,1,2)
    if sort:
        errors.sort(order='real_error')
    pylab.title('Real error')
    pylab.plot(range(len(errors['real_error'])), errors['real_error'])
    pylab.savefig(os.path.join(path, 'error_plot.png'))

def select_by_error_cutoff(path, cutoff):
    errors=extract_final_errors(path)
    return errors[where(errors['real_error']<cutoff)]

def return_last_iteration_integer(path):
    img_nr=[int(i.split('_')[-1][:-3]) for i in os.listdir(path) if i.endswith('.h5')]
    img_nr.sort()
    if len(img_nr)>1:
        return(img_nr[-1])
    else:
        print "{n} files in {p}".format(n=len(img_nr), p=path)

def calc_average_img(path, cutoff=None):
    if cutoff==None:
        run_folders=[]
    run_folders=select_by_error_cutoff(path, cutoff)['run']
    for r in run_folders:
        img=spimage.sp_image_read(r+'/output_mnt/model_{n}.h5'.format(n=return_last_iteration_integer(os.path.join(r,'output_mnt'))),0)
        try:
            added_imgs=added_imgs+img.image
        except NameError:
            added_imgs=img.image
    avg_img=added_imgs/float(len(run_folders))
    new=spimage.sp_image_alloc(*shape(img.image))
    new.image[:,:,:]=avg_img
    spimage.sp_image_write(new,'{p}/average_final_model_{c}_{t}.h5'.format(p=path,c=cutoff,t=time.strftime('%Y%m%d')),0)

def calc_prtf(cutoff='all'):
    if cutoff=='all':
        errors=extract_final_errors('.')
    else:
        errors=select_by_error_cutoff('.',cutoff)
    currdir=os.getcwd()
    if currdir.startswith('/mnt'):
        l=[f+'/output_mnt/model_{n}.h5'.format(n=return_last_iteration_integer(os.path.join(f,'output_mnt'))) for f in errors['run']]
    else:
        l=[f+'/output/model_{n}.h5'.format(n=return_last_iteration_integer(os.path.join(f,'output'))) for f in errors['run']]
    output_dir='prtf_output_{c}_{i}_{t}'.format(c=str(cutoff),i=str(len(l))+'imgs',t=time.strftime('%Y%m%d'))
    os.mkdir(output_dir)
    command='prtf {output_dir}/PRTF {list_of_files}'.format(output_dir=output_dir, list_of_files=' '.join(l))
    print command
    os.system(command)
    #ida_img.show_absolute_phase_real(os.path.join(output_dir, 'PRTF-avg_image.h5'), save_img=True)

def pix_to_q(x,wl, dd, ps):
    #1/q gives full period resolution
    return (ps*x)/(dd*wl)

def pix_to_res(x, wl, dd, ps):
    #resolution element corresponds to half period resolution
    return wl / 4. / numpy.sin( numpy.arctan2( x * ps, dd ) / 2. )



def pix_to_q_2(x,wl, dd, ps):
    return (2*ps)/(dd*wl)


def plot_prtf_broken(di=None,legend=True, downsampling=8., plot_beyond_edge=False):
    '''When input is none, plots all prtf_output* directories availiabe, other input may be one directory or a list of directories'''
    if di==None:
        dir_list=[i for i in os.listdir('.') if i.startswith('prtf_output')]
    elif type(di)==str:
        dir_list=[di]
    elif type(di)==list:
        dir_list=di
    else:
        'Nothing to plot :\'('
        return None
    for d in dir_list:
        p=genfromtxt(os.path.join(d,'PRTF'))
        img=spimage.sp_image_read(os.path.join(d,'PRTF-avg_image.h5'),0)
        s=shape(img.image)
        edge_num_of_pix=(sqrt(sum((array(s)*downsampling)**2))/2.)
        print edge_num_of_pix
        x=p[:,0].copy()
        x/=x.max()
        print x
        x*=edge_num_of_pix
        print x
        q=pix_to_q(x,1.035e-9, 0.7317, 0.000075)
        fig=pylab.figure('PRTF')
        #q=pix_to_q(1.035e-9, 0.7317, 0.0006)
        ax1=fig.add_subplot(111)
        ax2=ax1.twiny()
        if plot_beyond_edge:
            ax1.plot(q, p[:,1], '.-',lw=1.5, label=d)
        else:
            max_q=pix_to_q(s[0],1.035e-9, 0.7317, 0.000075*downsampling)
            new_q=q[q<=max_q]
            ax1.plot(new_q, p[:len(new_q),1], '.-',lw=1.5, label=d)
            #ax1.set_xlim(0,max_q)
            spimage.sp_image_free(img)
        ax1Ticks = ax1.get_xticks()   
        ax2Ticks = ax1Ticks
        
        def tick_function(X):
            V = 1e09/X
            return ["%.3f" % z for z in V]
        
        ax2.set_xticks(ax2Ticks)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(tick_function(ax2Ticks))
    if legend:
        ax1.legend()
    ax1.set_xlabel(r'$|q|[nm^{-1}]$', fontsize=18)
    ax2.set_xlabel(r'$resolution (nm)$', fontsize=18)
    ax1.set_ylabel(r'$PRTF$', fontsize=18)
    ax1.axhline(1/e, c='k')
    return q


def plot_prtf(di=None,legend=True, custom_legend=None, downsampling=8., plot_beyond_edge=False):
    '''When input is none, plots all prtf_output* directories availiabe, other input may be one directory or a list of directories'''
    if di==None:
        dir_list=[i for i in os.listdir('.') if i.startswith('prtf_output')]
        print dir_list
    elif type(di)==str:
        dir_list=[di]
    elif type(di)==list:
        dir_list=di
    else:
        'Nothing to plot :\'('
        return None
    for i,d in enumerate(dir_list):
        prtf=spimage.sp_image_read(os.path.join(d,'PRTF-prtf.h5'),0)
        s=shape(prtf.image)
        r,prtf_radavg=spimage.radialMeanImage(prtf.image, cx=0., cy=0., cz=0., output_r=True)
        q=pix_to_q(r,1.035e-9, 0.7317, 0.000075*downsampling)
        q/=1e09 #reciprocal nanometres
        fig=pylab.figure('PRTF')
        ax1=fig.add_subplot(111)
        ax2=ax1.twiny()
        if custom_legend!=None:
            l=custom_legend[i]
        else:
            l=d
        if plot_beyond_edge:
            ax1.plot(q, prtf_radavg, '.-',lw=1.5, label=l)
        else:
            max_q=pix_to_q(s[0]/2,1.035e-9, 0.7317, 0.000075*downsampling)/1e09
            new_q=q[q<=max_q]
            ax1.plot(new_q, prtf_radavg[:len(new_q)], '.-',lw=1.5, label=l)
        ax1Ticks = ax1.get_xticks()   
        ax2Ticks = ax1Ticks
        
        def tick_function(X):
            V = 1./X
            return ["%.3f" % z for z in V]
        
        ax2.set_xticks(ax2Ticks)
        ax2.set_xbound(ax1.get_xbound())
        ax2.set_xticklabels(tick_function(ax2Ticks))
    if legend:
        ax1.legend()
    ax1.set_xlabel(r'$|q|[nm^{-1}]$', fontsize=18)
    ax2.set_xlabel(r'Full period resolution ($nm$)', fontsize=18)
    ax1.set_ylabel(r'PRTF', fontsize=18)
    ax1.axhline(1/e, c='k')
    spimage.sp_image_free(prtf)
    return r,q


def create_local_output_symlink(local_folder):
    dirs=[d for d in os.listdir(local_folder) if d.startswith('run_')]
    for d in dirs:
        try:
            scratch_folder='/mnt/davinci'+os.readlink(d+'/output')
        except OSError:
            print d+ ' has no symlinked output folder'
        try:
            os.symlink(scratch_folder, os.path.join(d, 'output_mnt'))
        except OSError:
            os.unlink(os.path.join(d, 'output_mnt'))
            os.symlink(scratch_folder, os.path.join(d, 'output_mnt'))

def unlink_all():
    dirs=[d for d in os.listdir('.') if d.startswith('run_')]
    for d in dirs:
        try:
            os.unlink(os.path.join(d,'output'))
        except OSError:
            print 'no symlinked output folder in '+d
        try:
            os.unlink(os.path.join(d, 'output_mnt'))
        except OSError:
            print 'no symlinked output_mnt folder in '+d

def save_absolute_phase_real_fourier(run_dir, output_folder, i=None, shift=True, save_img=True):
    #matplotlib.use('Agg')
    #matplotlib.ioff()
    if i==None:
        i=return_last_iteration_integer(os.path.join(run_dir, 'output_mnt'))
    model=spimage.sp_image_read('{r}/output_mnt/model_{i}.h5'.format(r=run_dir,i=i),0)
    fmodel=spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir,i=i),0)
    if model==None or fmodel==None:
        print run_dir+' failed'
        return
    if shift:
        model=spimage.sp_image_shift(model)
        fmodel=spimage.sp_image_shift(fmodel)
    s=shape(model.image)[0]/2
    matplotlib.rcParams.update({'font.size': 16})
    #pylab.figure(1, figsize=(20,30))
    f, ((ax1, ax2), (ax3, ax4)) = pylab.subplots(2, 2, figsize=(15,15))#, sharex='col', sharey='row')
    #f(figsize=(10,15))
    #f.set_figheight=50
    #f.set_figweight=20
    ax1.set_title('Absolute')
    ax1.imshow(numpy.absolute(model.image[s,:,:]))
    ax2.set_title('Phase')
    ax2.imshow(numpy.angle(model.image[s,:,:]),cmap='PiYG')
    ax3.set_title('Real part')
    #m=array(numpy.absolute(numpy.real(model.image[s,:,:]).min()), numpy.absolute(numpy.real(model.image[s,:,:]).max())).max()
    #ax3.imshow(numpy.real(model.image[s,:,:]),vmin=-m, vmax=m, cmap='coolwarm')
    ax3.imshow(numpy.real(model.image[s,:,:]), cmap='coolwarm')
    ax4.set_title('Fourier')
    ax4.imshow(log10(absolute(fmodel.image[s,:,:])))
    #f.set_figheight(10.)
    #f.set_figwidth(5.)
    f.subplots_adjust(wspace=1,hspace=1)
    pylab.tight_layout()
    if save_img:
        pylab.savefig(os.path.join(output_folder,run_dir+'.png'))
    spimage.sp_image_free(model)
    spimage.sp_image_free(fmodel)
    pylab.close(f)

def save_pngs_all(i=None, output_folder='pngs', start_from=0):
    try:
        os.mkdir(output_folder)
    except:
        None
    rundirs=[d for d in os.listdir('.') if d.startswith('run_')]
    #matplotlib.ioff()
    rundirs.sort()
    for r in rundirs[start_from:]:
        save_absolute_phase_real_fourier(r, output_folder, i)

def show_support_fmodel_model_slice(run_dir, iteration=None, save_imgs=False, output_folder='pngs'):
    if iteration==None:
        iteration=return_last_iteration_integer(run_dir+'/output_mnt/')
    print os.getcwd(), iteration
    model=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/model_{i}.h5'.format(r=run_dir, i=iteration),0))
    fmodel=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration),0))
    support=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/support_{i}.h5'.format(r=run_dir, i=iteration),0))
    s=shape(model.image)[0]/2
    matplotlib.rcParams.update({'font.size': 16})
    f, ((ax1, ax2), (ax3, ax4)) = pylab.subplots(2, 2, figsize=(15,15))
    ax1.set_title('Model')
    ax1.imshow(absolute(model.image[s,:,:]))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(log10(absolute(fmodel.image[s,:,:])))
    ax3.set_title('Support')
    ax3.imshow(absolute(support.image[s,:,:]))
    ax4.set_title('Support x Model')
    ax4.imshow(absolute(support.image[s,:,:])*absolute(model.image[s,:,:]))
    if save_imgs:
        pylab.savefig(os.path.join(output_folder,run_dir+'.png'))
        pylab.close(f)

def show_support_fmodel_model_slice_2D(run_dir, iteration=None, save_imgs=False, output_folder='pngs'):
    if iteration==None:
        iteration=return_last_iteration_integer(run_dir+'/output_mnt/')
    print os.getcwd(), iteration
    model=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/model_{i}.h5'.format(r=run_dir, i=iteration),0))
    fmodel=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration),0))
    support=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/support_{i}.h5'.format(r=run_dir, i=iteration),0))
    s=shape(model.image)[0]/2
    matplotlib.rcParams.update({'font.size': 16})
    f, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=(30,10))
    ax1.set_title('Model')
    ax1.imshow(absolute(model.image))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(log10(absolute(fmodel.image)))
    ax3.set_title('Support')
    ax3.imshow(absolute(support.image))
    pylab.tight_layout() 
    if save_imgs:
        pylab.savefig(os.path.join(output_folder,run_dir+'.png'))
        pylab.close(f)
        
def save_pngs_all_support(output_folder='pngs', start_from=0):
    try:
        os.mkdir(output_folder)
    except:
        None
    rundirs=[d for d in os.listdir('.') if d.startswith('run_')]
    #matplotlib.ioff()
    rundirs.sort()
    for r in rundirs[start_from:]:
        show_support_fmodel_model_slice(r, save_imgs=True, output_folder=output_folder)

def plot_prtf_results_2D(d, input_image, shift_input=False, save_file=True, image_name = 'PRTF_results.png'):
    f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = pylab.subplots(2, 3, figsize=(30,20))
    ax1.set_title('Average Real space')
    avg_img=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_image.h5'), 0))
    avg_f=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_fft.h5'), 0))
    if shift_input:
        input_img=spimage.sp_image_shift(spimage.sp_image_read(input_image,0))
    else:
        input_img=spimage.sp_image_read(input_image,0)
    ax1.imshow(absolute(avg_img.image))
    ax2.set_title('Input pattern')
    ax2.imshow(absolute(input_img.image*input_img.mask), norm=LogNorm())
    ax3.set_title('PRTF')
    p=genfromtxt(os.path.join(d,'PRTF'))
    ax3.plot(pix_to_q(p[:,0],1.035e-9, 0.7317, 0.0006), p[:,1], lw=1.5)
    ax3.set_xlabel(r'$|q|[nm^{-1}]$', fontsize=18)
    ax3.set_ylabel(r'$PRTF$', fontsize=18)
    ax3.axhline(1/e, c='k')
    ax4.set_title('Phase')
    ax4.imshow(numpy.angle(avg_img.image),cmap='PiYG')
    ax5.set_title('Average Fourier Space')
    ax5.imshow(absolute(avg_f.image))
    ax6.set_title('Errors')
    errors=extract_final_errors(os.getcwd())
    errors.sort(order='fourier_error')
    ax6.plot(range(len(errors['fourier_error'])), errors['fourier_error'], lw=1.5, c='r', label='Fourier Error')
    errors.sort(order='real_error')
    ax6.plot(range(len(errors['real_error'])), errors['real_error'], lw=1.5, c='g', label='Real Error')
    pylab.legend()
    pylab.tight_layout()
    if save_file:
        pylab.savefig(image_name)
        
def plot_prtf_results_3D(d, input_image, downsampling, save_file=True, image_name = 'PRTF_results.png'):
    f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = pylab.subplots(2, 3, figsize=(20,13))
    ax1.set_title('Average Real space')
    avg_img=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_image.h5'), 0))
    avg_f=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_fft.h5'), 0))
    input_img=spimage.sp_image_read(input_image,0)
    size=shape(input_img.image)[0]
    ax1.imshow(absolute(avg_img.image)[size/2,:,:])
    ax2.set_title('Input pattern')
    ax2.imshow(absolute(input_img.image*input_img.mask)[size/2,:,:], norm=LogNorm())
    ax3.set_title('PRTF')
    p=genfromtxt(os.path.join(d,'PRTF'))
    q=pix_to_q(p[:,0],1.035e-9, 0.7317, 0.000075*downsampling)
    ax3.plot(q, p[:,1], lw=1.5)
    ax3.set_xlabel(r'$|q|[nm^{-1}]$', fontsize=18)
    ax3.set_ylabel(r'$PRTF$', fontsize=18)
    ax3.axhline(1/e, c='k')
    ax4.set_title('Phase')
    ax4.imshow(numpy.angle(avg_img.image)[size/2,:,:],cmap='PiYG')
    ax5.set_title('Average Fourier Space')
    ax5.imshow(absolute(avg_f.image)[size/2,:,:])
    ax6.set_title('Errors')
    errors=extract_final_errors(os.getcwd())
    errors.sort(order='fourier_error')
    ax6.plot(range(len(errors['fourier_error'])), errors['fourier_error'], lw=1.5, c='r', label='Fourier Error')
    errors.sort(order='real_error')
    ax6.plot(range(len(errors['real_error'])), errors['real_error'], lw=1.5, c='g', label='Real Error')
    pylab.legend()
    pylab.tight_layout()
    if save_file:
        pylab.savefig(image_name)

def phase_shift(fmodel):
    c=shape(fmodel.image[:])[0]/2
    fmodel.image[:]/=fmodel.image[c,c,c]/absolute(fmodel.image[:])
    return fmodel


def fourier_error(a,f,m, w=None):
    if w==None:
        w=numpy.ones(shape(a))
    efourier_nom=pow((absolute(f[where(m)])-absolute(a[where(m)])),2)*w[where(m)]
    efourier_den=sum(pow(absolute(a[where(m)]),2))+sum(pow(absolute(f[where(~m)]),2))
    return sqrt(efourier_nom.sum()/efourier_den)

def resolution_weights(dim, mask, wl=1.035e-09, ps=0.000075*4., dd=0.7317):
    r_array=stuff.r_array_3D(dim)
    q_array=pix_to_q(r_array, wl, dd, ps)
    res_array=1e09/q_array
    return res_array/res_array[where(mask.astype(bool))].max()

def radial_weights(dim, mask):
    r_array=stuff.r_array_3D(dim)
    return r_array/r_array[where(mask.astype(bool))].max()  

def radial_fourier_error(A,F,bins, weights=None, dim=None, pickle_output=False, file_name=None):
    if dim==None:
        dim=shape(A.image)[0]
    r_array=stuff.r_array_3D(dim)
    bin_range=linspace(0,dim/2,bins)
    ferr=[]
    xbin=[]
    for i in range(bins):
        try:
            index=((r_array>bin_range[i])&(r_array<bin_range[i+1]))
        except:
            index=((r_array>bin_range[i]))
        if index.sum()>0:
            if weights==None:
                ferr.append(fourier_error(A.image[where(index)], F.image[where(index)], A.mask.astype(bool)[where(index)]))
                xbin.append(bin_range[i])
            else:
                ferr.append(fourier_error(A.image[where(index)], F.image[where(index)], A.mask.astype(bool)[where(index)], weights[where(index)]))
                xbin.append(bin_range[i])
    if pickle_output:
        pickle.dump(array(ferr), open(file_name, 'wb'))
    else:
        return ferr, xbin






    
        #return array(ferr)[:,0]
        #return sqrt(sum(array(ferr)[:,0])/sum(array(ferr)[:,1]))
        #return sqrt((array(ferr)[:,0]*array(pir))/array(pir).sum())
        #return average(array(ferr)[:,0], weights=array(pir))/array(pir).max()

def resolution_weighted_fourier_error(fourier_error_array, wl=1.035e-09, ps=0.000075*4., dd=0.7317):
    fe=fourier_error_array[:,0]
    index=~numpy.isnan(fe)
    x_pix=fourier_error_array[index,1]
    q=pix_to_q(x_pix, wl, dd, ps)
    res=1e09/q
    res_norm=res/res.max()
    return average(fe[index], weights=res_norm)

def calc_average_img(r2, r1=0, r_skip=None, iteration=None):
    rundir_range=range(r1,r2)
    if r_skip!=None:
        for i in r_skip:
            rundir_range.remove(i)
    fail=0
    for r in rundir_range:
        run_dir='run_%04d'%r
        print run_dir
        if iteration==None:
            iteration=return_last_iteration_integer('{r}/output_mnt/'.format(r=run_dir))
        with h5py.File('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration)) as f:
            try:
                added_real+=f['real'][...]
                added_imag+=f['imag'][...]
            except NameError:
                added_real=f['real'][...]
                added_imag=f['imag'][...]
            except KeyError:
                print 'fail'
                fail+=1
            if r==r2-1:
                mask=f['mask'][:]
    added_real/=float(len(rundir_range)-fail)
    added_imag/=float(len(rundir_range)-fail)
    complex_fmodel=added_real+1j*added_imag
    new=spimage.sp_image_alloc(*shape(added_real))
    new.phased=1
    new.mask[:]=mask
    new.image[:]=complex_fmodel
    spimage.sp_image_write(new,'avg_fmodel_runs_{i}_{j}_iteration{k}.h5'.format(i=r1, j=r2, k=iteration),0)
    new.image[:,:,:]=fft.fftn(complex_fmodel)
    spimage.sp_image_write(new,'avg_model_runs_{i}_{j}_iteration{k}.h5'.format(i=r1, j=r2, k=iteration),0)

def calc_average_img_translated(r2, r1=0, r_skip=None, iteration=None, reference_model=None):
    rundir_range=range(r1,r2)
    if r_skip!=None:
        for i in r_skip:
            rundir_range.remove(i)
    for r in rundir_range:
        run_dir='run_%04d'%r
        print run_dir
        if iteration==None:
            iteration=return_last_iteration_integer('{r}/output_mnt/'.format(r=run_dir))
        f=spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration),0)
        if r==r1:
            if reference_model==None:
                ref=f
            else:
                ref=reference_model
            added_img=f.image[:]
            mask=f.mask[:]
        if r!=r1:
            spimage.sp_image_superimpose(ref, f, 0)
        added_img+=f.image[:]
    added_img/=float(len(rundir_range))
    new=spimage.sp_image_alloc(*shape(added_img))
    new.phased=1
    new.mask[:]=mask
    new.image[:]=added_img
    spimage.sp_image_write(new,'avg_fmodel_runs_{i}_{j}_iteration{k}.h5'.format(i=r1, j=r2, k=iteration),0)
    new.image[:,:,:]=fft.fftn(added_img)
    spimage.sp_image_write(new,'avg_model_runs_{i}_{j}_iteration{k}.h5'.format(i=r1, j=r2, k=iteration),0)
    

#Structured_errors=numpy.core.records.fromarrays(array(ferr).transpose(),names='fourier_error, radial bin', formats = 'f8, f8')

    #return Structured_errors

        #for r in errors['run']:
#    ida_img.show_img_and_phase(r+'/output_mnt/model_4009.h5', save_fig=True, output_file='img_and_phase_model4009/'+r)

def image_histogram(file_name, shift=False, mode='absolute', only_nonzero=True, cf=0., b=100):
    if mode=='absolute': f=numpy.absolute
    elif mode=='angle': f=numpy.angle
    elif mode=='real': f=numpy.real
    elif mode=='imag': f=numpy.imag

    if type(file_name)=='str':
        img=spimage.sp_image_read(file_name, 0)
    if type(file_name)!='str':
        img=file_name
    flat_img=f(img.image[:]).flatten()
    print flat_img
    if only_nonzero:
        flat_img=flat_img[flat_img!=0.]
    if cf!=0.:
        flat_img=flat_img[flat_img>cf]
    pylab.hist(flat_img, bins=b)
    return flat_img

def real_space_residual_all_iterations(reference_file=None, support_file=None, iteration=None, mode='absolute', normalize_ref_to_model=False, skip='None', model_cf=None):
    if mode=='absolute': f=numpy.absolute
    elif mode=='angle': f=numpy.angle
    elif mode=='real': f=numpy.real
    elif mode=='imag': f=numpy.imag

    if os.getcwd().startswith('/mnt'):
        models=[i+'/output_mnt/' for i in os.listdir('.') if i.startswith('run_') and not i.startswith(skip)]
    else:
        models=[i+'/output/' for i in os.listdir('.') if i.startswith('run_') and not i.startswith(skip)]
    models.sort()
    #if skip!=None:
    #    for s in skip:
    #        models.remove('run_%04d/output_mnt/'%s)
    files_dir_0=models[0]
    print files_dir_0
    if iteration==None:
        iteration=return_last_iteration_integer(files_dir_0)
        print iteration
            
    if support_file!=None:
        support_sp=spimage.sp_image_read(support_file,0)
        support_arr=real(support_sp.image[:])
    elif model_cf!=None:
        prtf_dir=[d for d in os.listdir('.') if d.startswith('prtf')]
        avg_model=f(spimage.sp_image_read(os.path.join(prtf_dir[0], 'PRTF-avg_image.h5'),0).image[:])
        support_arr=numpy.zeros_like(avg_model)
        support_arr[(avg_model/avg_model.max())>=model_cf]=1
        print 'using {} pixels'.format(sum(support_arr))
        
    else:
        print models[0]
        support_sp=spimage.sp_image_read(files_dir_0 + 'support_%04d.h5'%iteration,0)
        support_arr=real(support_sp.image[:])
        
    #spimage.sp_image_free(support_sp)
    if reference_file!=None:
        reference_sp=spimage.sp_image_read(reference_file,0)
        reference=f(reference_sp.image[:])
        #spimage.sp_image_free(reference_sp)
    else:
        reference=support_arr.copy()
    rscs=[]   
    for m in models:
        #embed()
        try:
            model_sp=spimage.sp_image_read(m+'model_%04d.h5'%iteration,0)
            model=f(model_sp.image[:])
            print m
            #if normalize_ref_to_model:
            #    reference*=mean(model)
            #    rscs.append(stuff.real_space_residual(reference, model, support, normalize=False))
            #else:
            #if model_cf!=None:
            #    support_arr=numpy.zeros_like(model)
            #    support_arr[(model/model.max())>=model_cf]=1
            #    print 'using {} pixels'.format(sum(support_arr))
            rscs.append(stuff.real_space_residual(reference, model, support=support_arr, normalize=True))
            spimage.sp_image_free(model_sp)
            del model
        except AttributeError:
            print 'Error: {} does not exist'.format(m)
    return rscs

def pickle_rsr_all_iterations(skip='None', functions=['absolute', 'angle', 'real', 'imag']):
    for j, f in enumerate(functions):
        rsr=real_space_residual_all_iterations(mode=f, skip=skip)
        try:
            all_rsr=column_stack((all_rsr, rsr))
        except:
            all_rsr=rsr
        print shape(all_rsr)
    if len(functions)>1:
        Structured_rsr=numpy.core.records.fromarrays(array(all_rsr).transpose(),
                                                     names=', '.join(functions),
                                                     formats = ', '.join(['f8',]*len(functions)))
    else:
        print array(rsr).dtype
        Structured_rsr=numpy.core.records.array(array(rsr), dtype=[(functions[0], 'float32')])
    pickle.dump(Structured_rsr, open('rsr_vs_runnumber.p', 'wb'))

def pickle_rsr_with_cf_all_iterations(skip='None', cf=0.1):
    rsr=real_space_residual_all_iterations(mode='absolute', skip=skip, model_cf=cf)
    Structured_rsr=numpy.core.records.array(array(rsr), dtype=[('absolute', 'float32')])
    pickle.dump(Structured_rsr, open('rsr_vs_runnumber_cf{}.p'.format(cf), 'wb'))
    
def real_space_residual_prtf_avg_image(reference_file=None, support_file=None, prtf_dir=None, d='.', mode='absolute'):
    if mode=='absolute': f=numpy.absolute
    elif mode=='angle': f=numpy.angle
    elif mode=='real': f=numpy.real
    elif mode=='imag': f=numpy.imag

    iteration=return_last_iteration_integer(os.path.join(d, 'run_0000/output_mnt'))
    if prtf_dir!=None:
        model=f(spimage.sp_image_read(os.path.join(prtf_dir, 'PRTF-avg_image.h5'),0).image[:])
    else:
        prtf_dirs=[os.path.join(d,i) for i in os.listdir(d) if i.startswith('prtf_')]
        print prtf_dirs[0]
        model=f(spimage.sp_image_read(os.path.join(prtf_dirs[0], 'PRTF-avg_image.h5'),0).image[:])

    if support_file!=None:
        support=real(spimage.sp_image_read(support_file,0).image[:])
    else:
        support=real(spimage.sp_image_read(os.path.join(d,'run_0000/output_mnt/support_%04d.h5'%iteration),0).image[:])
    if reference_file!=None: 
        reference=f(spimage.sp_image_read(reference_file,0).image[:])
    else:
        reference=support.copy()
    return stuff.real_space_residual(reference, model, support, normalize=True)

def find_unique(list_of_strings, split_char='_'):
    identifiers=[]
    for s in list_of_strings:
        unique_id=[]
        keys=s.split(split_char)
        for k in keys:
            o=0
            for s in list_of_strings:
                if k in s:
                    o=o+1
            if o<len(list_of_strings):
                unique_id.append(k)
        identifiers.append(split_char.join(unique_id))
    return identifiers

def plot_rsr(keys, mode, sort=False):
    dirs=[d for d in os.listdir('.') if keys in d]
    dirs.sort()
    labels=find_unique(dirs)
    cmap = matplotlib.cm.get_cmap('gist_rainbow')
    clr_index = linspace(0.,1.,len(dirs))
    for i,d in enumerate(dirs):
        result=pickle.load(open(os.path.join(d, 'rsr_vs_runnumber.p')))
        #embed()
        rsr=result[mode]
        if sort:
            rsr.sort()
        pylab.plot(rsr, label=labels[i], lw=1.5, color=cmap(clr_index[i]))
    pylab.legend(loc='upper left')
