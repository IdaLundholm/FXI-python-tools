#!/usr/bin/env python

import numpy as np
import scipy.stats
from scipy import ndimage
import spimage
import os
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import sys
from eke import image_manipulation
import cPickle as pickle
import time
import math
import FXI_python_tools.stuff as stuff
import h5py
from IPython import embed

reload(stuff)

def prep_emc_output_for_phasing(input_file, output_file, corner_to_zero=False):
    "Sets all negative values in image to 0 and create a rectangular mask"
    img=spimage.sp_image_read(input_file,0)
    #set all negative values to 0:
    img.image[:,:,:]=img.image.clip(0)
    #set corners of mask to 1:
    r_array = stuff.r_array_3D(img.image.shape[0])
    img.mask[r_array>img.image.shape[0]/2. + 0.5]=1
    if corner_to_zero:
        img.image[r_array>img.image.shape[0]]=0.
    spimage.sp_image_write(img,output_file,0)

def mask_center(img_file_name, radius, output_file_name=None, save_file=True):
    """Create a new mask around the center with given radius"""
    img=spimage.sp_image_read(img_file_name,0)
    r_array = stuff.r_array_3D(img.image.shape[0])
    img.mask[r_array<radius]=0
    img.mask[r_array>radius]=1
    new_mask=spimage.sp_image_alloc(*np.shape(img.mask))
    new_mask.mask[:,:,:]=img.mask
    new_mask.image[:,:,:]=img.image
    new_mask.shifted=img.shifted
    new_mask.scaled=img.scaled
    new_mask.detector=img.detector
    if save_file:
        spimage.sp_image_write(new_mask, output_file_name, 0)
    else:
        return(new_mask)
    
def plot_error_per_iteration(dir='.', with_legend=True):
    """ Plots the errors saved in efourier.data and ereal.data files in given directory """
    efourier=np.genfromtxt(dir+'/efourier.data')
    ereal=np.genfromtxt(dir+'/ereal.data')
    fig=plt.figure('Phasing errors')
    plt.plot(range(len(efourier)), efourier, lw=2., c='r', label='Fourier error')
    plt.plot(range(len(ereal)), ereal, lw=2., c='k', label='Real error')
    if with_legend:
        plt.legend()
    plt.show()

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
                errors.append([np.genfromtxt(d+'/efourier.data')[-1], np.genfromtxt(d+'/ereal.data')[-1], d])
            except:
                print d + ' is not a directory or does not contain error files'
        Structured_errors=np.core.records.fromarrays(np.array(errors).transpose(),names='fourier_error, real_error, run', formats = 'f8, f8, S8')
        if pickle_output:
            pickle.dump(Structured_errors, open(path+'/final_errors.p', 'wb'))
    return(Structured_errors)
    
def plot_final_error_hist(path, b=10):
    errors=extract_final_errors(path)
    fig=plt.figure('Final Errors Histogram')
    fig.clear()
    fig.add_subplot(2,1,1)
    plt.hist(errors['fourier_error'], bins=b, label='Fourier error')
    plt.legend()
    fig.add_subplot(2,1,2)
    plt.hist(errors['real_error'], bins=b, label='Real error')
    plt.legend()
    
def plot_final_errors(path, sort=True):
    errors=extract_final_errors(path)
    fig=plt.figure('Errors')
    fig.add_subplot(2,1,1)
    if sort:
        errors.sort(order='fourier_error')
    plt.title('Fourier error')
    plt.plot(range(len(errors['fourier_error'])), errors['fourier_error'])
    fig.add_subplot(2,1,2)
    if sort:
        errors.sort(order='real_error')
    plt.title('Real error')
    plt.plot(range(len(errors['real_error'])), errors['real_error'])
    plt.savefig(os.path.join(path, 'error_plot.png'))

def select_by_error_cutoff(cutoff):
    try:
        errors=pickle.load(open(path+'/final_errors.p', 'rb'))
    except:
        errors=extract_final_errors()
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
    new=spimage.sp_image_alloc(*np.shape(img.image))
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
    return wl / 4. / np.sin( np.arctan2( x * ps, dd ) / 2. )

def pix_to_q_2(x,wl, dd, ps):
    return (2*ps)/(dd*wl)

def radial_average_q(file_name, downsampling):
    img=spimage.sp_image_read(file_name,0)
    s=np.shape(img.image)
    r,radavg=spimage.radialMeanImage(img.image, cx=0., cy=0., cz=0., output_r=True)
    q=pix_to_q(r,1.035e-9, 0.7317, 0.000075*downsampling)
    q/=1e09 #reciprocal nanometres
    q_edge=pix_to_q(s[0]/2,1.035e-9, 0.7317, 0.000075*downsampling)/1e09
    q_short=q[q<=q_edge] #Truncate prtf at detector edge
    return radavg[:len(q_short)], q_short

def prtf_radial_average(prtf_dir, downsampling):
    prtf=spimage.sp_image_read(os.path.join(prtf_dir,'PRTF-prtf.h5'),0)
    s=np.shape(prtf.image)
    r,prtf_radavg=spimage.radialMeanImage(prtf.image, cx=0., cy=0., cz=0., output_r=True)
    q=pix_to_q(r,1.035e-9, 0.7317, 0.000075*downsampling)
    q/=1e09 #reciprocal nanometres
    q_edge=pix_to_q(s[0]/2,1.035e-9, 0.7317, 0.000075*downsampling)/1e09
    q_short=q[q<=q_edge] #Truncate prtf at detector edge
    return prtf_radavg[:len(q_short)], q_short

def plot_prtf(di=None,legend=True, custom_legend=None, downsampling=8., plot_beyond_edge=False, save_file=True, output_file_name=None):
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
        prtf_radavg, q = prtf_radial_average(d,downsampling)
        fig=plt.figure('PRTF')
        ax1=fig.add_subplot(111)
        ax2=ax1.twiny()
        if custom_legend!=None:
            l=custom_legend[i]
        else:
            l=d
        ax1.plot(q, prtf_radavg, '.-',lw=1.5, label=l)
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
    ax1.axhline(1/math.e, c='k')
    if save_file:
        if output_file_name!=None:
            plt.savefig(output_file_name)
        else:
            plt.savefig('prtf.png')
    


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
    s=np.shape(model.image)[0]/2
    #matplotlib.rcParams.update({'font.size': 16})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))
    ax1.set_title('Absolute')
    ax1.imshow(np.absolute(model.image[s,:,:]))
    ax2.set_title('Phase')
    ax2.imshow(np.angle(model.image[s,:,:]),cmap='PiYG')
    ax3.set_title('Real part')
    ax3.imshow(np.real(model.image[s,:,:]), cmap='coolwarm')
    ax4.set_title('Fourier')
    ax4.imshow(np.absolute(fmodel.image[s,:,:]), norm=LogNorm())
    f.subplots_adjust(wspace=1,hspace=1)
    plt.tight_layout()
    if save_img:
        plt.savefig(os.path.join(output_folder,run_dir+'.png'))
    spimage.sp_image_free(model)
    spimage.sp_image_free(fmodel)
    plt.close(f)

def save_pngs_all(i=None, output_folder='pngs', start_from=0):
    try:
        os.mkdir(output_folder)
    except:
        None
    rundirs=[d for d in os.listdir('.') if d.startswith('run_')]
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
    s=np.shape(model.image)[0]/2
    matplotlib.rcParams.update({'font.size': 16})
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,15))
    ax1.set_title('Model')
    ax1.imshow(np.absolute(model.image[s,:,:]))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(log10(np.absolute(fmodel.image[s,:,:])))
    ax3.set_title('Support')
    ax3.imshow(np.absolute(support.image[s,:,:]))
    ax4.set_title('Support x Model')
    ax4.imshow(np.absolute(support.image[s,:,:])*absolute(model.image[s,:,:]))
    if save_imgs:
        plt.savefig(os.path.join(output_folder,run_dir+'.png'))
        plt.close(f)

def show_support_fmodel_model_slice_2D(run_dir, iteration=None, save_imgs=False, output_folder='pngs'):
    if iteration==None:
        iteration=return_last_iteration_integer(run_dir+'/output_mnt/')
    print os.getcwd(), iteration
    model=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/model_{i}.h5'.format(r=run_dir, i=iteration),0))
    fmodel=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/fmodel_{i}.h5'.format(r=run_dir, i=iteration),0))
    support=spimage.sp_image_shift(spimage.sp_image_read('{r}/output_mnt/support_{i}.h5'.format(r=run_dir, i=iteration),0))
    s=np.shape(model.image)[0]/2
    matplotlib.rcParams.update({'font.size': 16})
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,10))
    ax1.set_title('Model')
    ax1.imshow(np.absolute(model.image))
    ax2.set_title('Fourier amplitude')
    ax2.imshow(log10(np.absolute(fmodel.image)))
    ax3.set_title('Support')
    ax3.imshow(np.absolute(support.image))
    plt.tight_layout() 
    if save_imgs:
        plt.savefig(os.path.join(output_folder,run_dir+'.png'))
        plt.close(f)
        
def save_pngs_all_support(output_folder='pngs', start_from=0):
    try:
        os.mkdir(output_folder)
    except:
        None
    rundirs=[d for d in os.listdir('.') if d.startswith('run_')]
    rundirs.sort()
    for r in rundirs[start_from:]:
        show_support_fmodel_model_slice(r, save_imgs=True, output_folder=output_folder)

def plot_prtf_results_2D(d, input_image, shift_input=False, save_file=True, image_name = 'PRTF_results.png'):
    f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30,20))
    ax1.set_title('Average Real space')
    avg_img=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_image.h5'), 0))
    avg_f=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_fft.h5'), 0))
    if shift_input:
        input_img=spimage.sp_image_shift(spimage.sp_image_read(input_image,0))
    else:
        input_img=spimage.sp_image_read(input_image,0)
    ax1.imshow(np.absolute(avg_img.image))
    ax2.set_title('Input pattern')
    ax2.imshow(np.absolute(input_img.image*input_img.mask), norm=LogNorm())
    ax3.set_title('PRTF')
    prtf_radavg, q = prtf_radial_average(d,downsampling)
    ax3.plot(pix_to_q(p[:,0],1.035e-9, 0.7317, 0.0006), p[:,1], lw=1.5)
    ax3.set_xlabel(r'$|q|[nm^{-1}]$', fontsize=18)
    ax3.set_ylabel(r'$PRTF$', fontsize=18)
    ax3.axhline(1/math.e, c='k')
    ax4.set_title('Phase')
    ax4.imshow(np.angle(avg_img.image),cmap='PiYG')
    ax5.set_title('Average Fourier Space')
    ax5.imshow(np.absolute(avg_f.image))
    ax6.set_title('Errors')
    errors=extract_final_errors(os.getcwd())
    errors.sort(order='fourier_error')
    ax6.plot(range(len(errors['fourier_error'])), errors['fourier_error'], lw=1.5, c='r', label='Fourier Error')
    errors.sort(order='real_error')
    ax6.plot(range(len(errors['real_error'])), errors['real_error'], lw=1.5, c='g', label='Real Error')
    plt.legend()
    plt.tight_layout()
    if save_file:
        plt.savefig(image_name)
        
def plot_prtf_results_3D(d, input_image, downsampling, save_file=True, image_name = 'PRTF_results.png', zoom=True):
    f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,13))
    ax1.set_title('Average Real space')
    avg_img=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_image.h5'), 0))
    avg_f=spimage.sp_image_shift(spimage.sp_image_read(os.path.join(d,'PRTF-avg_fft.h5'), 0))
    input_img=spimage.sp_image_read(input_image,0)
    size=np.shape(input_img.image)[0]
    sl=(size/2, slice(None), slice(None))
    if zoom:
        z=int((size-np.count_nonzero(avg_img.image[size/2, size/2, :]))/2.2)
        sl_real=(size/2, slice(z, size-z), slice(z,size-z))
    elif zoom==False:
        sl_real=sl
    print sl_real
    ax1.imshow(np.absolute(avg_img.image)[sl_real])
    ax2.set_title('Input pattern')
    ax2.imshow(np.absolute(input_img.image*input_img.mask)[sl], norm=LogNorm())
    ax3.set_title('PRTF')
    prtf_radavg, q = prtf_radial_average(d,downsampling)
    ax3.plot(q, prtf_radavg, lw=1.5)
    ax3.set_xlabel(r'$|q|[nm^{-1}]$', fontsize=18)
    ax3.set_ylabel(r'$PRTF$', fontsize=18)
    ax3.axhline(1/math.e, c='k')
    ax4.set_title('Phase')
    ax4.imshow(np.angle(avg_img.image)[sl_real],cmap='PiYG')
    ax5.set_title('Average Fourier Space')
    ax5.imshow(np.absolute(avg_f.image)[sl])
    ax6.set_title('Errors')
    errors=extract_final_errors(os.getcwd())
    errors.sort(order='fourier_error')
    ax6.plot(range(len(errors['fourier_error'])), errors['fourier_error'], lw=1.5, label='Fourier Error')
    errors.sort(order='real_error')
    ax6.plot(range(len(errors['real_error'])), errors['real_error'], lw=1.5, label='Real Error')
    plt.legend()
    plt.tight_layout()
    if save_file:
        plt.savefig(image_name)

def phase_shift(fmodel):
    c=np.shape(fmodel.image[:])[0]/2
    fmodel.image[:]/=fmodel.image[c,c,c]/np.absolute(fmodel.image[:])
    return fmodel

def fourier_error(a,f,m, w=None):
    if w==None:
        w=np.ones(np.shape(a))
    efourier_nom=np.pow((np.absolute(f[np.where(m)])-np.absolute(a[np.where(m)])),2)*w[np.where(m)]
    efourier_den=np.sum(np.pow(np.absolute(a[np.where(m)]),2))+np.sum(np.pow(np.absolute(f[np.where(~m)]),2))
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
        dim=np.shape(A.image)[0]
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

def resolution_weighted_fourier_error(fourier_error_array, wl=1.035e-09, ps=0.000075*4., dd=0.7317):
    fe=fourier_error_array[:,0]
    index=~np.isnan(fe)
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
    new=spimage.sp_image_alloc(*np.shape(added_real))
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
    new=spimage.sp_image_alloc(*np.shape(added_img))
    new.phased=1
    new.mask[:]=mask
    new.image[:]=added_img
    spimage.sp_image_write(new,'avg_fmodel_runs_{i}_{j}_iteration{k}.h5'.format(i=r1, j=r2, k=iteration),0)
    new.image[:,:,:]=fft.fftn(added_img)
    spimage.sp_image_write(new,'avg_model_runs_{i}_{j}_iteration{k}.h5'.format(i=r1, j=r2, k=iteration),0)
    
def image_histogram(file_name, shift=False, mode='absolute', only_nonzero=True, cf=0., radius_cutoff=None, b=100):
    if mode=='absolute': f=np.absolute
    elif mode=='angle': f=np.angle
    elif mode=='real': f=np.real
    elif mode=='imag': f=np.imag
    print type(file_name)
    if type(file_name)==str:
        img=spimage.sp_image_read(file_name, 0)
    elif type(file_name)!=str:
        img=file_name
    if shift:
        print 'shifting image'
        img=spimage.sp_image_shift(img)
    if radius_cutoff!=None:
        r=stuff.r_array_3D(np.shape(img.image)[0])
        new_img=img.image[np.where(r<radius_cutoff)]
        flat_img=f(new_img).flatten()
    else:
        flat_img=f(img.image[:]).flatten()
    print flat_img
    if only_nonzero:
        flat_img=flat_img[flat_img!=0.]
    if cf!=0.:
        flat_img=flat_img[flat_img>cf]        
    plt.hist(flat_img, bins=b)
    return flat_img

def real_space_residual_all_iterations(reference_file=None, support_file=None, iteration=None, mode='absolute', normalize_ref_to_model=False, skip='None', model_cf=None):
    if mode=='absolute': f=np.absolute
    elif mode=='angle': f=np.angle
    elif mode=='real': f=np.real
    elif mode=='imag': f=np.imag

    if os.getcwd().startswith('/mnt'):
        models=[i+'/output_mnt/' for i in os.listdir('.') if i.startswith('run_') and not i.startswith(skip)]
    else:
        models=[i+'/output/' for i in os.listdir('.') if i.startswith('run_') and not i.startswith(skip)]
    models.sort()
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
        support_arr=np.zeros_like(avg_model)
        support_arr[(avg_model/avg_model.max())>=model_cf]=1
        print 'using {} pixels'.format(np.sum(support_arr))
        
    else:
        print models[0]
        support_sp=spimage.sp_image_read(files_dir_0 + 'support_%04d.h5'%iteration,0)
        support_arr=real(support_sp.image[:])
        
    if reference_file!=None:
        reference_sp=spimage.sp_image_read(reference_file,0)
        reference=f(reference_sp.image[:])
        
    else:
        reference=support_arr.copy()
    rscs=[]   
    for m in models:
        try:
            model_sp=spimage.sp_image_read(m+'model_%04d.h5'%iteration,0)
            model=f(model_sp.image[:])
            print m
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
            all_rsr=np.column_stack((all_rsr, rsr))
        except:
            all_rsr=rsr
        print np.shape(all_rsr)
    if len(functions)>1:
        Structured_rsr=np.core.records.fromarrays(np.array(all_rsr).transpose(),
                                                     names=', '.join(functions),
                                                     formats = ', '.join(['f8',]*len(functions)))
    else:
        print np.array(rsr).dtype
        Structured_rsr=np.core.records.array(np.array(rsr), dtype=[(functions[0], 'float32')])
    pickle.dump(Structured_rsr, open('rsr_vs_runnumber.p', 'wb'))

def pickle_rsr_with_cf_all_iterations(skip='None', cf=0.1):
    rsr=real_space_residual_all_iterations(mode='absolute', skip=skip, model_cf=cf)
    Structured_rsr=np.core.records.array(np.array(rsr), dtype=[('absolute', 'float32')])
    pickle.dump(Structured_rsr, open('rsr_vs_runnumber_cf{}.p'.format(cf), 'wb'))
    
def real_space_residual_prtf_avg_image(reference_file=None, support_file=None, prtf_dir=None, d='.', mode='absolute'):
    if mode=='absolute': f=np.absolute
    elif mode=='angle': f=np.angle
    elif mode=='real': f=np.real
    elif mode=='imag': f=np.imag

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
        rsr=result[mode]
        if sort:
            rsr.sort()
        plt.plot(rsr, label=labels[i], lw=1.5, color=cmap(clr_index[i]))
    plt.legend(loc='upper left')

    
def radial_sort(img_file, shift=True):
    img_array=np.absolute(spimage.sp_image_read(img_file,0).image)
    if shift:
        img_array=np.fft.fftshift(img_array)
    #r=stuff.r_array_3D(np.shape(img_array)[0], ndimage.measurements.center_of_mass(img_array))
    r=stuff.r_array_3D(np.shape(img_array)[0])
    r_flat=r.flatten()
    img_flat=img_array.flatten()
    sort_order=np.argsort(r_flat)
    return r_flat[sort_order], img_flat[sort_order]

def image_to_list(file_name, cutoff=0.):
    img=np.absolute(spimage.sp_image_read(file_name,0).image).flatten()
    return img[np.where(img>np.max(img)*cutoff)]

def round_by_prec(x, prec):
    return np.round(x*prec)/prec

def average_by_diff(r):
    diff=r[1:]-r[:-1]
    a=np.argwhere(diff>0.1).max()
    new_rounded_r=round_by_prec(r[:a], 5)
    r[:a]=new_rounded_r
    return r

def average_where_r(radi, i):
    new_i_avg=[]
    new_i_std=[]
    for r in np.unique(radi):
        new_i_avg.append(np.average(i[np.where(radi==r)]))
        new_i_std.append(np.std(i[np.where(radi==r)]))
    return np.unique(radi), new_i_avg, new_i_std

def dot_strength(img_file, cutoff_r, c=2.5, mode='median', strongest=None, plot=False, plot_lines=False, fig_name=None, return_sorted_voxels=False):
    if mode=='median': f=np.median
    elif mode=='average': f=np.average
    r,voxels=radial_sort(img_file, cutoff_r)
    if strongest==None:
        dot_border=f(voxels[np.where(r<cutoff_r)])+c*np.std(voxels[np.where(r<cutoff_r)])
        dot=voxels[np.where((voxels>dot_border)&(r<cutoff_r*0.8))]
    else:
        strongest_voxels=voxels[r<cutoff_r*0.8]
        strongest_voxels.sort()
        dot=strongest_voxels[-strongest:]
        #dot=voxels[:strongest]
        dot_border=dot.min()
        voxels_no_dot=voxels[((voxels<dot.min())&(r<cutoff_r))]
    print dot, f(voxels_no_dot)
    print 'dot contains %s voxels'%str(len(dot))
    dot_strength=f(dot)/f(voxels_no_dot)
    print 'The dot is %s times stronger than the %s'%(str(dot_strength),mode)
    if return_sorted_voxels:
        return r, voxels
    if plot:
        if fig_name==None:
            plt.figure(img_file)
        else:
            plt.figure(fig_name)
        new_r=r[r<cutoff_r]
        new_r=new_r.round(1)
        new_r=average_by_diff(new_r)
        ravg, iavg, istd = average_where_r(new_r.round(1),voxels[r<cutoff_r])
        plt.errorbar(ravg, iavg, yerr=istd, fmt='o')
        if plot_lines:
            plt.axhline(f(voxels_no_dot), c='k')
            plt.axhline(dot_border, c='r')
            plt.axvline(cutoff_r*0.8, c='r')
        

        
        
