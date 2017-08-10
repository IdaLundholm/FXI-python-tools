#!/usr/bin/env python

import numpy
from numpy import *
import pylab
import spimage
import os
from matplotlib.colors import LogNorm
import sys
from eke import image_manipulation
import cPickle as pickle
import time
import h5py


class Responsability(object):
    def __init__(self, in_file):
        with h5py.File(in_file, 'r') as f:
            self.data=f['data'][:]
            self.data_modified=self.data.copy()
        self.number_of_images=self.data.shape[0]
        self.number_of_rotations=self.data.shape[1]
        self.count_rotations_per_image()
        self.calc_maximum_occupancy_per_image()
        
    def count_rotations_per_image(self, original_data=True):
        nonzero=[]
        for i in range(self.number_of_images):
            if original_data:
                nonzero.append(count_nonzero(self.data[i,:]))
            else:
                nonzero.append(count_nonzero(self.data_modified[i,:]))
        self.rotations_per_image=nonzero

    def count_images_per_rotations(self, original_data=True):
        num=[]
        for i in range(self.number_of_rotations):
            if original_data:
                num.append(count_nonzero(self.data[:,i]))
            else:
               num.append(count_nonzero(self.data[:,i]))        
    
    def calc_maximum_occupancy_per_image(self, original_data=True):
        if original_data:
            self.maximum_occupancy=self.data.max(axis=1)
        else:
            self.maximum_occupancy=self.data_modified.max(axis=1)
            
    def exclude_and_rescale(self, occupancy_cutoff=0., top=0, squared_rescale=False):
        self.cutoff=occupancy_cutoff
        self.data_modified[self.data_modified<occupancy_cutoff]=0.
        self.excluded_patterns=[]
        for i in range(self.number_of_images):
            resp=self.data_modified[i,:]
            resp[resp.argsort()[:-top]]=0.
            if sum(self.data_modified[i,:])==0.:
                self.excluded_patterns.append(i)
            else:
                if squared_rescale:
                    resp_scaled=resp**2/sum(resp**2)
                    resp_scaled[resp_scaled<occupancy_cutoff]=0.
                else:
                    resp_scaled=resp/sum(resp)
                self.data_modified[i,:]=resp_scaled

    def plot_num_rotations_vs_max_occupancy(self):
        pylab.figure('Number of rotations vs maximum occupancy', figsize=(10,5))
        pylab.plot(self.rotations_per_image, self.maximum_occupancy, 'o')
        pylab.tight_layout()
        pylab.xlabel('Number of rotations')
        pylab.ylabel('Maximum occupancy')

    def histogram_rotations_per_image(self, save_fig=False, figure_name='Rotations_per_image.png',output=False):
        pylab.figure('Number of rotations per image', figsize=(int(round(self.number_of_images*0.04)),6))
        pylab.bar(range(self.number_of_images), self.rotations_per_image)
        pylab.tight_layout()
        if save_fig:
            pylab.savefig(figure_name)
        if output:
            return num_of_rotations
        
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

def read_emc_conf_as_dict(path):
    f=open(os.path.join(path,'emc.conf'),'r')
    lines=f.readlines()
    f.close()
    emc_dict={}
    for l in lines:
        l_split=l.split('=')
        val=l_split[1].replace(';','')
        val=val.replace('/n','')
        val=val.replace('"','')
        emc_dict[l_split[0].strip()]=val.strip()
    return emc_dict

            
def create_symlinked_output_folders(path='.'):
    run_folders=[r for r in os.listdir(path) if r.startswith('run_')]
    for r in run_folders:
        scratch_output=read_emc_conf_as_dict(r)['output_dir']
        try:
            os.unlink(r+'/output_mnt')
        except:
            None
        os.symlink('/mnt/davinci'+scratch_output, r+'/output_mnt')

def add_mask(path, number_of_files, output_file):
    """Reads in files in paths, outputs h5 file output_file with accumulative mask from input images saved in both image and mask"""
    imgs=[os.path.join(path,i) for i in os.listdir(path) if i.endswith('.h5')]
    for img_file in imgs[:number_of_files]:
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

def convert_to_png(output_folder='pngs'):
    try:
        os.mkdir(output_folder)
    except OSError:
        None
    imgs=[i for i in os.listdir('.') if i.endswith('.h5')]
    for im in imgs:
        img=spimage.sp_image_read(im,0)
        fig=pylab.figure(1)
        pylab.imshow(absolute(numpy.log10(img.image))*img.mask)
        pylab.savefig(os.path.join(output_folder, im.replace('.h5', '.png')))
        fig.clf()

def plot_responsability_2D_hist(file_name, save_img=False, output_file_name='responsability_histogram.png'):
    with h5py.File(file_name, 'r') as f:
        responsabilities=numpy.transpose(f['data'])
    max_o=[]
    unique_rots=[]
    for i in range(numpy.shape(responsabilities)[1]):
        max_o.append(max(responsabilities[:,i]))
        unique_rots.append(len(unique(responsabilities[:,i])))
    fig=pylab.figure('responsability histogram')
    pylab.hist2d(max_o, unique_rots, bins=100, norm=LogNorm())
    if save_img:
        pylab.savefig(output_file_name)


def plot_scaling_with_error(scaling, fignum=1, save_fig=False, figure_name='Scaling_Error_plot.png'):
    if type(scaling)==str:
        scaling_obj=h5py.File(scaling, 'r')
    elif type(scaling)==h5py._hl.files.File:
        scaling_obj=scaling
    else:
        print 'provide input as either file name or h5py File object'
        exit
    mean_scaling_per_image=mean(scaling_obj['data'][:], axis=0)
    std_scaling_per_image=std(scaling_obj['data'][:], axis=0)
    num=len(mean_scaling_per_image)
    pylab.figure('Scaling errorbar%d'%fignum, figsize=(int(round(num*0.04)),6))
    pylab.errorbar(range(num),mean_scaling_per_image, std_scaling_per_image, fmt='o', elinewidth=1.5, capsize=1.5)
    if save_fig:
        pylab.savefig(figure_name)

def plot_mean_scaling_per_iteration(path, number_of_iterations, fignum=1, save_fig=False, figure_name='Scaling_per_image_plot.png'):
    mean_scaling_per_image=[]
    for i in range(number_of_iterations):
        print i
        with h5py.File(os.path.join(path,'scaling_%04d.h5'%i), 'r') as scaling_obj:
            mean_scaling_per_image.append(mean(scaling_obj['data'][:], axis=0))
    pylab.figure('Scaling %d'%fignum, figsize=(int(round(num*0.04)),6))
    pylab.errorbar(range(num),mean_scaling_per_image, std_scaling_per_image, fmt='o', elinewidth=1.5, capsize=1.5)
    if save_fig:
        pylab.savefig(figure_name)
            
def histogram_rotations_per_image(resp, cutoff=0., fignum=1, save_fig=False, figure_name='Scaling_Error_plot.png', output=False):
    if type(resp)==str:
        responsabilities_obj=h5py.File(resp, 'r')
    elif type(resp)==h5py._hl.files.File:
        responsabilities_obj=resp
    else:
        print 'provide input as either file name or h5py File object'
        exit
    num=len(responsabilities_obj['data'])
    num_of_rotations=[]
    for i in range(num):
        num_of_rotations.append(count_nonzero(where(responsabilities_obj['data'][i]>cutoff)))
    pylab.figure('Number of rotations per image%d'%fignum, figsize=(int(round(num*0.04)),6))
    pylab.bar(range(num), num_of_rotations)
    if save_fig:
        pylab.savefig(figure_name)
    if output:
        return num_of_rotations


def find_images_to_exclude(responsabilities, number_of_patterns, cutoff, top):
    if type(responsabilities)==str:
        with h5py.File(responsabilities, "r") as file_handle:
            responsabilities_obj = numpy.transpose(file_handle["data"][:number_of_patterns, :])
    elif type(resp)==h5py._hl.files.File:
        responsabilities_obj=numpy.transpose(responsabilities["data"][:number_of_patterns, :])
    responsabilities_obj[responsabilities_obj<cutoff]=0.
    img_list=[]
    top_resp=[]
    for i in range(number_of_patterns):
        top_resp.append(max(responsabilities_obj[:,i]))
        if sum(responsabilities_obj[:,i])!=0.:
            img_list.append(i)
    #print "Top %d higest responsabilities: %s for imgs %s"%(top, str(sort(top_resp)[:top]), str(argsort(top_resp)[:top]))
    return img_list, top_resp

def plot_num_rot_vs_maxresp(responsabilities):
    if type(responsabilities)==str:
        with h5py.File(responsabilities, "r") as file_handle:
            responsabilities_obj = numpy.transpose(file_handle["data"][:number_of_patterns, :])
    elif type(resp)==h5py._hl.files.File:
        responsabilities_obj=numpy.transpose(responsabilities["data"][:number_of_patterns, :])
    top_resp=responsabilites_obj[:].max(axis=0)
    num_rot=count_nonzero(responsabilities_obj[:])

def count_all_occurences(l):
    u=unique(l)
    o={}
    for i in u:
        o[i] = l.count(i)
    return o

def rmsd(a1,a2,n):
    return sqrt((sum(pow((a1-a2),2)))/n)

def quad_rmsd(img_array):
    dim=img_array.shape[0]/2
    q1=img_array[:dim, :dim, :dim]
    q2=img_array[:dim, dim:, :dim]
    q3=img_array[:dim, dim:, dim:]
    q4=img_array[:dim, :dim, dim:]
    q5=img_array[dim:, :dim, :dim]
    q6=img_array[dim:, dim:, :dim]
    q7=img_array[dim:, dim:, dim:]
    q8=img_array[dim:, :dim, dim:]
    n=dim**3
    return [rmsd(q1,q7[::-1,::-1,::-1],n), rmsd(q2,q8[::-1,::-1,::-1],n),rmsd(q3,q5[::-1,::-1,::-1],n),rmsd(q4,q6[::-1,::-1,::-1],n)]

def centro_sym(img_array):
    dim=img_array.shape[0]/2
    q1=img_array[:dim, :dim, :dim]
    q2=img_array[:dim, dim:, :dim]
    q3=img_array[:dim, dim:, dim:]
    q4=img_array[:dim, :dim, dim:]
    img_array[dim:, :dim, :dim]=q3[::-1,::-1,::-1]
    img_array[dim:, dim:, :dim]=q4[::-1,::-1,::-1]
    img_array[dim:, dim:, dim:]=q1[::-1,::-1,::-1]
    img_array[dim:, :dim, dim:]=q2[::-1,::-1,::-1]
    return img_array
    
