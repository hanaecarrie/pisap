"""  images  """

# imports
import numpy as np
import scipy.io
import os
from nilearn.masking import compute_epi_mask
from nilearn.image import load_img
import matplotlib.pyplot as plt
from pisap.base.utils import extract_paches_from_2d_images
from pisap.base.utils import min_max_normalize    

import scipy.fftpack as pfft


class CamcanImages_complex(object):
    """This class defines the input images of the CamCan database"""
    
    def __init__(self, data_path_begin, patch_shape, size_ratio_sets,num_slice):
        """Initialize the CamcanImages class
        -----------
        Inputs:
            data_path_begin -- string, path to get the database
            patch_shape -- tuple of int (wp,hp), wight and height of the patches
                WARNING: WORKING ONLY FOR SQUARE PATCHES wp=hp, a bug in
                pisap.numerics.linear, in the DictionaryLearningWavelet.op
                and .adj_op will appear otherwise
            size_ratio_sets -- tuple of floats between 0 and 1, 
                ratio_train,ratio_test) with ratio_train= ratio of the
                training set, ratio_test= ratio of the testing set
                and with ratio_train + ratio_test = 1
            num_slice -- int, the slice number to select in the 3D brain
        -----------
        Outputs:
            nb_imgs -- int, length of the whole database
            size_training -- int, ratio_train*nb_imgs, size of the training set
            size_testing -- int, ratio_test*nb_imgs, size of the testing set
            imgs -- initialized as an empty list, list of the 2D images
                (np.ndarray) from the database after preprocessing
            imgs_flat -- initialized as an empty list, contains the list of the
                corresponding 1D flattened images (np.array) after preprocessing
            imgs_patches_flat -- initialized as an empty list, contains the 
                list of the corresponding 1D flattened patched images (np.array)
                after preprocessing
            masks --  initialized as an empty list, contains the list of the
                corresponding 2D masks (np.ndarray) after preprocessing
            training_set -- initialized as an empty list, contains the 
                list of the 2D images of the training set after preprcoessing
            testing_set_ref -- initialized as an empty list, contains the 
                list of the 2D images of the testing set after preprcoessing
            img_shape=(w,h) -- tuple of int, image shape
            w -- int, weight of the images
            h -- int, hight of the images
            
        """
        self.data_path_begin=data_path_begin
        self.patch_shape=patch_shape
        self.size_ratio_sets=size_ratio_sets
        self.num_slice=num_slice

        list_folders=os.listdir(self.data_path_begin)
        sub_0=list_folders[0]
        data_path_0=self.data_path_begin+sub_0+"/anat/"
        data_filename_0 = os.path.join(data_path_0, "w"+sub_0+"_T1w.nii.gz")
        img_0=load_img(data_filename_0).get_data()
        img_0=img_0[self.num_slice][:][:]

        self.nb_imgs=len(list_folders)
        self.size_training=int(size_ratio_sets[0]*self.nb_imgs)
        self.size_testing=int(size_ratio_sets[1]*self.nb_imgs)
        self.imgs=[] 
        self.imgs_patches_flat_train_real=[] #initialization
        self.imgs_patches_flat_train_imag=[] #initialization
        self.imgs_patches_flat_test=[]
        self.imgs_flat=[] #initialization
        self.masks=[] #initialization
        self.training_set=[] #initialization
        self.testing_set_ref=[] #initialization
        self.img_shape=img_0.shape
        self.w=self.img_shape[0]
        self.h=self.img_shape[1]
        print self.__dict__


    def preprocessing_complex(self, sampling_scheme_path):
        """Preprocess the data with sklearn.extract_patches_2d to compute
        the patches and with nilearn to compute the masks
        ----------
        Inputs:self
        ----------
        Outputs: # self with several updated attributes, rq: size_set stands
                   for size_training, size_validation or size_testing depending
                   of the input
            imgs -- list len(size_set) of 2darray dim (w,h) of floats,
                list of the 2D brain slices, min max normalized
            imgs_patches_flat -- list len(size_set) of 2darray dim(nb_imgs,7*7)
                of floats, list of images represented with 2D patches which
                are here flattened
            imgs_flat -- list len(size_set) of 1darray (w*h) of floats,
                list of the imgs flattened masks= list len(size_set) of 2darray
                dim (w,h) of floats, list of maks
        """
        l=self.img_shape[0]*self.img_shape[1]
        print 'preprocessing starting...'
        for i in range(self.nb_imgs):
            #getting the path
            sub_i=os.listdir(self.data_path_begin)[i]
            data_path_i=self.data_path_begin+sub_i+"/anat/"
            data_filename_i = os.path.join(data_path_i,"w"+sub_i+"_T1w.nii.gz")
            #loading the desired slice
            img_i=load_img(data_filename_i).get_data()
            img_i=img_i[self.num_slice][:][:]
            #normalizing the image slice
            sampling_scheme = self.get_sampling_scheme(sampling_scheme_path)
            if i < self.size_training: # taking the zero order solution of undersampled references
                img_i=min_max_normalize(img_i)
                kspaceCS_i=np.fft.fft2(img_i)*pfft.fftshift(sampling_scheme)
                kspaceCS_i=pfft.fftshift(kspaceCS_i)
                img_i=pfft.ifft2(kspaceCS_i)
                plt.figure()
                plt.imshow(np.abs(img_i), cmap='gray')
                plt.colorbar()
                plt.show()
            else:
                img_i=min_max_normalize(img_i)
            #computing the corresponding mask
            mask_i=compute_epi_mask(data_filename_i)
            mask_i=os.path.join(mask_i)
            mask_i=load_img(mask_i).get_data()
            mask_i=mask_i[self.num_slice][:][:]
            self.masks.append(mask_i)
            self.imgs.append(img_i)
            #flattening the image slice
            img_flat_i=img_i.flatten()
            self.imgs_flat.append(img_flat_i)
            if i < self.size_training: 
               #extracting real patches and flattening them
               patches_real = extract_patches_from_2d_images(np.real(img_i), self.patch_shape)
               self.imgs_patches_flat_train_real.append(patches_real)
               #extracting imaginary patches and flattening them
               patches_imag = extract_patches_from_2d_images(np.imag(img_i), self.patch_shape)
               self.imgs_patches_flat_train_imag.append(patches_imag)
            else: 
               #extracting patches and flattening them
               patches = extract_patches_from_2d_images(img_i, self.patch_shape)
               self.imgs_patches_flat_test.append(patches)
            #progress bar
            a=i+1
            if a==1:
                print('1/'+str(self.nb_imgs))
            if a %10==0:
                print(str(a)+'/'+str(self.nb_imgs))
        print 'preprocessing ended!'
        self.training_set=self.imgs[:self.size_training]
        self.testing_set_ref=self.imgs[self.size_training:self.nb_imgs]
        return self.__dict__
        
        
    def crop_sampling_scheme(self, sampling_scheme):
        """Crop the sampling scheme from a input sampling scheme
        with an even wigth and hight
        ---------
        Inputs:
        sampling_scheme -- np.ndarray of {0,1}, 2d matrix, the sampling scheme
        ---------
        Outputs:
        sampling_scheme -- np.ndarray of {0,1}, 2d matrix, the cropped
            sampling scheme
        """
        ss_size=sampling_scheme.shape
        sampling_scheme = sampling_scheme[
                                ss_size[0]/2-int((self.w+1)/2):ss_size[0]/2+int((self.w)/2),
                                ss_size[1]/2-int((self.h+1)/2):ss_size[1]/2+int((self.h)/2)]
        return sampling_scheme
        
        
    def get_sampling_scheme(self, sampling_scheme_path):
        """Get the sampling scheme from its path
        ---------
        Inputs:
        sampling_scheme_path -- string, path to get the sampling scheme
        ---------
        Outputs:
        sampling_scheme -- np.ndarray of {0,1}, 2d matrix, the sampling scheme 
            not shifted with low frequencies at the center
        """
        sampling_scheme = scipy.io.loadmat(sampling_scheme_path)
        sampling_scheme = sampling_scheme['sigma']
        sampling_scheme = self.crop_sampling_scheme(sampling_scheme)
        return sampling_scheme


    def retrospectiveCS(self, sampling_scheme_path): #XXX make retrospective sampling in the middle of the kspace
        """Undersample the testing set in the Fourier domain
        ---------
        Inputs:
        sampling_scheme_path -- string, path to get the sampling scheme
            WARNING: the sampling scheme should contain low frequencies
            at the center of the matrix
        ---------
        Outputs:
        kspacesCS -- list len(size_testing) of 2darray dim (w,h) of floats,
            corresponding undersampled kspace of the images
        """
        sampling_scheme = self.get_sampling_scheme(sampling_scheme_path)
        kspaceCS=[]
        for i in range(self.size_testing):
            kspaceCS_i=np.fft.fft2(self.testing_set_ref[i])*pfft.fftshift(sampling_scheme)
            kspaceCS_i=pfft.fftshift(kspaceCS_i)
            kspaceCS.append(kspaceCS_i)
        return kspaceCS

