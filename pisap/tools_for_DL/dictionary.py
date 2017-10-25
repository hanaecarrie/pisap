
""" Dictionary """


# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning


class Dictionary(object):
    """ This class defines the dictionary to learn """

    def __init__ (self, n_components, alpha=1, n_iter=100):
         """Initialize the Dictionary class
         -----------
         Inputs:
             n_components -- int, number of atoms
             WARNING: CHOOSE A SQUARE NUMBER, IF NOT THE PLOTTING METHOD WILL BUG
             regulation term (default=1)
             n_iter -- int, number of iterations (default=100)
             alpha -- alpha (default=1)
         -----------
         Outputs:
             n_components -- n_components
             alpha -- alpha
             n_iter -- n_iter
             atoms -- np.ndarray, initialized at [1], after learning_atoms
                 contains the matrix of learnt atoms
             dico -- sklearn MiniBatchDictionaryLearning object
         """
         self.n_components = n_components
         self.n_iter = n_iter
         self.alpha = alpha
         self.atoms = np.zeros(1) #initialization
         self.dico = [] #initialization
         self.complex = False
         print self.__dict__

    def learning_atoms(self, images, option):
         """Learn the dictionary from the complex images
         -----------
         Inputs:
             images -- instance of class CamCanImages, the preprocessed
                 input images
             option -- 'real' or 'imaginary'
         -----------
         Outputs:
             self.__dict__ -- updated attributes
         """
         self.dico=MiniBatchDictionaryLearning(n_components=self.n_components,
                                               alpha=self.alpha,
                                               n_iter=self.n_iter)
         if option=='real':
             imgs = images.imgs_patches_flat_train_real
             img_size_train = len(imgs)
         if option == 'imaginary':
             imgs = images.imgs_patches_flat_train_imag
             img_size_train = len(imgs)
         buffer = []
         index = 0
         print 'learning_atoms starting...'
         for _ in range(6):
             for i in range(len(imgs)):
                 patches=imgs[i]
                 index += 1
                 buffer.append(patches)
                 if index % 100 == 0:
                     print(str(index)+'/'+str(6*img_size_train))
                     patches = np.concatenate(buffer, axis=0)
                     self.dico.fit(patches)
                     buffer = []
         print 'learning_atoms ended!'
         self.atoms=self.dico.components_
         return(self)

    def saving_atoms(self,title):
         """Save the atoms into a .npy file
         -----------
         Inputs:
             title -- string, the .npy file title
         -----------
         Outputs:
             title -- string, the .npy file title
         """
         np.save(title,self.atoms)
         return(title)

    def plotting(self, images, title='Dictionary atoms'):
         """Plot the learnt atoms
         -----------
         Inputs:
             images -- instance of class CamCanImages, the preprocessed
                 input images
             title -- string, (default='Dictionary atoms'), the .npy file title
         -----------
         Outputs:
             title -- string, the .npy file title
         """
         patch_shape=images.patch_shape
         plt.figure(figsize=(4.2, 4))
         for i, patch in enumerate(self.dico.components_):
             plt.subplot(np.int(np.sqrt(self.n_components)),
                         np.int(np.sqrt(self.n_components)), i+1)
             plt.imshow(patch.reshape(patch_shape), cmap=plt.cm.gray,
                        interpolation='nearest')
             plt.xticks(())
             plt.yticks(())
         plt.suptitle(title, fontsize=16)
         plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
         plt.show()
         return(title)
