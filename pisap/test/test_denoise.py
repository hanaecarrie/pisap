##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import os.path as osp
import numpy as np

# Package import
from pisap.numerics.noise import soft_thresholding, hard_thresholding

class TestDenoise(unittest.TestCase):
    """ Test the soft/hard threshold function.
    """
    def setUp(self):
        """ Create synthetic data.
        """
        self.shape = (10, 10)
        self.level = 0.5
        self.ref = np.ones(self.shape)
        self.imgs = [
            2.0 * self.level * self.ref,  # after the level
            1.0 * self.level * self.ref,  # on the level
            0.5 * self.level * self.ref   # before the level
        ]
        self.ref_cplx = (np.sqrt(2) / 2 + 1.j *
                         np.sqrt(2) / 2 * np.ones(self.shape))
        self.imgs_cplx = [
            2.0 * self.level * self.ref_cplx,  # modulus after the level
            1.0 * self.level * self.ref_cplx,  # modulus on the level
            0.5 * self.level * self.ref_cplx   # modulus before the level
        ]

    def test_soft_thresholding(self):
        """ Test soft thresholding.
        """
        # real case
        # after the level
        res = soft_thresholding(self.imgs[0], self.level)
        np.testing.assert_allclose(res, self.imgs[0] - self.level * self.ref)
        # on the level
        res = soft_thresholding(self.imgs[1], self.level)
        np.testing.assert_allclose(res, self.imgs[1] - self.level * self.ref)
        # before the level
        res = soft_thresholding(self.imgs[2], self.level)
        np.testing.assert_allclose(res, np.zeros(self.shape))
        # complex case
        # after the level
        res = soft_thresholding(self.imgs_cplx[0], self.level)
        np.testing.assert_allclose(
            res, self.imgs_cplx[0] - self.level * self.ref_cplx)
        # on the level
        res = soft_thresholding(self.imgs_cplx[1], self.level)
        np.testing.assert_allclose(
            res, self.imgs_cplx[1] - self.level * self.ref_cplx)
        # before the level
        res = soft_thresholding(self.imgs_cplx[2], self.level)
        np.testing.assert_allclose(res, np.zeros(self.shape))

    def test_hard_thresholding(self):
        """ Test hard thresholding.
        """
        # complex and real case
        # after the level
        res = hard_thresholding(self.imgs[0], self.level)
        np.testing.assert_allclose(res, self.imgs[0])
        # on the level
        res = hard_thresholding(self.imgs[1], self.level)
        np.testing.assert_allclose(res, self.imgs[1])


if __name__ == "__main__":
    unittest.main()
