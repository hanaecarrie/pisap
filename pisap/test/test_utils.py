import unittest
import os.path as osp
import numpy as np

import pisap
from pisap.base.utils import convert_mask_to_locations, \
                             convert_locations_to_mask, \
                             l2_normalize, min_max_normalize, \
                             trunc_to_zero


# global cst
IMG = min_max_normalize(pisap.io.load(osp.join("data", "M31_128.fits")).data)


class TestUtils(unittest.TestCase):

    def test_convert_mask(self):
        img_size = 128
        mask = np.random.randint(0, 2, size=(img_size, img_size))
        recons_mask = convert_locations_to_mask(convert_mask_to_locations(mask), img_size)
        self.assertTrue((mask == recons_mask).all())

    def test_l2_normalize(self):
        img_size = 128
        img = np.random.random(size=(img_size, img_size))
        norm_l2 = np.linalg.norm(img)
        np.testing.assert_allclose(img, norm_l2 * l2_normalize(img))

    def test_min_max_normalize(self):
        img_size = 128
        img = np.random.random(size=(img_size, img_size))
        norm_img = min_max_normalize(img)
        self.assertEqual(norm_img.min(), 0.0)
        self.assertEqual(norm_img.max(), 1.0)

    def test_trunc_to_zero(self):
        EPS = 5.0e-1
        img_size = 2
        img = (np.random.random(size=(img_size, img_size)) +
                1.j *  np.random.random(size=(img_size, img_size)))
        trunc_img = trunc_to_zero(img, eps=EPS)
        idx_i, idx_j = np.where(((np.abs(trunc_img.real) > 0.0) &
                                 (np.abs(trunc_img.real) <=EPS)))
        self.assertEqual(len(idx_i), 0)
        self.assertEqual(len(idx_j), 0)
        idx_i, idx_j = np.where(((np.abs(trunc_img.imag) > 0.0) &
                                 (np.abs(trunc_img.imag) <=EPS)))
        self.assertEqual(len(idx_i), 0)
        self.assertEqual(len(idx_j), 0)
