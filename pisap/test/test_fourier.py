import unittest
import os.path as osp
import numpy as np

import pisap
from pisap.numerics.fourier import NFFT, FFT
from pisap.base.utils import convert_mask_to_locations, min_max_normalize, \
                             trunc_to_zero


# global cst
IMG = min_max_normalize(pisap.io.load(osp.join("data", "M31_128.fits")).data)


class TestFourier(unittest.TestCase):

    def test_fft(self):
        fft = FFT(convert_mask_to_locations(np.ones_like(IMG, dtype='int')),
                  img_size=128)
        ft_img = fft.op(IMG)
        rec_ft = fft.adj_op(ft_img)
        np.testing.assert_allclose(trunc_to_zero(rec_ft, eps=1.0e-8), IMG, rtol=1e-4)

    def test_nfft(self):
        nfft = NFFT(convert_mask_to_locations(np.ones_like(IMG, dtype='int')),
                  img_size=128)
        ft_img = nfft.op(IMG)
        rec_ft = nfft.adj_op(ft_img)
        np.testing.assert_allclose(trunc_to_zero(rec_ft, eps=1.0e-8), IMG, rtol=1e-4)
