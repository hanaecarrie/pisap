import unittest
import os.path as osp
import numpy as np

import pisap
from pisap.data import get_sample_data
from pisap.numerics.fourier import NFFT, FFT
from pisap.base.utils import convert_mask_to_locations, min_max_normalize


def trunc_to_zero(data, eps=1.0e-7):
    """ Threshold the given entries of data to zero if they are lesser than eps.
    Return:
    -----
    new_data: np.ndarray, copy of data with a truncated entries.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("wrong argument: only accept numpy array.")
    new_data = np.copy(data) # copy
    if np.issubsctype(data.dtype, np.complex):
        new_data.real[np.abs(new_data.real) < eps] = 0
        new_data.imag[np.abs(new_data.imag) < eps] = 0
    else:
        new_data[np.abs(new_data) < eps] = 0
    return new_data


class TestFourier(unittest.TestCase):

    def setUp(self):
        filepath = get_sample_data("astro-fits")
        self.img = min_max_normalize(pisap.io.load(filepath).data)

    def test_fft(self):
        fft = FFT(convert_mask_to_locations(np.ones_like(self.img, dtype='int')),
                  img_size=128)
        ft_img = fft.op(self.img)
        rec_ft = fft.adj_op(ft_img)
        rec_ft = trunc_to_zero(rec_ft)
        np.testing.assert_allclose(rec_ft, self.img, rtol=1e-4)

    def test_nfft(self):
        nfft = NFFT(convert_mask_to_locations(np.ones_like(self.img, dtype='int')),
                    img_size=128)
        ft_img = nfft.op(self.img)
        rec_ft = nfft.adj_op(ft_img)
        rec_ft =  trunc_to_zero(rec_ft)
        np.testing.assert_allclose(rec_ft, self.img, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
