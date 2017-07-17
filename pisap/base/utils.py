##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module define usefull helper.
"""
import os
import tempfile
import uuid
import numpy as np
import pisap

def convert_mask_to_locations(mask):
    """ Return the converted Cartesian mask as sampling locations.

    Parameters:
    -----------
        mask: np.ndarray, {0,1} 2D matrix
    Returns:
    -------
        samples_locations: np.ndarray,
    """
    row, col = np.where(mask==1)
    row = row.astype('float') / mask.shape[0] - 0.5
    col = col.astype('float') / mask.shape[0] - 0.5
    return np.c_[row, col]


def convert_locations_to_mask(samples_locations, img_size):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters:
    -----------
        samples_locations: np.ndarray,
        img_size: int, size of the desired square mask
    Returns:
    -------
        mask: np.ndarray, {0,1} 2D matrix
    """
    samples_locations = samples_locations.astype('float')
    samples_locations += 0.5
    samples_locations *= img_size
    samples_locations = samples_locations.astype('int')
    mask = np.zeros((img_size, img_size))
    mask[samples_locations[:,0], samples_locations[:,1]] = 1
    return mask


def to_complex_pywt_wavelet(cube_r, cube_i):
    """ merge two real pywt wavelet to one complex.
    """
    cube = [cube_r[0] + 1.j * cube_i[0]]
    for ks in range(1, len(cube_r)):
        tmp = []
        tmp.append(cube_r[ks][0] + 1.j * cube_i[ks][0])
        tmp.append(cube_r[ks][1] + 1.j * cube_i[ks][1])
        tmp.append(cube_r[ks][2] + 1.j * cube_i[ks][2])
        cube.append(tuple(tmp))
    return cube


def pywt_coef_is_cplx(cube):
    """ Return True if any of the coef in a pywavelet wavelet coefs is complex.
    """
    res = np.iscomplex(cube[0]).any()
    for scale in cube[1:]:
        res = res or any([np.iscomplex(band).any() for band in scale])
    return res


def pywt_coef_cplx_sep(cube):
    """ Separate the pywavelet wavelet complex coefs in two pywavelet wavelet
        coefs: real and imaginary part.
    """
    cube_r = [cube[0].real]
    cube_i = [cube[0].imag]
    for scale in cube[1:]:
        cube_r.append(tuple([band.real for band in scale]))
        cube_i.append(tuple([band.imag for band in scale]))
    return cube_r, cube_i


def generic_l2_norm(x):
    """ Compute the L2 norm for the given input.

    Parameters:
    -----------
    x: np.ndarray or DictionaryBase

    Return:
    ------
    norm: float, the L2 norm of the input.
    """
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x)
    elif isinstance(x, pisap.base.dictionary.DictionaryBase):
        return np.linalg.norm(x._data)
    else:
        TypeError("'generic_l2_norm' only accpet 'np.ndarray' or "
                    + "'DictionaryBase': {0} not reconized.".format(type(x)))


def generic_l1_norm(x):
    """ Compute the L1 norm for the given input.

    Parameters:
    -----------
    x: np.ndarray or DictionaryBase

    Return:
    ------
    norm: float, the L1 norm of the input.
    """
    if isinstance(x, np.ndarray):
        return np.abs(x).sum()
    elif isinstance(x, pisap.base.dictionary.DictionaryBase):
        return x.absolute._data.sum()
    else:
        TypeError("'generic_l1_norm' only accpet 'np.ndarray' or "
                    + "'DictionaryBase': {0} not reconized.".format(type(x)))


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


def min_max_normalize(img):
    """ Center and normalize the given array.
    Parameters:
    ----------
    img: np.ndarray
    """
    min_img = img.min()
    max_img = img.max()
    return (img - min_img) / (max_img - min_img)


def l2_normalize(img):
    """ Center and normalize the given array.
    Parameters:
    ----------
    img: np.ndarray
    """
    return img / np.linalg.norm(img)


def isapproof_pathname(pathname):
    """ Return the isap-sanityzed pathname.
    Note:
    -----
    If 'jpg' or 'pgm' (with any case for each letter) are in the pathname, it
    will corrupt the format detection in ISAP.
    """
    new_pathname = pathname # copy
    for frmt in ["pgm", "jpg"]:
        idx = pathname.lower().find(frmt)
        if idx == -1:
            continue
        tmp = "".join([str(nb) for nb in np.random.randint(9, size=len(frmt))])
        new_pathname = new_pathname[:idx] + tmp + new_pathname[idx+len(frmt):]
    return new_pathname


def isapproof_mkdtemp():
    """ The ISAP proof version of tempfile.mkdtemp.
    Note:
    -----
    If 'jpg' or 'pgm' (with any case for each letter) are in the pathname, it
    will corrupt the format detection in ISAP.
    """
    dirname = os.path.join(tempfile.gettempdir(), "tmp" + str(uuid.uuid4()).split('-')[0])
    dirname = isapproof_pathname(dirname)
    os.mkdir(dirname)
    return dirname


def set_bands_shapes(bands_lengths, ratio=None):
    """ Return the bands_shapes attributs from the given bands_lengths.
    """
    if ratio is None:
        ratio = np.ones_like(bands_lengths)
    bands_shapes = []
    for ks, scale in enumerate(bands_lengths):
        scale_shapes = []
        for kb, padd in enumerate(scale):
            shape = (int(np.sqrt(padd*ratio[ks, kb])), int(np.sqrt(padd/ratio[ks, kb])))
            scale_shapes.append(shape)
        bands_shapes.append(scale_shapes)
    return bands_shapes


def to_2d_array(a):
    """Convert:
        2d array to 2d array
        1d array to 2d array
        scalar to 2d array.

        Parameters:
        ----------
        a: matplotlib axe or np.ndarray of matplotlib axe (1d or 2d)

        Note:
        -----
        no check done in the function
    """
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            return a
        elif a.ndim == 1:
            return a[:, None]
    else:
        return np.array(a)[None, None]


def secure_mkdir(dirname):
    """ Silently pass if directory ``dirname`` already exist.
    """
    try:
        os.mkdir(dirname)
    except OSError:
        pass


def _isap_transform(data, **kwargs):
    """ Helper return the transformation coefficient in a isap-cube format.
    """
    tmpdir = isapproof_mkdtemp()
    in_image = os.path.join(tmpdir, "in.fits")
    out_mr_file = os.path.join(tmpdir, "cube.mr")
    try:
        pisap.io.save(data, in_image)
        pisap.extensions.mr_transform(in_image, out_mr_file, **kwargs)
        image = pisap.io.load(out_mr_file)
        isap_trf_buf = image.data
        header = image.metadata
    except:
        raise
    finally:
        for path in (in_image, out_mr_file):
            if os.path.isfile(path):
                os.remove(path)
        os.rmdir(tmpdir)
    return isap_trf_buf, header


def isap_transform(data, **kwargs):
    """ Return the transformation coefficient in a isap-cube format.

    Return:
    -------
    cube: np.ndarray, the ISAP output
    header: Python dictionary, the .fits header of the ISAP output
    """
    if np.any(np.iscomplex(data)):
        isap_trf_buf_r, header = _isap_transform(data.real, **kwargs)
        isap_trf_buf_i, _ = _isap_transform(data.imag, **kwargs)
        return isap_trf_buf_r + 1.j * isap_trf_buf_i, header
    else:
        return _isap_transform(data.astype(float), **kwargs) # cube, header


def _isap_recons(data, header):
    """ Helper return the reconstructed image.
    """
    cube = pisap.Image(data=data, metadata=header)
    tmpdir = isapproof_mkdtemp()
    in_mr_file = os.path.join(tmpdir, "cube.mr")
    out_image = os.path.join(tmpdir, "out.fits")
    try:
        pisap.io.save(cube, in_mr_file)
        pisap.extensions.mr_recons(in_mr_file, out_image)
        isap_recs_buf = pisap.io.load(out_image)
    except:
        raise
    finally:
        for path in (in_mr_file, out_image):
            if os.path.isfile(path):
                os.remove(path)
        os.rmdir(tmpdir)
    return isap_recs_buf


def isap_recons(data, header):
    """ Return the reconstructed image.
    """
    if np.any(np.iscomplex(data)):
        isap_recs_buf_r = _isap_recons(data.real, header)
        isap_recs_buf_i = _isap_recons(data.imag, header)
        return isap_recs_buf_r + 1.j * isap_recs_buf_i
    else:
        return _isap_recons(data.astype(float), header)


def run_both(linear_op, data, nb_scale, isap_kwargs):
    """ Run ispa and pisap trf and reconstruction.
    """
    init_kwarg = {'maxscale': nb_scale}
    linear_op = linear_op(**init_kwarg)
    trf = linear_op.op(data)
    trf_img = trf.to_cube()
    trf_isap_img, header =  isap_transform(data, **isap_kwargs)
    recs_img = linear_op.adj_op(trf)
    recs_isap_img = isap_recons(trf.to_cube(), header)
    return (trf_img, trf_isap_img), (recs_img, recs_isap_img)


def get_curvelet_bands_shapes(img_shape, nb_scale, nb_band_per_scale):
    """ Return the 'bands_shapes' for FastCurveletTransform.
    """
    img = np.zeros(img_shape)
    tmpdir = isapproof_mkdtemp()
    in_image = os.path.join(tmpdir, "in.fits")
    out_mr_file = os.path.join(tmpdir, "cube.mr")
    kwargs = {'number_of_scales': nb_scale,
              'type_of_multiresolution_transform': 28,
               }
    try:
        pisap.io.save(img, in_image)
        pisap.extensions.mr_transform(in_image, out_mr_file, **kwargs)
        cube = pisap.io.load(out_mr_file).data
    except:
        raise
    finally:
        for path in (in_image, out_mr_file):
            if os.path.isfile(path):
                os.remove(path)
        os.rmdir(tmpdir)
    bands_shapes = []
    padd = 1 + nb_scale
    for ks in range(nb_scale):
        band_shapes = []
        for kb in range(nb_band_per_scale[ks]):
            Nx = int(cube[padd])
            Ny = int(cube[padd+1])
            band_shapes.append((Nx, Ny))
            padd += (Nx * Ny + 2)
        bands_shapes.append(band_shapes)
    return bands_shapes
