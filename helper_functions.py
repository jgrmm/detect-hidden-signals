#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper routines for extracting coherency features from DAS data

Created on Thu Oct 28 12:27:05 2021

@author: Julius Grimm (ISTerre, Université Grenoble Alpes)
"""



def spec_plot(a1, ch=0, fig=None, cmap='inferno', vrange=None):
    """
    Plot spectrogram for a single DAS channel

    Parameters
    ----------
    a1 : A1Section class instance
        Object containing strain-rate data and header information
    ch : int
        Channel number
    fig :
        A fig handle necessary to plot several plot on the same figure.
        Defaults to None
    cmap : matplotlib.colors.ListedColormap
        Matplotlib colormap. Default is "inferno"
    vrange : tuple or list
        (vmin,vmax) range for colormap. The default will set the max value

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from scipy import signal
    import matplotlib.pyplot as plt
    import numpy as np

    if not fig:
        fig, ax = plt.subplots(2, figsize=(12,7), constrained_layout=True)


    # Setting range for colorbar
    if not vrange:
        vmax = np.max(np.absolute(a1.data), axis=None)
        vmin = -vmax
    else:
        vmin = vrange[0]
        vmax = vrange[1]


    # Plotting time series
    ax[0].plot(a1.time(), a1.data[:,ch], 'k')
    ax[0].set_xlim(min(a1.time()), max(a1.time()))
    ax[0].set_ylabel("Strain-rate [nm/m/s]")

    # Plotting spectrogram
    fs = 1/a1.data_header['dt']
    freq, t, Sxx = signal.spectrogram(a1.data[:,ch], fs)
    s = ax[1].pcolormesh(t+min(a1.time()), freq, np.log10(Sxx/Sxx.max()),
                         shading='gouraud',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlim(min(a1.time()),max(a1.time()))
    fig.colorbar(mappable=s, ax=ax[1])
    fig.suptitle(f'channel {ch}: {a1.dist()[ch]:.2f} m', fontsize=16)

    return fig



def tx_plot(a1, fig=None, cmap='seismic', vrange=None, splot=(1, 1, 1),
            title='', drange=None, trange=None, outfile=None):
    '''
    Plot a DAS section as a raster image

    Parameters
    ----------
    a1 : A1Section class instance
        Object containing strain-rate data and header information
    fig :
        A fig handle necessary to plot several plot on the same figure.
        Defaults to None
    cmap : matplotlib.colors.ListedColormap
        Matplotlib colormap. Default is "seismic"
    vrange : tuple or list
        (vmin,vmax) range for colormap. The default will set the max value
    splot : tuple
        Subplot position. Default is (1,1,1)
    title : string
        Title of figure
    drange : tuple or list
        Channel range to be plotted
    trange : tuple or list
        Time range to be plotted
    outfile : string
        Location where to save figure. Default is None, i.e. figure is not
        saved

    Returns
    -------
    fig : matplotlib.figure.Figure
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    if not fig:
        fig = plt.figure(figsize=(18, 9))

    # Making sure space and time axis are in correct order, otherwise switch
    transpose = False
    if a1.data_header['data_axis'] == 'space_x_time':
        transpose = True

    # Extracting correct axes labels
    dist = a1.data_header['dist']
    time = a1.data_header['time']
    if drange is None:
        d1=0
        if transpose:
            d2 = a1.data.shape[1]
        else:
            d2=a1.data.shape[0]
        dmin=dist[0]
        dmax=dist[-1]
    else:
        dmin=drange[0]
        dmax=drange[1]
        d1=np.nanargmin(np.abs(dist-dmin))
        d2=np.nanargmin(np.abs(dist-dmax))

    if trange is None:
        t1=0
        if transpose:
            t2 = a1.data.shape[1]
        else:
            t2 = a1.data.shape[0]
        tmin=time[0]
        tmax=time[-1]
    else:
        tmin=trange[0]
        tmax=trange[1]
        t1=np.nanargmin(np.abs(time-tmin))
        t2=np.nanargmin(np.abs(time-tmax))


    # Setting range for colorbar
    if not vrange:
        vmax = np.max(np.absolute(a1.data[d1:d2]), axis=None)
        vmin = -vmax
    else:
        vmin = vrange[0]
        vmax = vrange[1]


    # Plotting raster image
    ax = plt.subplot(splot[0], splot[1], splot[2])
    extent = [dmin, dmax, tmax-tmin, 0]

    if transpose:
        pos = ax.imshow(a1.data[d1:d2, t1:t2], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect='auto', extent=extent)
    else:
        pos = ax.imshow(a1.data[t1:t2,d1:d2], cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect='auto', extent=extent)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title('DAS Section ' + title)
    fig.colorbar(pos)

    if outfile:
        plt.savefig(outfile, dpi=300)

    return fig



def arg_closest(array, value):
    """
    Find the argument of entry in array closest to value

    Parameters
    ----------
    array : array-like
        Input array
    value : float
        Single float for which the closest element in `array` is sought after

    Returns
    -------
    arg_close : int
        Index of the element in `array` that is closest to `value`
    """
    import numpy as np
    res = array - value
    arg_close = np.argmin(np.abs(res))
    return arg_close

def fftfreq(n, fs=1.0):
    """
    Create array of frequencies as obtained after fft of arbitrary data of
    length n and sampling frequency fs

    Parameters
    ----------
    n : int
        Length of data (time axis)
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    freqs : array-like
        Array of frequencies as obtained after fft
    """
    import numpy as np
    dt = 1 / fs
    scaler = 1.0 / (n * dt)
    freqs = np.empty(n, int)
    N = (n-1)//2 + 1
    f_pos = np.arange(0, N, dtype=int) # positive frequencies
    f_neg = np.arange(-(n//2), 0, dtype=int) # negative frequencies
    freqs[:N] = f_pos
    freqs[N:] = f_neg
    freqs *= scaler
    return freqs

def dft_single(data, freq, fs):
    """
    Calculate the Fourier coefficient of single time series for a single
    specified frequency

    Parameters
    ----------
    data : 1D array
        Input data, one single channel
    freq : float
        Frequency for which to extract the Fourier coefficient
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    X_k : complex
        Fourier coefficient for frequency `freq`
    """
    import numpy as np
    N = len(data)
    freqs = fftfreq(N, fs)
    k = arg_closest(freqs, freq)
    n = np.array(range(N))
    weight = np.cos(2*np.pi*k*n/N) - 1j * np.sin(2*np.pi*k*n/N)
    X_k = sum(data * weight)
    return X_k


def dft_single_matrix(data_matrix, freq, fs):
    """Calculate the Fourier coefficient of multiple time series for a single
    specified frequency

    Parameters
    ----------
    data_matrix : 2D array
        Input data, multiple channels
    freq : float
        Frequency for which to extract the Fourier coefficient
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    X_coeff : complex 1D array
        Fourier coefficients for all channels for frequency `freq`
    """
    import numpy as np
    from scipy.signal.windows import hann
    import config

    # Set weights in config file to avoid multiple computations each time
    # function is called
    weights = config.weights

    # Initialize sinusoidal weights for the first call of function
    if weights is None:
        print("Initializing weights...")
        N = data_matrix.shape[0]
        freqs = fftfreq(N, fs)
        k = arg_closest(freqs, freq)
        n = np.array(range(N))
        hann_window = hann(N)
        weights = np.cos(2*np.pi*k*n/N) - 1j * np.sin(2*np.pi*k*n/N)
        # Taper sinusoids by Hann window to reduce spectral leakage
        weights *= hann_window
        config.weights = weights

    X_coeff = np.dot(data_matrix.T, weights)

    return X_coeff



def dft_matrix_band(data_matrix, freq_band, fs):
    """
    Calculate the Fourier coefficient of multiple time series for a multiple
    frequencies

    Parameters
    ----------
    data_matrix : 2D array
        Input data, multiple channels
    freq_band : list
        List of frequencies for which to extract Fourier coefficients
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    X_coeff : complex 2D array
        Fourier coefficients for all channels for frequencies in `freq_band`
    """
    import numpy as np
    from scipy.signal.windows import hann
    import config

    # Set weights in config file to avoid multiple computations each time
    # function is called
    weight_matrix = config.weights

    # Initialize sinusoidal weights for the first call of function
    if weight_matrix is None:
        #print("Initializing weight matrix...")
        N = data_matrix.shape[0]
        n = np.array(range(N))
        hann_window = hann(N)
        weight_matrix = np.empty((freq_band.shape[0], N), dtype=np.complex128)
        freqs = fftfreq(N, fs)

        for ind, freq in enumerate(freq_band):
            k = arg_closest(freqs, freq)
            weight_matrix[ind, :] = (np.cos(2*np.pi*k*n/N) -
                                     1j * np.sin(2*np.pi*k*n/N))
        # Taper sinusoids by Hann window to reduce spectral leakage
        weight_matrix *= hann_window
        config.weights = weight_matrix

#     X_coeff = np.dot(data_matrix_window.T, weight_matrix.T)
    X_coeff = np.einsum('ij, ki', data_matrix, weight_matrix)

    return X_coeff



def cohm_short_single(data_matrix, freq, fs):
    """
    Compute the coherence matrix of short time window for a single frequency.

    Parameters:
    ----------
    data_matrix : 2D array
        Input data, multiple channels
    freq : float
        Frequency for which to extract the Fourier coefficient
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    cohm : complex matrix
        Coherence matrix of a short window (not normalized)
    power : complex 1D array
        Signal power of all channels at frequency `freq`
    """
    import numpy as np
    npts, nStations = data_matrix.shape
    data_scalar = np.empty(nStations, dtype=np.complex128)
    data_scalar = dft_single_matrix(data_matrix, freq, fs)

    # Computing (not normalized) coherence matrix, same as cross-spectral
    # density
    cohm = data_scalar.ravel()[:,None]*np.conj(
            data_scalar).ravel()[None,:]

    # Computing signal power
    power = np.abs(data_scalar) ** 2

    return cohm, power



def cohm_short_band(window_id, ws, freq_band, fs):
    """Compute coherence matrix of short time window for a frequency band using
    matrix multiplication.

    Parameters:
    ----------
    window_id : int
        Identifier of the short window number inside long window
    ws : int
        Length of short window in seconds
    freq_band : list
        List of frequencies for which to extract Fourier coefficients
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    cohm : complex matrix
        Coherence matrix of a short window (not normalized)
    power : complex 1D array
        Signal power of all channels at frequency `freq`
    """
    # Extract short window from the longer window
    first_sample = window_id * (ws//2)
    last_sample = first_sample + ws
    data_matrix_window = data_ave_window[first_sample:last_sample, :]

    npts, nStations = data_matrix_window.shape
    nfreqs = freq_band.shape[0]
    data_scalar = dft_matrix_band(data_matrix_window, freq_band, fs)

    # Computing (not normalized) coherence matrix, same as cross-spectral
    # density
    data_scalar_conj = np.conj(data_scalar)
    cohm = np.einsum('i..., k... -> ...ik', data_scalar, data_scalar_conj)

    # Computing signal power
    power = (np.abs(data_scalar) ** 2).swapaxes(0,1)

    return cohm, power



def cohm_long_single(data_matrix, ws, freq, fs):
    """
    Compute full-sample normalized coherence

    Parameters:
    ----------
    data_matrix : 2D array
        Input data, multiple channels
    ws : int
        Length of short window in seconds
    freq : float
        Frequency for which to extract the Fourier coefficient
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    cohm_norm : complex 2D array
        Full-sample coherence matrix (normalized by power)
    power_matrix : complex matrix
        Outer product of signal power of all channels at frequency `freq`
    """
    import numpy as np

    npts, nStations = data_matrix.shape
    nr_windows = (npts // (ws//2))  - 1

    cohm_ave = np.zeros((nStations, nStations), dtype=np.complex128)
    power_ave = np.zeros(nStations)

    # Calculating (not normalized) coherence matrix for each window, then
    # averaging all results and normalize by signal power
    for window_id in range(nr_windows):
        print(window_id)
        first_sample = window_id * (ws//2)
        last_sample = first_sample + ws
        data_window = data_matrix[first_sample:last_sample, :]
        cohm_window, power_window = cov_matrix_single_freq_short_matrix(
                data_window,freq,fs)
        cohm_ave += cohm_window
        power_ave += power_window

    cohm_ave /= nr_windows
    power_root = np.sqrt(power_ave / nr_windows)
    power_matrix = power_root.ravel()[:,None]*power_root.ravel()[None,:]

    power_matrix[power_matrix==0] = 1e-7  # To avoid dividing by zero
    cohm_norm = cohm_ave / (power_matrix)

    return cohm_norm, power_matrix



def coherence_matrix_parallel(data, ws, freq_band_list, fs):
    """
    Compute full-sample normalized coherence using multiprocessing

    Parameters:
    ----------
    data : 2D array
        Input data, multiple channels
    ws : int
        Length of short window in seconds
    freq_band_list : List of arrays
        List of frequency bands for which to extract Fourier coefficients
    fs : float
        Sampling frequency of data in Hz

    Returns
    -------
    cohm_norm : complex 2D array
        Full-sample coherence matrix (normalized by power)
    """
    import numpy as np
    import multiprocessing
    from itertools import repeat

    npts, nStations = data.shape
    nr_freq_band = len(freq_band_list)
    nrF_per_band = [len(fb) for fb in freq_band_list]
    flat_list = [f for fband in freq_band_list for f in fband]
    freq_array = np.array(flat_list)
    total_freqs = freq_array.shape[0]
    nr_windows = (npts // (ws//2))  - 1


    # Initialize arrays for the coherence matrix and signal power
    cohm_ave = np.zeros((total_freqs, nStations, nStations), dtype=np.complex128)
    power_ave = np.zeros((total_freqs,nStations))


    # Arguments to be passed to starmap
    arg0 = range(nr_windows)
    arg1 = ws
    arg2 = freq_array
    arg3 = fs
    # Open multiprocessing pool and compute coherence matrix for short windows
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    results = pool.starmap(cohm_short_band, zip(arg0, repeat(arg1),
                                                repeat(arg2), repeat(arg3)))
    pool.close()
    results_covmat = [result[0] for result in results]
    results_power = [result[1] for result in results]
    covmat_array = np.stack(results_covmat)
    power_array = np.stack(results_power)
    print("Shape after parallel processing: ", covmat_array.shape)

    # Averaging all short windows for one long window
    cohm_ave = np.mean(covmat_array,axis=0)
    power_ave = np.mean(power_array,axis=0)

    # Computing signal power matrix
    power_matrix = np.einsum('...i, ...j -> ...ij', power_ave, power_ave)
    power_matrix[power_matrix==0] = 1e-7

    # Normalize coherence matrix by signal power
    cohm_norm = np.abs(cohm_ave)**2 / (power_matrix)

    return cohm_norm



def reconstruct_matrix(X, ind, inv_scale=True):
    first_covmat_recov = np.eye(first_covmat.shape[0])
    if inv_scale:
        X_ind = scaler.inverse_transform(X[ind])
    first_covmat_recov[np.triu_indices(first_covmat.shape[0], k = 1)] = X_ind
    first_covmat_recov.T[np.triu_indices(first_covmat.shape[0], k = 1)] = X_ind
    return first_covmat_recov

def reconstruct_matrix_pca(X_pca, ind, pca,inv_scale=True):
    first_features_recov = X_pca[ind,:] @ pca.components_
    if inv_scale:
        first_features_recov = scaler.inverse_transform(first_features_recov)
    first_covmat_recov = np.eye(first_covmat.shape[0])
    first_covmat_recov[np.triu_indices(first_covmat.shape[0], k = 1)] = first_features_recov
    first_covmat_recov.T[np.triu_indices(first_covmat.shape[0], k = 1)] = first_features_recov
    return first_covmat_recov
