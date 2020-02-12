# SNIPPET OF CODE TAKEN FROM librosa/core/spectrum.py
# https://github.com/librosa/librosa/blob/master/librosa/core/spectrum.py

import numpy as np

def db_to_power(S_db, ref=1.0):
    '''Convert a dB-scale spectrogram to a power spectrogram.
    This effectively inverts `power_to_db`:
        `db_to_power(S_db) ~= ref * 10.0**(S_db / 10)`
    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram
    ref : number > 0
        Reference power: output will be scaled by this value
    Returns
    -------
    S : np.ndarray
        Power spectrogram
    Notes
    -----
    This function caches at level 30.
    '''
    return ref * np.power(10.0, 0.1 * S_db)

def power_to_db(S, ref=1.0, amin=1e-10, top_db=None):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units
    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.
    Parameters
    ----------
    S : np.ndarray
        input power
    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.
        If callable, the reference value is computed as `ref(S)`.
    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`
    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``
    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``
    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude
    Notes
    -----
    This function caches at level 30.
    Examples
    --------
    Get a power spectrogram from a waveform ``y``
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)
    Compute dB relative to peak power
    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)
    Or compare to median power
    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)
    And plot the results
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()
    >>> plt.show()
    """

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec