"""Audio Signal Processing Toolkit — a modular DSP library.

Each module is independent and covers one stage of the classic audio
processing pipeline:

    io_utils      -> loading/saving audio, file metadata
    framing       -> splitting a signal into short-time frames
    vad           -> voice activity detection (energy + spectral features)
    spectral      -> FFT / STFT / spectrogram computation
    filters       -> Butterworth low/high/band-pass/stop filters
    separation    -> vocal/instrumental separation (classical DSP method)
    noise_reduction -> spectral-subtraction based denoising
    sampling      -> sampling-rate / aliasing / reconstruction demos
    metrics       -> MSE, SNR, correlation between two signals
"""
