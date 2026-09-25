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

Editing and production layers, built on the modules above:

    editing       -> trim, cut, splice, merge, fades, level matching
    silence       -> silence detection (Schmitt trigger) and removal
    timescale     -> time stretching, pitch shifting, F0 verification
    vocals        -> karaoke / a cappella stems with separation metrics

Classical stereo source separation (see stereo_sep/README.md):

    stereo_sep    -> stereo karaoke / vocal-focus separation from Mid/Side
                     geometry, STFT soft masking, HPSS, harmonic and
                     repetition analysis. No ML. Requires a stereo source:
                     the mono downmix (L+R)/2 destroys the Side signal, in
                     which a centre-panned vocal cancels exactly.
"""
