export interface SignalSummary {
  id: string
  name: string
  sampleRate: number
  samples: number
  duration: number
  origin: 'upload' | 'demo' | 'derived'
  sourceId: string | null
  peak: number
  rms: number
  peakDb: number
  crestFactor: number
  format?: string
  channels?: number
  bitDepth?: number | null
  bitrateKbps?: number
  isLossy?: boolean
  fileSizeBytes?: number
  groundTruth?: string
}

export interface WaveformStats {
  peak: number
  rms: number
  peakDb: number
  rmsDb: number
  crestFactor: number
  zeroCrossings: number
}

export interface WaveformData {
  min: number[]
  max: number[]
  startTime: number
  endTime: number
  duration: number
  stats: WaveformStats
}

export interface Curve {
  x: number[]
  y: number[]
}

export interface FftData extends Curve {
  nyquist: number
  peakFrequency: number
  spectralEnergy: number
  dominant: { frequency: number; magnitude: number }[]
}

export interface SpectrogramData {
  width: number
  height: number
  data: string
  dbMin: number
  dbMax: number
  maxFrequency: number
  duration: number
  freqResolution: number
  timeResolution: number
}

export interface VadData {
  frameTimes: number[]
  energyDb: number[]
  bandRatio: number[]
  isSpeech: boolean[]
  segments: { start: number; end: number }[]
  speechDuration: number
  silenceDuration: number
  speechRatio: number
  segmentCount: number
}

export interface FilterData {
  response: Curve
  cutoffs: number[]
  nyquist: number
  label: string
  result?: SignalSummary
}

export interface DenoiseData {
  result: SignalSummary
  snrBefore?: number
  snrAfter?: number
  correlationAfter?: number
}

export interface SeparateData {
  foreground: SignalSummary
  background: SignalSummary
  truth?: { foregroundCorrelation: number; backgroundCorrelation: number }
}

export interface ResampleData {
  result: SignalSummary
  mse: number
  snr: number
  correlation: number
}

export interface SamplingDemo {
  continuous: { t: number[]; x: number[] }
  reconstructed: { t: number[]; x: number[] }
  samples: { t: number[]; x: number[] }
  nyquist: number
  aliasing: boolean
  aliasedFrequency: number | null
  mse: number
  correlation: number
  snr: number
}

export interface CompareData {
  a: SignalSummary
  b: SignalSummary
  resampled: boolean
  metrics: {
    mse: number
    snr: number
    correlation: number
    rmsDifference: number
    peakDifference: number
  }
  waveformA: { min: number[]; max: number[] }
  waveformB: { min: number[]; max: number[] }
  spectrumA: Curve
  spectrumB: Curve
}

export type ViewId =
  | 'dashboard'
  | 'waveform'
  | 'spectrum'
  | 'spectrogram'
  | 'vad'
  | 'sampling'
  | 'filter'
  | 'separation'
  | 'denoise'
  | 'compare'
