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

export interface Region {
  start: number
  end: number
}

export interface TrimData {
  result: SignalSummary
  sourceDuration: number
  selectionDuration: number
  removedDuration: number
}

export interface FadeData {
  result: SignalSummary
  shape: string
}

export interface FadeShapes {
  t: number[]
  curves: Record<string, number[]>
}

export interface MergeClip {
  id: string
  name: string
  duration: number
  resampled: boolean
  sourceSampleRate: number
}

export interface MergeData {
  result: SignalSummary
  sampleRate: number
  clips: MergeClip[]
  /** Output time of each seam, for marking where clips meet. */
  boundaries: number[]
  totalSourceDuration: number
}

export interface SilenceData {
  levels: Curve
  /** What the slider holds: dB below this signal's own peak. */
  relativeThresholdDb: number
  /** What the level distribution suggests, for the Auto button. */
  suggestedThresholdDb: number
  /** The resulting absolute dBFS threshold, for the chart. */
  thresholdDb: number
  peakDb: number
  regions: Region[]
  keptRegions: Region[]
  removedDuration: number
  keptDuration: number
  totalDuration: number
  regionCount: number
  result?: SignalSummary
}

/** Measured evidence that a time/pitch transform did what it claimed. */
export interface Verification {
  expectedCents: number
  measuredCents: number | null
  errorCents: number | null
  confidence: number | null
  sourceF0: number | null
  resultF0: number | null
  f0Stable: boolean
  f0SpreadCents: number | null
}

export interface SpeedData {
  result: SignalSummary
  rate: number
  preservePitch: boolean
  method: string
  sourceDuration: number
  resultDuration: number
  verification: Verification
}

export interface PitchData {
  result: SignalSummary
  semitones: number
  ratio: number
  sourceDuration: number
  resultDuration: number
  verification: Verification
}

export interface VocalsData {
  preset: string
  levelMatched: boolean
  metrics: {
    vocalSuppressionDb: number
    stemCorrelation: number
    vocalEnergyShare: number
  }
  karaoke?: SignalSummary
  acapella?: SignalSummary
  truth?: { vocalsCorrelation: number; instrumentalCorrelation: number }
}

/* -------------------------------------------------------------------------- */
/* Secure vault                                                                */
/* -------------------------------------------------------------------------- */

export interface VaultHeader {
  magic: string
  version: number
  kdf: string
  scryptN: number
  scryptR: number
  scryptP: number
  saltHex: string
  nonceHex: string
  payloadLength: number
  headerSize: number
  tagSize: number
}

export interface VaultCapacity {
  width: number
  height: number
  pixels: number
  bitsPerChannel: number
  bitsPerPixel: number
  capacityBits: number
  capacityBytes: number
  usedBytes: number
  remainingBytes: number
  utilization: number
}

export interface VaultAudioInfo {
  sampleRate?: number
  channels?: number
  durationSeconds?: number
  format?: string
  bitDepth?: number | null
  bitrateKbps?: number
  isLossy?: boolean
}

export interface VaultEncodeData {
  imageId: string
  imageName: string
  signalId: string | null
  usedCover: boolean
  source: {
    filename: string
    size: number
    sha256: string
    audio: VaultAudioInfo
    histogram: number[]
    entropy: number
  }
  originalSize: number
  containerSize: number
  ciphertextSize: number
  headerSize: number
  tagSize: number
  pngSize: number
  overheadBytes: number
  sha256: string
  capacity: VaultCapacity
  timingsMs: Record<string, number>
  header: VaultHeader
  cipherHistogram: number[]
  cipherEntropy: number
}

export interface VaultDecodeData {
  fileId: string
  signalId: string | null
  filename: string
  extension: string
  declaredSize: number
  recoveredSize: number
  expectedSha256: string
  recoveredSha256: string
  integrityVerified: boolean
  sizeMatches: boolean
  audio: VaultAudioInfo
  timingsMs: Record<string, number>
  header: VaultHeader
}

export interface VaultInspectData {
  valid: boolean
  header: VaultHeader
  capacity: VaultCapacity
  pngSize: number
}

export interface VaultPixelRow {
  index: number
  x: number
  y: number
  before: number[]
  after: number[]
  beforeBits: string[]
  afterBits: string[]
  embedded: number[]
  changed: boolean[]
}

export interface VaultPixelData {
  rows: VaultPixelRow[]
  windowSize: number
  bitsPerChannel: number
}

export interface VaultTamperData {
  imageId: string
  imageName: string
  flippedBits: number
  totalChannels: number
  fractionChanged: number
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
  | 'editor'
  | 'silence'
  | 'merge'
  | 'timepitch'
  | 'vocals'
  | 'vault'
  | 'compare'
