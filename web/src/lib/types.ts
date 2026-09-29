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

/** A karaoke or lyrics run on the server, which the page polls until done. */
export interface Job<T> {
  id: string
  status: 'queued' | 'running' | 'done' | 'error'
  /** Jobs that will use the GPU before this one; null once it has started. */
  ahead: number | null
  elapsedS: number | null
  /** True when another request had already started or finished this work. */
  reused: boolean
  result: T | null
  error: string | null
}

export type VocalsEngine = 'ml' | 'classical'

export interface VocalsEngines {
  ml: boolean
  classical: boolean
  mlModel: string
}

export interface VocalsData {
  engine: VocalsEngine
  preset: string
  levelMatched: boolean
  metrics: {
    vocalSuppressionDb: number
    stemCorrelation: number
    vocalEnergyShare: number
  }
  karaoke?: SignalSummary
  acapella?: SignalSummary
  truth?: {
    vocalsCorrelation: number
    instrumentalCorrelation: number
    vocalsSdrDb: number
    instrumentalSdrDb: number
  }
}

export type LyricsLanguage = 'en' | 'hi' | 'bn'
export type LyricsMode = 'song' | 'voice'

export interface TranscribeStatus {
  available: boolean
  model: string
  device: string | null
  languages: { code: LyricsLanguage; name: string }[]
}

export interface LyricWord {
  start: number
  end: number
  text: string
}

export interface LyricLine {
  start: number
  end: number
  text: string
  words: LyricWord[]
}

export interface TranscribeData {
  language: LyricsLanguage
  languageName: string
  mode: LyricsMode
  model: string
  device: string
  lines: LyricLine[]
  timings: {
    separateS: number | null
    transcribeS: number
    stemReused: boolean
    loadModelsS: number | null
    alignS: number | null
  }
  droppedSilent: number
  recoveredLines: number
  /** True when the lines are the user's own lyrics, synced to the audio. */
  aligned: boolean
  /** Share of the user's words Whisper also heard (aligned only). */
  matchRate: number | null
  /** Mean probability of the user's words given the audio (aligned only). */
  confidence: number | null
  vocals: SignalSummary | null
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

/* -------------------------------------------------------------------------- */
/* Hidden note in a song                                                       */
/* -------------------------------------------------------------------------- */

export interface NoteCapacity {
  capacityBytes: number
  channels: number
}

/** One interleaved 16-bit sample whose lowest bit was nudged to carry a bit. */
export interface NoteSampleRow {
  index: number
  before: number
  after: number
}

export interface HideNoteData {
  /** The WAV to keep — named like the song, stored in the vault session. */
  file: { id: string; filename: string; size: number }
  /** The same audio, registered as a session signal for playback. */
  signal: SignalSummary
  stats: {
    messageBytes: number
    capacityBytes: number
    totalSamples: number
    samplesUsed: number
    samplesChanged: number
    changeFraction: number
    maxAmplitudeChange: number
    psnrDb: number
    channels: number
    sampleRate: number
  }
  changedSamples: NoteSampleRow[]
  timingsMs: Record<string, number>
}

export interface RevealNoteData {
  message: string
  stats: { messageBytes: number; channels: number; sampleRate: number; format?: string }
  timingsMs: Record<string, number>
}

export type ViewId =
  | 'dashboard'
  | 'waveform'
  | 'spectrum'
  | 'spectrogram'
  | 'vad'
  | 'sampling'
  | 'filter'
  | 'denoise'
  | 'editor'
  | 'silence'
  | 'merge'
  | 'timepitch'
  | 'vocals'
  | 'lyrics'
  | 'vault'
  | 'note'
  | 'shared'
  | 'compare'

/** A file published to everyone on the network (the room's drop box). */
export interface SharedFile {
  id: string
  name: string
  size: number
  /** Unix time in seconds. */
  sharedAt: number
  kind: 'image' | 'audio' | 'file'
}
