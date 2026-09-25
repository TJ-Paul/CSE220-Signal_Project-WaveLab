import type {
  CompareData,
  DenoiseData,
  FadeData,
  FadeShapes,
  FftData,
  FilterData,
  MergeData,
  PitchData,
  ResampleData,
  SamplingDemo,
  SeparateData,
  SignalSummary,
  SilenceData,
  SpectrogramData,
  SpeedData,
  TrimData,
  VadData,
  VaultDecodeData,
  VaultEncodeData,
  VaultInspectData,
  VaultPixelData,
  VaultTamperData,
  VocalsData,
  WaveformData,
} from './types'

const BASE = '/api'

class ApiError extends Error {
  status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError('Cannot reach the Signal Lab API. Is the server running?', 0)
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new ApiError(detail?.detail ?? `Request failed (${res.status})`, res.status)
  }
  return res.json() as Promise<T>
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
}

export const api = {
  listSignals: () => request<{ signals: SignalSummary[] }>('/signals'),

  uploadSignal: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<SignalSummary>('/signals/upload', { method: 'POST', body: form })
  },

  loadDemo: (kind: 'speech' | 'song' | 'noisy') =>
    post<SignalSummary>(`/signals/demo/${kind}`),

  deleteSignal: (id: string) => request<{ ok: boolean }>(`/signals/${id}`, { method: 'DELETE' }),

  clearSignals: () => post<{ ok: boolean }>('/signals/clear'),

  audioUrl: (id: string) => `${BASE}/signals/${id}/audio`,

  waveform: (id: string, points = 2000, start = 0, end = -1) =>
    request<WaveformData>(
      `/signals/${id}/waveform?points=${points}&start=${start}&end=${end}`,
    ),

  fft: (id: string, opts: { points?: number; peaks?: number; frameMs?: number; center?: number } = {}) =>
    request<FftData>(
      `/signals/${id}/fft?points=${opts.points ?? 1400}&peaks=${opts.peaks ?? 5}` +
        `&frame_ms=${opts.frameMs ?? 0}&center=${opts.center ?? 0}`,
    ),

  spectrogram: (
    id: string,
    nFft = 2048,
    hop = 512,
    rangeDb = 70,
    scale?: { dbMin: number; dbMax: number },
  ) =>
    request<SpectrogramData>(
      `/signals/${id}/spectrogram?n_fft=${nFft}&hop=${hop}&range_db=${rangeDb}` +
        (scale ? `&db_min=${scale.dbMin}&db_max=${scale.dbMax}` : ''),
    ),

  vad: (id: string, body: { frameMs: number; hopMs: number; energyPercentile: number; bandRatioThresh: number }) =>
    post<VadData>(`/signals/${id}/vad`, body),

  filter: (
    id: string,
    body: { kind: string; cutoff?: number; band?: number[]; order: number; apply?: boolean },
  ) => post<FilterData>(`/signals/${id}/filter`, body),

  denoise: (id: string, body: { noiseDurationS: number; alpha: number; beta: number }) =>
    post<DenoiseData>(`/signals/${id}/denoise`, body),

  separate: (id: string, body: { marginBackground: number; marginForeground: number }) =>
    post<SeparateData>(`/signals/${id}/separate`, body),

  resample: (id: string, targetSr: number) =>
    post<ResampleData>(`/signals/${id}/resample`, { targetSr }),

  samplingDemo: (freq: number, fs: number, duration: number) =>
    request<SamplingDemo>(`/sampling/demo?freq=${freq}&fs=${fs}&duration=${duration}`),

  compare: (a: string, b: string) => request<CompareData>(`/compare?a=${a}&b=${b}`),

  /* Editing ---------------------------------------------------------------- */

  trim: (id: string, body: { start: number; end: number; mode: 'keep' | 'cut' }) =>
    post<TrimData>(`/signals/${id}/trim`, body),

  fade: (
    id: string,
    body: { fadeInS: number; fadeOutS: number; shape: string; normalize?: boolean },
  ) => post<FadeData>(`/signals/${id}/fade`, body),

  fadeShapes: () => request<FadeShapes>('/fades/shapes'),

  merge: (body: {
    ids: string[]
    crossfadeS: number
    gapS: number
    law: string
    normalize: boolean
    name?: string
  }) => post<MergeData>('/merge', body),

  silence: (
    id: string,
    body: {
      thresholdDb: number | null
      minSilenceS: number
      padS: number
      hysteresisDb: number
      apply?: boolean
    },
  ) => post<SilenceData>(`/signals/${id}/silence`, body),

  /* Time, pitch and vocals ------------------------------------------------- */

  speed: (id: string, body: { rate: number; preservePitch: boolean }) =>
    post<SpeedData>(`/signals/${id}/speed`, body),

  pitch: (id: string, body: { semitones: number }) =>
    post<PitchData>(`/signals/${id}/pitch`, body),

  vocals: (
    id: string,
    body: { preset: string; outputs: 'karaoke' | 'acapella' | 'both'; levelMatch: boolean },
  ) => post<VocalsData>(`/signals/${id}/vocals`, body),

  /* Secure vault ----------------------------------------------------------- */

  vaultEncode: (opts: {
    file: File
    password: string
    bitsPerChannel: number
    cover?: File | null
  }) => {
    const form = new FormData()
    form.append('file', opts.file)
    form.append('password', opts.password)
    form.append('bitsPerChannel', String(opts.bitsPerChannel))
    if (opts.cover) form.append('cover', opts.cover)
    return request<VaultEncodeData>('/vault/encode', { method: 'POST', body: form })
  },

  vaultInspect: (image: File, bitsPerChannel: number) => {
    const form = new FormData()
    form.append('image', image)
    form.append('bitsPerChannel', String(bitsPerChannel))
    return request<VaultInspectData>('/vault/inspect', { method: 'POST', body: form })
  },

  vaultDecode: (opts: { image: File; password: string; bitsPerChannel: number }) => {
    const form = new FormData()
    form.append('image', opts.image)
    form.append('password', opts.password)
    form.append('bitsPerChannel', String(opts.bitsPerChannel))
    return request<VaultDecodeData>('/vault/decode', { method: 'POST', body: form })
  },

  vaultPixels: (imageId: string, offset: number, count: number) =>
    request<VaultPixelData>(`/vault/pixels/${imageId}?offset=${offset}&count=${count}`),

  vaultTamper: (imageId: string, pixels: number) =>
    post<VaultTamperData>('/vault/tamper', { imageId, pixels }),

  vaultImageUrl: (imageId: string) => `${BASE}/vault/image/${imageId}`,
  vaultPreviewUrl: (imageId: string) => `${BASE}/vault/preview/${imageId}`,
  vaultFileUrl: (fileId: string) => `${BASE}/vault/file/${fileId}`,

  /** Fetch a stego PNG back as a File, so a just-encoded image can be fed
   *  straight into the decode path without a manual download and re-upload. */
  vaultImageAsFile: async (imageId: string, name: string): Promise<File> => {
    const res = await fetch(`${BASE}/vault/image/${imageId}`)
    if (!res.ok) throw new ApiError('Could not read the encoded image', res.status)
    return new File([await res.blob()], name, { type: 'image/png' })
  },
}

export { ApiError }
