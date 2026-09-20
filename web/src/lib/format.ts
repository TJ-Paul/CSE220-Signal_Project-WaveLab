export const fmt = {
  /** mm:ss.SS — stable width so it doesn't jitter while playing. */
  duration(seconds: number): string {
    const m = Math.floor(seconds / 60)
    const s = seconds - m * 60
    return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`
  },

  seconds(v: number): string {
    return `${v.toFixed(v < 1 ? 3 : 2)} s`
  },

  hz(v: number): string {
    if (v >= 1000) return `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k`
    return v.toFixed(0)
  },

  hzFull(v: number): string {
    return v >= 1000 ? `${(v / 1000).toFixed(2)} kHz` : `${v.toFixed(1)} Hz`
  },

  db(v: number): string {
    return `${v >= 0 ? '' : ''}${v.toFixed(1)} dB`
  },

  fixed(v: number, digits = 3): string {
    return v.toFixed(digits)
  },

  compact(v: number): string {
    if (v === 0) return '0'
    const abs = Math.abs(v)
    if (abs >= 1e6 || abs < 1e-4) return v.toExponential(2)
    return v.toFixed(abs < 1 ? 4 : 2)
  },

  int(v: number): string {
    return v.toLocaleString('en-US')
  },

  bytes(v: number): string {
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)} MB`
    if (v >= 1e3) return `${(v / 1e3).toFixed(0)} KB`
    return `${v} B`
  },

  percent(v: number): string {
    return `${(v * 100).toFixed(0)}%`
  },
}
