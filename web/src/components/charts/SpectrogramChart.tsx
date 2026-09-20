import { useCallback, useMemo, useState } from 'react'
import { niceTicks, type ChartTheme } from '../../lib/chart'
import { fmt } from '../../lib/format'
import type { SpectrogramData } from '../../lib/types'
import { ChartCanvas } from './ChartCanvas'

/** Perceptually ordered magma-style ramp: dark → violet → orange → cream.
 *  Ordered lightness keeps intensity readable in greyscale too. */
const RAMP: [number, number, number][] = [
  [4, 6, 18],
  [40, 16, 76],
  [104, 24, 110],
  [168, 45, 96],
  [221, 81, 58],
  [250, 143, 40],
  [253, 205, 108],
  [252, 240, 205],
]

function rampColor(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t))
  const pos = clamped * (RAMP.length - 1)
  const i = Math.min(RAMP.length - 2, Math.floor(pos))
  const f = pos - i
  const a = RAMP[i]
  const b = RAMP[i + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ]
}

function decodeToBitmap(spec: SpectrogramData): HTMLCanvasElement | null {
  const binary = atob(spec.data)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)

  const { width, height } = spec
  const offscreen = document.createElement('canvas')
  offscreen.width = width
  offscreen.height = height
  const ctx = offscreen.getContext('2d')
  if (!ctx) return null

  const image = ctx.createImageData(width, height)
  // Source rows run low→high frequency; canvas rows run top→bottom, so the
  // vertical axis is flipped here to put high frequencies at the top.
  for (let row = 0; row < height; row++) {
    const src = (height - 1 - row) * width
    const dst = row * width * 4
    for (let col = 0; col < width; col++) {
      const [r, g, b] = rampColor(bytes[src + col] / 255)
      const o = dst + col * 4
      image.data[o] = r
      image.data[o + 1] = g
      image.data[o + 2] = b
      image.data[o + 3] = 255
    }
  }
  ctx.putImageData(image, 0, 0)
  return offscreen
}

export function SpectrogramChart({
  spec,
  height = 300,
  playhead,
  onSeek,
}: {
  spec: SpectrogramData
  height?: number
  playhead?: number | null
  onSeek?: (time: number) => void
}) {
  const [hover, setHover] = useState<{ x: number; time: number; freq: number; db: number } | null>(
    null,
  )
  const bitmap = useMemo(() => decodeToBitmap(spec), [spec])
  const bytes = useMemo(() => {
    const binary = atob(spec.data)
    const out = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
    return out
  }, [spec.data])

  const PAD = { left: 58, right: 74, top: 10, bottom: 40 }

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, h: number, theme: ChartTheme) => {
      const box = {
        left: PAD.left,
        top: PAD.top,
        width: Math.max(10, width - PAD.left - PAD.right),
        height: Math.max(10, h - PAD.top - PAD.bottom),
      }

      if (bitmap) {
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(bitmap, box.left, box.top, box.width, box.height)
      }
      ctx.strokeStyle = theme.border
      ctx.lineWidth = 1
      ctx.strokeRect(box.left + 0.5, box.top + 0.5, box.width, box.height)

      ctx.font = '11px "Fira Code", ui-monospace, monospace'
      ctx.fillStyle = theme.muted

      // Frequency axis (kHz)
      ctx.textAlign = 'right'
      ctx.textBaseline = 'middle'
      for (const f of niceTicks(0, spec.maxFrequency, 5)) {
        const y = box.top + box.height - (f / spec.maxFrequency) * box.height
        ctx.fillText(fmt.hz(f), box.left - 8, y)
      }

      // Time axis
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      for (const t of niceTicks(0, spec.duration, Math.max(3, Math.floor(box.width / 110)))) {
        const x = box.left + (t / spec.duration) * box.width
        if (x > box.left + box.width + 1) continue
        ctx.fillText(t.toFixed(1), x, box.top + box.height + 7)
      }

      ctx.font = '11px "Fira Sans", sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'bottom'
      ctx.fillText('Time (s)', box.left + box.width / 2, h - 4)
      ctx.save()
      ctx.translate(12, box.top + box.height / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textBaseline = 'top'
      ctx.fillText('Frequency (kHz)', 0, 0)
      ctx.restore()

      // Colour bar — a heatmap without a numeric scale can't be read
      const barX = box.left + box.width + 16
      const barW = 12
      for (let i = 0; i < box.height; i++) {
        const [r, g, b] = rampColor(1 - i / box.height)
        ctx.fillStyle = `rgb(${r},${g},${b})`
        ctx.fillRect(barX, box.top + i, barW, 1)
      }
      ctx.strokeStyle = theme.border
      ctx.strokeRect(barX + 0.5, box.top + 0.5, barW, box.height)

      ctx.font = '10px "Fira Code", ui-monospace, monospace'
      ctx.fillStyle = theme.muted
      ctx.textAlign = 'left'
      ctx.textBaseline = 'middle'
      const steps = 4
      for (let i = 0; i <= steps; i++) {
        const value = spec.dbMax - (i / steps) * (spec.dbMax - spec.dbMin)
        ctx.fillText(`${value.toFixed(0)}`, barX + barW + 5, box.top + (i / steps) * box.height)
      }
      ctx.save()
      ctx.font = '10px "Fira Sans", sans-serif'
      ctx.translate(barX + barW + 44, box.top + box.height / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText('dB', 0, 0)
      ctx.restore()

      if (hover && hover.x >= box.left && hover.x <= box.left + box.width) {
        ctx.strokeStyle = 'rgba(255,255,255,0.65)'
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.moveTo(Math.round(hover.x) + 0.5, box.top)
        ctx.lineTo(Math.round(hover.x) + 0.5, box.top + box.height)
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Playhead sits on top of the image, in white so it stays legible over
      // both the dark floor and the bright harmonics.
      if (playhead != null && spec.duration > 0 && playhead <= spec.duration) {
        const px = Math.round(box.left + (playhead / spec.duration) * box.width) + 0.5
        ctx.strokeStyle = 'rgba(255,255,255,0.92)'
        ctx.lineWidth = 1.5
        ctx.shadowColor = 'rgba(0,0,0,0.6)'
        ctx.shadowBlur = 3
        ctx.beginPath()
        ctx.moveTo(px, box.top)
        ctx.lineTo(px, box.top + box.height)
        ctx.stroke()
        ctx.shadowBlur = 0
      }
    },
    [bitmap, spec, hover, playhead],
  )

  const handleMove = useCallback(
    (x: number, y: number, width: number, h: number) => {
      const box = {
        left: PAD.left,
        top: PAD.top,
        width: width - PAD.left - PAD.right,
        height: h - PAD.top - PAD.bottom,
      }
      if (x < box.left || x > box.left + box.width || y < box.top || y > box.top + box.height) {
        setHover(null)
        return
      }
      const tRatio = (x - box.left) / box.width
      const fRatio = 1 - (y - box.top) / box.height
      const col = Math.min(spec.width - 1, Math.floor(tRatio * spec.width))
      const row = Math.min(spec.height - 1, Math.floor(fRatio * spec.height))
      const value = bytes[row * spec.width + col] ?? 0
      setHover({
        x,
        time: tRatio * spec.duration,
        freq: fRatio * spec.maxFrequency,
        db: spec.dbMin + (value / 255) * (spec.dbMax - spec.dbMin),
      })
    },
    [spec, bytes],
  )

  return (
    <div className="relative">
      <ChartCanvas
        height={height}
        draw={draw}
        onPointerMove={handleMove}
        onPointerLeave={() => setHover(null)}
        onClick={
          onSeek
            ? (x, width) => {
                const plotWidth = width - PAD.left - PAD.right
                const ratio = Math.min(1, Math.max(0, (x - PAD.left) / plotWidth))
                onSeek(ratio * spec.duration)
              }
            : undefined
        }
        label={`Spectrogram: ${spec.duration.toFixed(1)} seconds, up to ${fmt.hzFull(spec.maxFrequency)}, magnitudes from ${spec.dbMin.toFixed(0)} to ${spec.dbMax.toFixed(0)} dB`}
      />
      {hover && (
        <div
          className="pointer-events-none absolute top-2 z-10 rounded-lg border border-border bg-surface-3/95 px-2.5 py-1.5 text-[11px] shadow-lg backdrop-blur"
          style={{ left: Math.min(hover.x + 10, 9999) }}
        >
          <div className="tnum text-text">{hover.time.toFixed(2)} s</div>
          <div className="tnum text-muted">{fmt.hzFull(hover.freq)}</div>
          <div className="tnum text-accent">{hover.db.toFixed(1)} dB</div>
        </div>
      )}
    </div>
  )
}
