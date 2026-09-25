import { useCallback, useState } from 'react'
import { drawAxes, drawCrosshair, makeScales, niceTicks, type ChartTheme } from '../../lib/chart'
import { ChartCanvas } from './ChartCanvas'

/**
 * Distribution of byte values 0–255.
 *
 * This is the chart that shows what encryption did. An audio container
 * has enormous structure — format headers, silence runs, the clustering
 * of PCM samples around zero — and it shows here as spikes and slopes.
 * Ciphertext has none: every byte value occurs about equally often, so
 * the plot flattens to noise around the mean. Any residual structure
 * would be a red flag.
 */
export function HistogramChart({
  counts,
  color = 'time',
  height = 170,
  label,
}: {
  counts: number[]
  color?: 'time' | 'freq' | 'detect'
  height?: number
  label: string
}) {
  const [hover, setHover] = useState<{ x: number; value: number; count: number } | null>(null)

  const total = counts.reduce((a, b) => a + b, 0) || 1
  const peak = Math.max(1, ...counts)
  const mean = total / 256

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, h: number, theme: ChartTheme) => {
      const box = drawAxes(ctx, width, h, theme, {
        xTicks: [0, 64, 128, 192, 255],
        yTicks: niceTicks(0, peak, 4),
        xRange: [0, 255],
        yRange: [0, peak * 1.08],
        formatX: (v) => String(Math.round(v)),
        formatY: (v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(0)),
        xLabel: 'Byte value',
        yLabel: 'Count',
      })
      const { sy } = makeScales(box, [0, 255], [0, peak * 1.08])
      const stroke = color === 'freq' ? theme.freq : color === 'detect' ? theme.detect : theme.time

      const barWidth = box.width / 256
      ctx.fillStyle = stroke
      ctx.globalAlpha = 0.85
      for (let v = 0; v < 256; v++) {
        const top = sy(counts[v] ?? 0)
        const bottom = sy(0)
        ctx.fillRect(box.left + v * barWidth, top, Math.max(1, barWidth - 0.4), bottom - top)
      }
      ctx.globalAlpha = 1

      // A uniform distribution would sit exactly on this line — the
      // reference that makes "flat" mean something specific.
      ctx.strokeStyle = theme.axis
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1.2
      ctx.beginPath()
      ctx.moveTo(box.left, Math.round(sy(mean)) + 0.5)
      ctx.lineTo(box.left + box.width, Math.round(sy(mean)) + 0.5)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = theme.muted
      ctx.font = '10px "Fira Code", monospace'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'bottom'
      ctx.fillText('uniform', box.left + box.width - 4, sy(mean) - 3)

      if (hover) drawCrosshair(ctx, box, hover.x, theme)
    },
    [counts, peak, mean, color, hover],
  )

  const handleMove = useCallback(
    (x: number, _y: number, width: number) => {
      const padLeft = 58
      const plotWidth = width - padLeft - 14
      if (x < padLeft || x > padLeft + plotWidth) {
        setHover(null)
        return
      }
      const value = Math.round(((x - padLeft) / plotWidth) * 255)
      setHover({ x, value, count: counts[value] ?? 0 })
    },
    [counts],
  )

  return (
    <div className="relative">
      <ChartCanvas
        height={height}
        draw={draw}
        onPointerMove={handleMove}
        onPointerLeave={() => setHover(null)}
        label={label}
      />
      {hover && (
        <div
          className="pointer-events-none absolute top-2 z-10 rounded-lg border border-border bg-surface-3/95 px-2.5 py-1.5 text-[11px] shadow-lg backdrop-blur"
          style={{ left: Math.min(hover.x + 10, 9999) }}
        >
          <div className="tnum text-text">
            0x{hover.value.toString(16).padStart(2, '0').toUpperCase()} ({hover.value})
          </div>
          <div className="tnum text-muted">{hover.count.toLocaleString('en-US')} bytes</div>
        </div>
      )}
    </div>
  )
}
