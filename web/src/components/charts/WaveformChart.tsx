import { useCallback, useState } from 'react'
import { drawAxes, drawCrosshair, makeScales, niceTicks, type ChartTheme } from '../../lib/chart'
import { fmt } from '../../lib/format'
import { ChartCanvas } from './ChartCanvas'

export interface WaveformChartProps {
  min: number[]
  max: number[]
  startTime: number
  endTime: number
  /** Shaded regions, e.g. detected speech. */
  segments?: { start: number; end: number }[]
  segmentLabel?: string
  color?: 'time' | 'freq' | 'detect'
  height?: number
  playhead?: number | null
  onSeek?: (time: number) => void
}

export function WaveformChart({
  min,
  max,
  startTime,
  endTime,
  segments,
  color = 'time',
  height = 200,
  playhead,
  onSeek,
}: WaveformChartProps) {
  const [hover, setHover] = useState<{ x: number; time: number; amp: number } | null>(null)

  const peak = Math.max(0.02, ...max.map(Math.abs), ...min.map(Math.abs))
  const yLimit = Math.min(1, peak * 1.12)

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, h: number, theme: ChartTheme) => {
      const span = endTime - startTime || 1
      const yTicks = niceTicks(-yLimit, yLimit, 4)
      const xTicks = niceTicks(startTime, endTime, Math.max(3, Math.floor(width / 110)))
      const box = drawAxes(ctx, width, h, theme, {
        xTicks,
        yTicks,
        xRange: [startTime, endTime],
        yRange: [-yLimit, yLimit],
        formatX: (v) => `${v.toFixed(span < 1 ? 2 : 1)}`,
        formatY: (v) => v.toFixed(2),
        xLabel: 'Time (s)',
        yLabel: 'Amplitude',
      })
      const { sx, sy } = makeScales(box, [startTime, endTime], [-yLimit, yLimit])
      const stroke = color === 'freq' ? theme.freq : color === 'detect' ? theme.detect : theme.time

      // Detected regions sit behind the trace so they read as context.
      if (segments?.length) {
        ctx.fillStyle = theme.detect
        ctx.globalAlpha = 0.14
        for (const seg of segments) {
          const x0 = sx(Math.max(seg.start, startTime))
          const x1 = sx(Math.min(seg.end, endTime))
          if (x1 > x0) ctx.fillRect(x0, box.top, x1 - x0, box.height)
        }
        ctx.globalAlpha = 1
      }

      // Zero line
      ctx.strokeStyle = theme.border
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(box.left, Math.round(sy(0)) + 0.5)
      ctx.lineTo(box.left + box.width, Math.round(sy(0)) + 0.5)
      ctx.stroke()

      // Min/max envelope as a single filled band — preserves transients that
      // an averaged line would flatten.
      const n = min.length
      if (n > 0) {
        const step = box.width / n
        const gradient = ctx.createLinearGradient(0, box.top, 0, box.top + box.height)
        gradient.addColorStop(0, `${stroke}dd`)
        gradient.addColorStop(0.5, `${stroke}99`)
        gradient.addColorStop(1, `${stroke}dd`)

        ctx.beginPath()
        for (let i = 0; i < n; i++) ctx.lineTo(box.left + i * step, sy(max[i]))
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(box.left + i * step, sy(min[i]))
        ctx.closePath()
        ctx.fillStyle = gradient
        ctx.fill()

        ctx.strokeStyle = stroke
        ctx.lineWidth = 1
        ctx.beginPath()
        for (let i = 0; i < n; i++) ctx.lineTo(box.left + i * step, sy(max[i]))
        ctx.stroke()
        ctx.beginPath()
        for (let i = 0; i < n; i++) ctx.lineTo(box.left + i * step, sy(min[i]))
        ctx.stroke()
      }

      if (playhead != null && playhead >= startTime && playhead <= endTime) {
        ctx.strokeStyle = theme.accent
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(Math.round(sx(playhead)) + 0.5, box.top)
        ctx.lineTo(Math.round(sx(playhead)) + 0.5, box.top + box.height)
        ctx.stroke()
      }

      if (hover) drawCrosshair(ctx, box, hover.x, theme)
    },
    [min, max, startTime, endTime, segments, color, yLimit, hover, playhead],
  )

  const handleMove = useCallback(
    (x: number, _y: number, width: number) => {
      const padLeft = 58
      const plotWidth = width - padLeft - 14
      if (x < padLeft || x > padLeft + plotWidth) {
        setHover(null)
        return
      }
      const ratio = (x - padLeft) / plotWidth
      const time = startTime + ratio * (endTime - startTime)
      const idx = Math.min(min.length - 1, Math.max(0, Math.floor(ratio * min.length)))
      setHover({ x, time, amp: Math.max(Math.abs(min[idx] ?? 0), Math.abs(max[idx] ?? 0)) })
    },
    [startTime, endTime, min, max],
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
                const padLeft = 58
                const plotWidth = width - padLeft - 14
                const ratio = Math.min(1, Math.max(0, (x - padLeft) / plotWidth))
                onSeek(startTime + ratio * (endTime - startTime))
              }
            : undefined
        }
        label={`Waveform from ${fmt.seconds(startTime)} to ${fmt.seconds(endTime)}, peak amplitude ${peak.toFixed(3)}`}
      />
      {hover && (
        <div
          className="pointer-events-none absolute top-2 z-10 rounded-lg border border-border bg-surface-3/95 px-2.5 py-1.5 text-[11px] shadow-lg backdrop-blur"
          style={{ left: Math.min(hover.x + 10, 9999) }}
        >
          <div className="tnum text-text">{hover.time.toFixed(3)} s</div>
          <div className="tnum text-muted">±{hover.amp.toFixed(3)}</div>
        </div>
      )}
    </div>
  )
}
