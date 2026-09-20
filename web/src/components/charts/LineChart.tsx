import { useCallback, useState } from 'react'
import { drawAxes, drawCrosshair, makeScales, niceTicks, type ChartTheme } from '../../lib/chart'
import { ChartCanvas } from './ChartCanvas'

export interface Series {
  x: number[]
  y: number[]
  /** Theme key or literal colour. */
  color: 'time' | 'freq' | 'detect' | 'accent' | 'muted'
  label: string
  fill?: boolean
  /** Where the fill closes: the zero line (magnitudes) or the floor (dB curves). */
  fillTo?: 'zero' | 'bottom'
  dashed?: boolean
  /** Draw discrete stems + dots instead of a continuous line. */
  stems?: boolean
  width?: number
}

/** Loop-based extent — spreading a large array into Math.min blows the stack. */
function extent(values: number[][]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const arr of values) {
    for (const v of arr) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : [0, 1]
}

export interface LineChartProps {
  series: Series[]
  height?: number
  xLabel?: string
  yLabel?: string
  xRange?: [number, number]
  yRange?: [number, number]
  logX?: boolean
  formatX?: (v: number) => string
  formatY?: (v: number) => string
  /** Vertical reference lines, e.g. filter cutoffs. */
  markers?: { x: number; label?: string }[]
  label: string
}

function resolveColor(theme: ChartTheme, key: Series['color']): string {
  switch (key) {
    case 'freq':
      return theme.freq
    case 'detect':
      return theme.detect
    case 'accent':
      return theme.accent
    case 'muted':
      return theme.axis
    default:
      return theme.time
  }
}

export function LineChart({
  series,
  height = 220,
  xLabel,
  yLabel,
  xRange,
  yRange,
  logX = false,
  formatX,
  formatY,
  markers,
  label,
}: LineChartProps) {
  const [hover, setHover] = useState<{ x: number; values: { label: string; color: string; value: number }[]; xValue: number } | null>(null)

  const [dataXMin, dataXMax] = extent(series.map((s) => s.x))
  const [dataYMin, dataYMax] = extent(series.map((s) => s.y))
  const xMin = xRange?.[0] ?? Math.min(dataXMin, 0)
  const xMaxRaw = xRange?.[1] ?? (dataXMax > xMin ? dataXMax : xMin + 1)
  const yMinRaw = yRange?.[0] ?? Math.min(dataYMin, 0)
  const yMaxRaw = yRange?.[1] ?? Math.max(dataYMax, yMinRaw + 1e-6)
  const pad = (yMaxRaw - yMinRaw) * 0.08 || 0.05
  const yMin = yRange ? yMinRaw : yMinRaw - (yMinRaw < 0 ? pad : 0)
  const yMax = yRange ? yMaxRaw : yMaxRaw + pad

  // Log scale is applied by transforming the domain, keeping one linear
  // drawing path for both modes.
  const tx = (v: number) => (logX ? Math.log10(Math.max(v, 1)) : v)
  const txMin = tx(Math.max(xMin, logX ? 10 : xMin))
  const txMax = tx(xMaxRaw)

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, h: number, theme: ChartTheme) => {
      const rawTicks = logX
        ? [10, 100, 1000, 10000, 100000].filter((v) => tx(v) >= txMin && tx(v) <= txMax)
        : niceTicks(xMin, xMaxRaw, Math.max(3, Math.floor(width / 120)))
      const box = drawAxes(ctx, width, h, theme, {
        xTicks: rawTicks.map(tx),
        yTicks: niceTicks(yMin, yMax, 4),
        xRange: [txMin, txMax],
        yRange: [yMin, yMax],
        formatX: (v) => {
          const real = logX ? Math.pow(10, v) : v
          return formatX ? formatX(real) : String(Math.round(real))
        },
        formatY: formatY ?? ((v) => (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(2))),
        xLabel,
        yLabel,
      })
      const { sx, sy } = makeScales(box, [txMin, txMax], [yMin, yMax])

      ctx.save()
      ctx.beginPath()
      ctx.rect(box.left, box.top, box.width, box.height)
      ctx.clip()

      for (const s of series) {
        const color = resolveColor(theme, s.color)

        if (s.stems) {
          ctx.strokeStyle = color
          ctx.lineWidth = 1.2
          ctx.fillStyle = color
          for (let i = 0; i < s.x.length; i++) {
            const px = sx(tx(s.x[i]))
            ctx.beginPath()
            ctx.moveTo(px, sy(0))
            ctx.lineTo(px, sy(s.y[i]))
            ctx.stroke()
            ctx.beginPath()
            ctx.arc(px, sy(s.y[i]), 2.6, 0, Math.PI * 2)
            ctx.fill()
          }
          continue
        }

        if (s.fill) {
          // Close to the floor for dB curves, where "below the line" is the
          // attenuated region; to zero for magnitudes.
          const baseValue = s.fillTo === 'bottom' ? yMin : Math.max(yMin, 0)
          const gradient = ctx.createLinearGradient(0, box.top, 0, box.top + box.height)
          gradient.addColorStop(0, `${color}55`)
          gradient.addColorStop(1, `${color}05`)
          ctx.beginPath()
          ctx.moveTo(sx(tx(s.x[0])), sy(baseValue))
          for (let i = 0; i < s.x.length; i++) ctx.lineTo(sx(tx(s.x[i])), sy(s.y[i]))
          ctx.lineTo(sx(tx(s.x[s.x.length - 1])), sy(baseValue))
          ctx.closePath()
          ctx.fillStyle = gradient
          ctx.fill()
        }

        ctx.strokeStyle = color
        ctx.lineWidth = s.width ?? 1.6
        ctx.setLineDash(s.dashed ? [5, 4] : [])
        ctx.lineJoin = 'round'
        ctx.beginPath()
        for (let i = 0; i < s.x.length; i++) {
          const px = sx(tx(s.x[i]))
          const py = sy(s.y[i])
          if (i === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        }
        ctx.stroke()
        ctx.setLineDash([])
      }

      if (markers?.length) {
        ctx.strokeStyle = theme.accent
        ctx.setLineDash([4, 4])
        ctx.lineWidth = 1.2
        for (const m of markers) {
          const px = Math.round(sx(tx(m.x))) + 0.5
          ctx.beginPath()
          ctx.moveTo(px, box.top)
          ctx.lineTo(px, box.top + box.height)
          ctx.stroke()
          if (m.label) {
            ctx.setLineDash([])
            ctx.fillStyle = theme.accent
            ctx.font = '10px "Fira Code", monospace'
            ctx.textAlign = 'left'
            ctx.textBaseline = 'top'
            ctx.fillText(m.label, px + 4, box.top + 3)
            ctx.setLineDash([4, 4])
          }
        }
        ctx.setLineDash([])
      }

      ctx.restore()
      if (hover) drawCrosshair(ctx, box, hover.x, theme)
    },
    [series, xMin, xMaxRaw, yMin, yMax, txMin, txMax, logX, formatX, formatY, xLabel, yLabel, markers, hover],
  )

  const handleMove = useCallback(
    (x: number, _y: number, width: number) => {
      const padLeft = yLabel ? 58 : 46
      const plotWidth = width - padLeft - 14
      if (x < padLeft || x > padLeft + plotWidth) {
        setHover(null)
        return
      }
      const ratio = (x - padLeft) / plotWidth
      const txValue = txMin + ratio * (txMax - txMin)
      const xValue = logX ? Math.pow(10, txValue) : txValue
      const values = series
        .filter((s) => !s.stems)
        .map((s) => {
          let lo = 0
          let hi = s.x.length - 1
          while (lo < hi) {
            const mid = (lo + hi) >> 1
            if (s.x[mid] < xValue) lo = mid + 1
            else hi = mid
          }
          return { label: s.label, color: s.color, value: s.y[lo] ?? 0 }
        })
        .map((v) => ({ ...v, color: v.color as string }))
      setHover({ x, values, xValue })
    },
    [series, txMin, txMax, logX, yLabel],
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
          <div className="tnum mb-0.5 font-semibold text-text">
            {formatX ? formatX(hover.xValue) : hover.xValue.toFixed(2)}
          </div>
          {hover.values.map((v) => (
            <div key={v.label} className="tnum text-muted">
              {v.label}: {Math.abs(v.value) >= 1000 ? v.value.toExponential(2) : v.value.toFixed(4)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
