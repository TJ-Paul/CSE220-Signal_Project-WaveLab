import { useEffect, useRef, useState } from 'react'

export interface Box {
  left: number
  top: number
  width: number
  height: number
}

export interface ChartTheme {
  grid: string
  axis: string
  text: string
  muted: string
  surface: string
  border: string
  time: string
  freq: string
  detect: string
  primary: string
  accent: string
}

/** Read live theme values off the CSS custom properties so canvas pixels
 *  always agree with the DOM, including after a theme switch. */
export function readChartTheme(): ChartTheme {
  const s = getComputedStyle(document.documentElement)
  const v = (name: string, fallback: string) => s.getPropertyValue(name).trim() || fallback
  return {
    grid: v('--grid', 'rgba(255,255,255,0.07)'),
    axis: v('--axis', '#56697f'),
    text: v('--text', '#e8eef6'),
    muted: v('--text-muted', '#94a7bd'),
    surface: v('--surface', '#111823'),
    border: v('--border', '#223044'),
    time: v('--series-time', '#4d94ff'),
    freq: v('--series-freq', '#f0a12e'),
    detect: v('--series-detect', '#2fd190'),
    primary: v('--primary', '#4d94ff'),
    accent: v('--accent', '#f0a12e'),
  }
}

/** Element size tracked through ResizeObserver, so charts reflow with the
 *  layout instead of being pinned to a fixed pixel width. */
export function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize({ width: Math.round(width), height: Math.round(height) })
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return [ref, size] as const
}

/** Size a canvas for the device pixel ratio so lines stay hairline-crisp. */
export function prepareCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  canvas.width = Math.max(1, Math.round(width * dpr))
  canvas.height = Math.max(1, Math.round(height * dpr))
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`
  const ctx = canvas.getContext('2d')
  if (!ctx) return null
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)
  return ctx
}

/** Human-friendly tick values (1/2/5 × 10ⁿ) covering [min, max]. */
export function niceTicks(min: number, max: number, target = 5): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min]
  const span = max - min
  const rawStep = span / target
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const norm = rawStep / magnitude
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * magnitude
  const ticks: number[] = []
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) {
    ticks.push(Math.abs(t) < step * 1e-6 ? 0 : t)
  }
  return ticks
}

export interface AxisOptions {
  xTicks: number[]
  yTicks: number[]
  xRange: [number, number]
  yRange: [number, number]
  formatX?: (v: number) => string
  formatY?: (v: number) => string
  xLabel?: string
  yLabel?: string
}

/**
 * Paint grid + axis furniture and return the plot box. Gridlines stay
 * low-contrast so the data reads first.
 */
export function drawAxes(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  theme: ChartTheme,
  opts: AxisOptions,
): Box {
  const padLeft = opts.yLabel ? 58 : 46
  const padBottom = opts.xLabel ? 40 : 26
  const box: Box = {
    left: padLeft,
    top: 10,
    width: Math.max(10, width - padLeft - 14),
    height: Math.max(10, height - padBottom - 10),
  }

  const [x0, x1] = opts.xRange
  const [y0, y1] = opts.yRange
  const sx = (v: number) => box.left + ((v - x0) / (x1 - x0 || 1)) * box.width
  const sy = (v: number) => box.top + box.height - ((v - y0) / (y1 - y0 || 1)) * box.height

  ctx.lineWidth = 1
  ctx.font = '11px "Fira Code", ui-monospace, monospace'
  ctx.strokeStyle = theme.grid
  ctx.fillStyle = theme.muted

  ctx.textAlign = 'right'
  ctx.textBaseline = 'middle'
  for (const t of opts.yTicks) {
    const y = Math.round(sy(t)) + 0.5
    if (y < box.top - 1 || y > box.top + box.height + 1) continue
    ctx.beginPath()
    ctx.moveTo(box.left, y)
    ctx.lineTo(box.left + box.width, y)
    ctx.stroke()
    ctx.fillText(opts.formatY ? opts.formatY(t) : String(t), box.left - 8, y)
  }

  ctx.textAlign = 'center'
  ctx.textBaseline = 'top'
  for (const t of opts.xTicks) {
    const x = Math.round(sx(t)) + 0.5
    if (x < box.left - 1 || x > box.left + box.width + 1) continue
    ctx.beginPath()
    ctx.moveTo(x, box.top)
    ctx.lineTo(x, box.top + box.height)
    ctx.stroke()
    ctx.fillText(opts.formatX ? opts.formatX(t) : String(t), x, box.top + box.height + 7)
  }

  ctx.strokeStyle = theme.border
  ctx.beginPath()
  ctx.moveTo(box.left, box.top + box.height + 0.5)
  ctx.lineTo(box.left + box.width, box.top + box.height + 0.5)
  ctx.stroke()

  ctx.fillStyle = theme.muted
  ctx.font = '11px "Fira Sans", sans-serif'
  if (opts.xLabel) {
    ctx.textAlign = 'center'
    ctx.textBaseline = 'bottom'
    ctx.fillText(opts.xLabel, box.left + box.width / 2, height - 4)
  }
  if (opts.yLabel) {
    ctx.save()
    ctx.translate(12, box.top + box.height / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillText(opts.yLabel, 0, 0)
    ctx.restore()
  }

  return box
}

export function makeScales(box: Box, xRange: [number, number], yRange: [number, number]) {
  const [x0, x1] = xRange
  const [y0, y1] = yRange
  return {
    sx: (v: number) => box.left + ((v - x0) / (x1 - x0 || 1)) * box.width,
    sy: (v: number) => box.top + box.height - ((v - y0) / (y1 - y0 || 1)) * box.height,
    invX: (px: number) => x0 + ((px - box.left) / (box.width || 1)) * (x1 - x0),
  }
}

/** Vertical readout line drawn on hover, shared by every line chart. */
export function drawCrosshair(
  ctx: CanvasRenderingContext2D,
  box: Box,
  x: number,
  theme: ChartTheme,
) {
  if (x < box.left || x > box.left + box.width) return
  ctx.save()
  ctx.strokeStyle = theme.axis
  ctx.setLineDash([3, 3])
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(Math.round(x) + 0.5, box.top)
  ctx.lineTo(Math.round(x) + 0.5, box.top + box.height)
  ctx.stroke()
  ctx.restore()
}
