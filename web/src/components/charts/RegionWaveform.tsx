import { useCallback, useRef, useState } from 'react'
import { drawAxes, makeScales, niceTicks, type ChartTheme } from '../../lib/chart'
import { fmt } from '../../lib/format'
import { ChartCanvas } from './ChartCanvas'

const PAD_LEFT = 58
const PAD_RIGHT = 14
/** How close a pointer must be to an edge, in pixels, to grab it instead
 *  of starting a new selection. Sized for a fingertip, not a mouse. */
const HANDLE_GRAB_PX = 7

export interface RegionWaveformProps {
  min: number[]
  max: number[]
  startTime: number
  endTime: number
  selection: { start: number; end: number } | null
  onSelectionChange: (selection: { start: number; end: number } | null) => void
  /** Regions drawn as "will be removed", e.g. detected silence. */
  markedRegions?: { start: number; end: number }[]
  height?: number
  playhead?: number | null
  onSeek?: (time: number) => void
}

type DragMode = 'new' | 'start' | 'end' | 'move'

/**
 * A waveform you can select on: drag to mark a region, drag its edges to
 * adjust, drag its middle to slide it. A click outside any selection moves
 * the playhead, so the chart stays a transport as well as an editor.
 */
export function RegionWaveform({
  min,
  max,
  startTime,
  endTime,
  selection,
  onSelectionChange,
  markedRegions,
  height = 220,
  playhead,
  onSeek,
}: RegionWaveformProps) {
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  // The cursor is state rather than derived from the drag ref: a ref change
  // does not re-render, so the shape would lag a frame behind the gesture.
  const [cursor, setCursor] = useState('crosshair')
  const drag = useRef<{ mode: DragMode; anchor: number; offset: number; moved: boolean } | null>(null)

  const peak = Math.max(0.02, ...max.map(Math.abs), ...min.map(Math.abs))
  const yLimit = Math.min(1, peak * 1.12)
  const span = endTime - startTime || 1

  const timeAt = useCallback(
    (x: number, width: number) => {
      const plotWidth = Math.max(1, width - PAD_LEFT - PAD_RIGHT)
      const ratio = Math.min(1, Math.max(0, (x - PAD_LEFT) / plotWidth))
      return startTime + ratio * span
    },
    [startTime, span],
  )

  const pixelsPerSecond = useCallback(
    (width: number) => Math.max(1, width - PAD_LEFT - PAD_RIGHT) / span,
    [span],
  )

  const handleDown = useCallback(
    (x: number, width: number) => {
      const time = timeAt(x, width)
      const pps = pixelsPerSecond(width)

      if (selection) {
        const startPx = (selection.start - startTime) * pps
        const endPx = (selection.end - startTime) * pps
        const pointerPx = x - PAD_LEFT
        if (Math.abs(pointerPx - startPx) <= HANDLE_GRAB_PX) {
          drag.current = { mode: 'start', anchor: selection.end, offset: 0, moved: false }
          return
        }
        if (Math.abs(pointerPx - endPx) <= HANDLE_GRAB_PX) {
          drag.current = { mode: 'end', anchor: selection.start, offset: 0, moved: false }
          return
        }
        if (time > selection.start && time < selection.end) {
          drag.current = { mode: 'move', anchor: selection.end - selection.start, offset: time - selection.start, moved: false }
          return
        }
      }
      drag.current = { mode: 'new', anchor: time, offset: 0, moved: false }
    },
    [selection, startTime, timeAt, pixelsPerSecond],
  )

  const handleMove = useCallback(
    (x: number, _y: number, width: number) => {
      const time = timeAt(x, width)
      setHoverTime(time)

      const state = drag.current
      if (!state) {
        // Not dragging: let the cursor advertise what a press would do here.
        if (selection) {
          const pps = pixelsPerSecond(width)
          const pointerPx = x - PAD_LEFT
          const nearEdge =
            Math.abs(pointerPx - (selection.start - startTime) * pps) <= HANDLE_GRAB_PX ||
            Math.abs(pointerPx - (selection.end - startTime) * pps) <= HANDLE_GRAB_PX
          if (nearEdge) setCursor('ew-resize')
          else if (time > selection.start && time < selection.end) setCursor('grab')
          else setCursor('crosshair')
        } else {
          setCursor('crosshair')
        }
        return
      }
      state.moved = true
      if (state.mode === 'move') setCursor('grabbing')

      if (state.mode === 'move') {
        const length = state.anchor
        const start = Math.min(Math.max(startTime, time - state.offset), endTime - length)
        onSelectionChange({ start, end: start + length })
        return
      }
      onSelectionChange({
        start: Math.min(state.anchor, time),
        end: Math.max(state.anchor, time),
      })
    },
    [timeAt, startTime, endTime, onSelectionChange, selection, pixelsPerSecond],
  )

  const handleUp = useCallback(
    (x: number, width: number) => {
      const state = drag.current
      drag.current = null
      setCursor('crosshair')
      if (!state) return

      // A press that never moved is a click, not a selection. Inside an
      // existing region it should not collapse it to nothing, so it only
      // seeks; outside, it clears the selection and seeks.
      if (!state.moved) {
        const time = timeAt(x, width)
        if (state.mode === 'new') onSelectionChange(null)
        onSeek?.(time)
        return
      }
      // Discard degenerate drags — a few stray pixels is not a region.
      if (state.mode !== 'move' && selection && selection.end - selection.start < span * 0.004) {
        onSelectionChange(null)
      }
    },
    [timeAt, onSeek, onSelectionChange, selection, span],
  )

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, h: number, theme: ChartTheme) => {
      const box = drawAxes(ctx, width, h, theme, {
        xTicks: niceTicks(startTime, endTime, Math.max(3, Math.floor(width / 110))),
        yTicks: niceTicks(-yLimit, yLimit, 4),
        xRange: [startTime, endTime],
        yRange: [-yLimit, yLimit],
        formatX: (v) => v.toFixed(span < 1 ? 2 : 1),
        formatY: (v) => v.toFixed(2),
        xLabel: 'Time (s)',
        yLabel: 'Amplitude',
      })
      const { sx, sy } = makeScales(box, [startTime, endTime], [-yLimit, yLimit])

      // Regions slated for removal sit behind everything, in the danger hue
      // so they never read as "selected".
      if (markedRegions?.length) {
        ctx.fillStyle = theme.freq
        ctx.globalAlpha = 0.16
        for (const region of markedRegions) {
          const x0 = sx(Math.max(region.start, startTime))
          const x1 = sx(Math.min(region.end, endTime))
          if (x1 > x0) ctx.fillRect(x0, box.top, x1 - x0, box.height)
        }
        ctx.globalAlpha = 1
      }

      // Everything outside the selection is dimmed, rather than the
      // selection being brightened — the eye reads the un-dimmed part as
      // the subject without the chart gaining a second accent colour.
      if (selection) {
        const x0 = Math.max(box.left, sx(selection.start))
        const x1 = Math.min(box.left + box.width, sx(selection.end))
        ctx.fillStyle = theme.surface
        ctx.globalAlpha = 0.62
        ctx.fillRect(box.left, box.top, x0 - box.left, box.height)
        ctx.fillRect(x1, box.top, box.left + box.width - x1, box.height)
        ctx.globalAlpha = 1
      }

      ctx.strokeStyle = theme.border
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(box.left, Math.round(sy(0)) + 0.5)
      ctx.lineTo(box.left + box.width, Math.round(sy(0)) + 0.5)
      ctx.stroke()

      const n = min.length
      if (n > 0) {
        const step = box.width / n
        const gradient = ctx.createLinearGradient(0, box.top, 0, box.top + box.height)
        gradient.addColorStop(0, `${theme.time}dd`)
        gradient.addColorStop(0.5, `${theme.time}99`)
        gradient.addColorStop(1, `${theme.time}dd`)

        ctx.beginPath()
        for (let i = 0; i < n; i++) ctx.lineTo(box.left + i * step, sy(max[i]))
        for (let i = n - 1; i >= 0; i--) ctx.lineTo(box.left + i * step, sy(min[i]))
        ctx.closePath()
        ctx.fillStyle = gradient
        ctx.fill()

        ctx.strokeStyle = theme.time
        ctx.lineWidth = 1
        ctx.beginPath()
        for (let i = 0; i < n; i++) ctx.lineTo(box.left + i * step, sy(max[i]))
        ctx.stroke()
        ctx.beginPath()
        for (let i = 0; i < n; i++) ctx.lineTo(box.left + i * step, sy(min[i]))
        ctx.stroke()
      }

      if (selection) {
        const x0 = sx(selection.start)
        const x1 = sx(selection.end)
        ctx.strokeStyle = theme.detect
        ctx.lineWidth = 2
        for (const x of [x0, x1]) {
          if (x < box.left - 1 || x > box.left + box.width + 1) continue
          ctx.beginPath()
          ctx.moveTo(Math.round(x) + 0.5, box.top)
          ctx.lineTo(Math.round(x) + 0.5, box.top + box.height)
          ctx.stroke()
          // Grip marks, so the draggable edges look draggable.
          ctx.fillStyle = theme.detect
          ctx.fillRect(Math.round(x) - 2.5, box.top, 5, 9)
          ctx.fillRect(Math.round(x) - 2.5, box.top + box.height - 9, 5, 9)
        }
      }

      if (playhead != null && playhead >= startTime && playhead <= endTime) {
        ctx.strokeStyle = theme.accent
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(Math.round(sx(playhead)) + 0.5, box.top)
        ctx.lineTo(Math.round(sx(playhead)) + 0.5, box.top + box.height)
        ctx.stroke()
      }
    },
    [min, max, startTime, endTime, yLimit, span, selection, markedRegions, playhead],
  )

  const length = selection ? selection.end - selection.start : 0

  return (
    <div className="relative">
      <ChartCanvas
        height={height}
        draw={draw}
        cursor={cursor}
        onPointerDown={handleDown}
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        onPointerLeave={() => {
          setHoverTime(null)
          if (!drag.current) setCursor('crosshair')
        }}
        label={
          selection
            ? `Waveform with a selection from ${fmt.seconds(selection.start)} to ${fmt.seconds(selection.end)}`
            : `Waveform from ${fmt.seconds(startTime)} to ${fmt.seconds(endTime)}. Drag to select a region.`
        }
      />
      {selection && (
        <div className="pointer-events-none absolute right-4 top-2 z-10 rounded-lg border border-success/40 bg-surface-3/95 px-2.5 py-1.5 text-[11px] shadow-lg backdrop-blur">
          <div className="tnum font-semibold text-success">{fmt.seconds(length)} selected</div>
          <div className="tnum text-muted">
            {selection.start.toFixed(3)} → {selection.end.toFixed(3)} s
          </div>
        </div>
      )}
      {!selection && hoverTime != null && (
        <div className="pointer-events-none absolute right-4 top-2 z-10 rounded-lg border border-border bg-surface-3/95 px-2.5 py-1.5 text-[11px] text-muted shadow-lg backdrop-blur">
          Drag to select · <span className="tnum text-text">{hoverTime.toFixed(2)} s</span>
        </div>
      )}
    </div>
  )
}
