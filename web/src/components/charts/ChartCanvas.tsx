import { useEffect, useRef, useState } from 'react'
import { prepareCanvas, readChartTheme, useElementSize, type ChartTheme } from '../../lib/chart'

/** Re-render canvases when the theme flips, since their pixels are baked. */
export function useThemeVersion(): number {
  const [version, setVersion] = useState(0)
  useEffect(() => {
    const observer = new MutationObserver(() => setVersion((v) => v + 1))
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })
    return () => observer.disconnect()
  }, [])
  return version
}

export interface ChartCanvasProps {
  height: number
  /** Must be wrapped in useCallback by the parent — it drives redraws. */
  draw: (ctx: CanvasRenderingContext2D, width: number, height: number, theme: ChartTheme) => void
  onPointerMove?: (x: number, y: number, width: number, height: number) => void
  onPointerLeave?: () => void
  onClick?: (x: number, width: number) => void
  /** Text alternative describing what the chart shows. */
  label: string
  className?: string
}

export function ChartCanvas({
  height,
  draw,
  onPointerMove,
  onPointerLeave,
  onClick,
  label,
  className,
}: ChartCanvasProps) {
  const [containerRef, size] = useElementSize<HTMLDivElement>()
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const themeVersion = useThemeVersion()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size.width === 0) return
    const ctx = prepareCanvas(canvas, size.width, height)
    if (!ctx) return
    draw(ctx, size.width, height, readChartTheme())
  }, [draw, size.width, height, themeVersion])

  return (
    <div ref={containerRef} className={className} style={{ height }}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={label}
        className={onPointerMove || onClick ? 'cursor-crosshair' : undefined}
        onPointerMove={(e) => {
          if (!onPointerMove) return
          const rect = e.currentTarget.getBoundingClientRect()
          onPointerMove(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height)
        }}
        onPointerLeave={onPointerLeave}
        onClick={(e) => {
          if (!onClick) return
          const rect = e.currentTarget.getBoundingClientRect()
          onClick(e.clientX - rect.left, rect.width)
        }}
      />
    </div>
  )
}
