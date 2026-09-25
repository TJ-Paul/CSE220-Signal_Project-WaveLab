import { ArrowRight, Download } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { WaveformChart } from './charts/WaveformChart'
import { MiniPlayer } from './MiniPlayer'
import { Card } from './ui'
import { ChartSkeleton, useAsyncData } from './ViewShell'

/**
 * The output of a processing step: hear it, see it, download it, or keep
 * working on it. "Continue from here" is what makes the editing views
 * chainable — trim, then fade the trimmed result, then merge that.
 */
export function ResultCard({
  signal,
  title,
  subtitle,
  tone = 'primary',
  height = 170,
  points = 1600,
  segments,
}: {
  signal: SignalSummary
  title: string
  subtitle?: string
  tone?: 'primary' | 'accent'
  height?: number
  points?: number
  /** Shaded spans over the result, e.g. which clip each stretch came from. */
  segments?: { start: number; end: number }[]
}) {
  const { setActiveId, activeId } = useSignals()
  const wave = useAsyncData(() => api.waveform(signal.id, points), [signal.id])
  const head = useSignalPlayhead(signal.id)
  const isActive = activeId === signal.id

  return (
    <Card
      title={title}
      subtitle={subtitle ?? `${fmt.duration(signal.duration)} · ${fmt.int(signal.samples)} samples`}
      actions={
        <>
          <a
            href={api.audioUrl(signal.id)}
            download={`${signal.name}.wav`}
            className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-surface-2 px-3 text-xs font-medium text-text transition-colors hover:border-primary hover:text-primary"
          >
            <Download size={13} aria-hidden />
            WAV
          </a>
          <button
            type="button"
            disabled={isActive}
            onClick={() => setActiveId(signal.id)}
            className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-surface-2 px-3 text-xs font-medium text-text transition-colors hover:border-primary hover:text-primary disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isActive ? 'Active' : 'Continue from here'}
            {!isActive && <ArrowRight size={13} aria-hidden />}
          </button>
        </>
      }
    >
      <MiniPlayer signal={signal} tone={tone} />
      {wave.loading && <ChartSkeleton height={height} />}
      {wave.data && (
        <WaveformChart
          min={wave.data.min}
          max={wave.data.max}
          startTime={wave.data.startTime}
          endTime={wave.data.endTime}
          segments={segments}
          color={tone === 'accent' ? 'freq' : 'time'}
          height={height}
          {...head}
        />
      )}
    </Card>
  )
}
