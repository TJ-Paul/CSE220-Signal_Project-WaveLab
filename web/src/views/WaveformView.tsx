import { useState } from 'react'
import { RotateCcw } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { WaveformChart } from '../components/charts/WaveformChart'
import { Button, Card, Stat } from '../components/ui'
import { ChartSkeleton, ErrorState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

export function WaveformView({ signal }: { signal: SignalSummary }) {
  const head = useSignalPlayhead(signal.id)
  const [range, setRange] = useState<[number, number]>([0, signal.duration])

  // Re-clamp when the active signal changes underneath the view.
  const [lastId, setLastId] = useState(signal.id)
  if (lastId !== signal.id) {
    setLastId(signal.id)
    setRange([0, signal.duration])
  }

  const { data, loading, error, reload } = useAsyncData(
    () => api.waveform(signal.id, 2400, range[0], range[1]),
    [signal.id, range[0], range[1]],
  )

  const zoomed = range[1] - range[0] < signal.duration - 1e-6

  return (
    <ViewBody>
      <ViewHeader
        title="Waveform"
        description="Time-domain view drawn as a min/max envelope, so short transients survive zooming out."
        actions={
          zoomed && (
            <Button size="sm" icon={<RotateCcw size={14} />} onClick={() => setRange([0, signal.duration])}>
              Reset zoom
            </Button>
          )
        }
      />

      <Card
        title="Amplitude over time"
        subtitle="Click the plot to move the playhead · drag the handles below to zoom"
      >
        {loading && !data && <ChartSkeleton height={260} />}
        {error && <ErrorState message={error} onRetry={reload} />}
        {data && (
          <>
            <WaveformChart
              min={data.min}
              max={data.max}
              startTime={data.startTime}
              endTime={data.endTime}
              height={260}
              {...head}
            />

            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <RangeInput
                label="Start"
                value={range[0]}
                min={0}
                max={signal.duration}
                onChange={(v) => setRange([Math.min(v, range[1] - 0.01), range[1]])}
              />
              <RangeInput
                label="End"
                value={range[1]}
                min={0}
                max={signal.duration}
                onChange={(v) => setRange([range[0], Math.max(v, range[0] + 0.01)])}
              />
            </div>
          </>
        )}
      </Card>

      {data && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <Stat label="Peak amplitude" value={data.stats.peak.toFixed(3)} tone="primary" />
          <Stat label="RMS level" value={data.stats.rms.toFixed(3)} />
          <Stat label="Peak level" value={data.stats.peakDb.toFixed(1)} unit="dBFS" />
          <Stat label="Crest factor" value={data.stats.crestFactor.toFixed(2)} hint="peak ÷ RMS" />
          <Stat
            label="Zero crossings"
            value={fmt.int(data.stats.zeroCrossings)}
            hint={`over ${fmt.seconds(data.endTime - data.startTime)}`}
          />
        </div>
      )}
    </ViewBody>
  )
}

function RangeInput({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  onChange: (v: number) => void
}) {
  const id = `wave-range-${label.toLowerCase()}`
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-xs font-medium text-muted">
          {label}
        </label>
        <span className="tnum text-xs font-semibold text-text">{value.toFixed(3)} s</span>
      </div>
      <input
        id={id}
        type="range"
        className="mt-2"
        min={min}
        max={max}
        step={Math.max(0.001, (max - min) / 1000)}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}
