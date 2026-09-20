import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { LineChart } from '../components/charts/LineChart'
import { WaveformChart } from '../components/charts/WaveformChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { Card, Notice, Select, Stat } from '../components/ui'
import {
  ChartSkeleton,
  EmptyState,
  ErrorState,
  ViewBody,
  ViewHeader,
  useAsyncData,
} from '../components/ViewShell'

export function CompareView({
  signals,
  activeId,
}: {
  signals: SignalSummary[]
  activeId: string | null
}) {
  const [idA, setIdA] = useState(activeId ?? signals[0]?.id ?? '')
  const [idB, setIdB] = useState(
    signals.find((s) => s.id !== (activeId ?? signals[0]?.id))?.id ?? signals[0]?.id ?? '',
  )

  // Keep both selections valid as the session's signal list changes.
  useEffect(() => {
    if (!signals.some((s) => s.id === idA)) setIdA(signals[0]?.id ?? '')
    if (!signals.some((s) => s.id === idB)) setIdB(signals[1]?.id ?? signals[0]?.id ?? '')
  }, [signals, idA, idB])

  const headA = useSignalPlayhead(idA)
  const headB = useSignalPlayhead(idB)

  const { data, loading, error, reload } = useAsyncData(
    () => api.compare(idA, idB),
    [idA, idB],
    { enabled: Boolean(idA && idB) },
  )

  if (signals.length < 2) {
    return (
      <ViewBody>
        <ViewHeader title="Compare" />
        <EmptyState
          title="Two signals are needed"
          description="Run a filter, denoise, separation or resample pass — each one is stored as a new signal you can measure against the original."
        />
      </ViewBody>
    )
  }

  const options = signals.map((s) => ({ value: s.id, label: s.name }))

  return (
    <ViewBody>
      <ViewHeader
        title="Signal Comparison"
        description="Quantify the difference between any two signals in the session. A is treated as the reference."
      />

      <Card title="Signals under comparison">
        <div className="grid gap-4 md:grid-cols-2">
          <div className="min-w-0">
            <Select label="Signal A — reference" value={idA} options={options} onChange={setIdA} />
            {data && (
              <p className="tnum mt-1.5 text-[11px] text-faint">
                {fmt.duration(data.a.duration)} · {(data.a.sampleRate / 1000).toFixed(1)} kHz
              </p>
            )}
          </div>
          <div className="min-w-0">
            <Select label="Signal B — test" value={idB} options={options} onChange={setIdB} />
            {data && (
              <p className="tnum mt-1.5 text-[11px] text-faint">
                {fmt.duration(data.b.duration)} · {(data.b.sampleRate / 1000).toFixed(1)} kHz
              </p>
            )}
          </div>
        </div>
        {data?.resampled && (
          <div className="mt-3">
            <Notice tone="info">
              Signal B was resampled to {fmt.int(data.a.sampleRate)} Hz so the two align sample-for-sample.
            </Notice>
          </div>
        )}
      </Card>

      {loading && !data && <ChartSkeleton height={200} />}
      {error && <ErrorState message={error} onRetry={reload} />}

      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <Stat label="SNR" value={data.metrics.snr.toFixed(1)} unit="dB" tone="primary" hint="higher = closer" />
            <Stat label="Correlation" value={data.metrics.correlation.toFixed(4)} hint="shape similarity" />
            <Stat label="MSE" value={fmt.compact(data.metrics.mse)} hint="lower = closer" />
            <Stat label="RMS difference" value={data.metrics.rmsDifference.toFixed(4)} />
            <Stat label="Peak difference" value={data.metrics.peakDifference.toFixed(4)} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Waveform — A" subtitle={data.a.name}>
              <MiniPlayer signal={data.a} label="A" />
              <WaveformChart
                min={data.waveformA.min}
                max={data.waveformA.max}
                startTime={0}
                endTime={data.a.duration}
                height={180}
                {...headA}
              />
            </Card>
            <Card title="Waveform — B" subtitle={data.b.name}>
              <MiniPlayer signal={data.b} label="B" tone="accent" />
              <WaveformChart
                min={data.waveformB.min}
                max={data.waveformB.max}
                startTime={0}
                endTime={data.b.duration}
                color="freq"
                height={180}
                {...headB}
              />
            </Card>
          </div>

          <Card
            title="Spectrum overlay"
            subtitle="Both spectra on one axis — where they diverge is what the processing changed"
          >
            <LineChart
              height={260}
              series={[
                { x: data.spectrumA.x, y: data.spectrumA.y, color: 'time', label: 'A' },
                { x: data.spectrumB.x, y: data.spectrumB.y, color: 'freq', label: 'B', dashed: true },
              ]}
              xLabel="Frequency (Hz)"
              yLabel="Magnitude"
              formatX={(v) => fmt.hz(v)}
              formatY={(v) => (v >= 0.01 ? v.toFixed(2) : v.toExponential(0))}
              label="Frequency spectra of signal A and signal B overlaid"
            />
            <p className="mt-3 text-[12px] text-faint">
              SNR treats A as ground truth and B's deviation from it as noise. Correlation measures
              shape similarity independent of level, in [-1, 1].
            </p>
          </Card>
        </>
      )}
    </ViewBody>
  )
}
