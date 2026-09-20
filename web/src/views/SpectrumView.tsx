import { useState } from 'react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { LineChart } from '../components/charts/LineChart'
import { Card, Legend, Slider, Stat, cx } from '../components/ui'
import { ChartSkeleton, ErrorState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

type Mode = 'whole' | 'frame'

export function SpectrumView({ signal }: { signal: SignalSummary }) {
  const [mode, setMode] = useState<Mode>('whole')
  const [logX, setLogX] = useState(false)
  const [frameMs, setFrameMs] = useState(25)
  const [center, setCenter] = useState(Math.min(0.5, signal.duration / 2))
  const [peakCount, setPeakCount] = useState(5)

  const { data, loading, error, reload } = useAsyncData(
    () =>
      api.fft(signal.id, {
        peaks: peakCount,
        frameMs: mode === 'frame' ? frameMs : 0,
        center: mode === 'frame' ? center : 0,
      }),
    [signal.id, mode, frameMs, center, peakCount],
  )

  return (
    <ViewBody>
      <ViewHeader
        title="FFT Spectrum"
        description="Magnitude spectrum of the signal. Analyse the whole signal, or window a single short frame to see how the spectrum changes locally."
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <Card
          title={mode === 'whole' ? 'Whole-signal spectrum' : `${frameMs} ms frame at ${center.toFixed(2)} s`}
          subtitle="Hamming-windowed when analysing a single frame"
          actions={
            <div
              role="group"
              aria-label="Analysis scope"
              className="flex rounded-lg border border-border bg-surface-2 p-0.5"
            >
              {(['whole', 'frame'] as Mode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  aria-pressed={mode === m}
                  className={cx(
                    'cursor-pointer rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors',
                    mode === m ? 'bg-primary text-white' : 'text-muted hover:text-text',
                  )}
                >
                  {m === 'whole' ? 'Whole signal' : 'Single frame'}
                </button>
              ))}
            </div>
          }
        >
          {loading && !data && <ChartSkeleton height={280} />}
          {error && <ErrorState message={error} onRetry={reload} />}
          {data && (
            <>
              <LineChart
                height={280}
                series={[{ x: data.x, y: data.y, color: 'freq', label: 'Magnitude', fill: true }]}
                xLabel="Frequency (Hz)"
                yLabel="Magnitude"
                logX={logX}
                xRange={logX ? undefined : [0, data.nyquist]}
                formatX={(v) => fmt.hz(v)}
                formatY={(v) => (v >= 0.01 ? v.toFixed(2) : v.toExponential(0))}
                label={`Frequency spectrum, peak at ${fmt.hzFull(data.peakFrequency)}`}
              />
              <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                <Legend items={[{ color: 'var(--series-freq)', label: 'Magnitude' }]} />
                <label className="flex cursor-pointer items-center gap-2 text-[12px] text-muted">
                  <input
                    type="checkbox"
                    checked={logX}
                    onChange={(e) => setLogX(e.target.checked)}
                    className="h-3.5 w-3.5 cursor-pointer accent-[var(--primary)]"
                  />
                  Logarithmic frequency axis
                </label>
              </div>
            </>
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Controls">
            <div className="space-y-4">
              {mode === 'frame' && (
                <>
                  <Slider
                    label="Frame length"
                    value={frameMs}
                    min={10}
                    max={40}
                    step={1}
                    onChange={setFrameMs}
                    format={(v) => `${v} ms`}
                    hint="Longer frames sharpen frequency detail but blur timing."
                  />
                  <Slider
                    label="Frame centre"
                    value={center}
                    min={0}
                    max={Math.max(0.01, signal.duration - 0.001)}
                    step={0.01}
                    onChange={setCenter}
                    format={(v) => `${v.toFixed(2)} s`}
                  />
                </>
              )}
              <Slider
                label="Peaks to list"
                value={peakCount}
                min={3}
                max={10}
                onChange={setPeakCount}
                format={(v) => String(v)}
              />
            </div>
          </Card>

          {data && (
            <Card title="Dominant frequencies" padded={false}>
              <table className="w-full text-left text-[13px]">
                <thead>
                  <tr className="border-b border-border text-[11px] uppercase tracking-[0.07em] text-faint">
                    <th scope="col" className="px-4 py-2 font-semibold">Frequency</th>
                    <th scope="col" className="px-4 py-2 text-right font-semibold">Magnitude</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {data.dominant.map((d, i) => (
                    <tr key={`${d.frequency}-${i}`} className="transition-colors hover:bg-surface-2">
                      <td className="tnum px-4 py-2 text-text">{fmt.hzFull(d.frequency)}</td>
                      <td className="tnum px-4 py-2 text-right text-muted">
                        {d.magnitude.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      </div>

      {data && (
        <div className="grid gap-3 sm:grid-cols-3">
          <Stat label="Peak frequency" value={fmt.hzFull(data.peakFrequency)} tone="accent" />
          <Stat label="Spectral energy" value={fmt.compact(data.spectralEnergy)} />
          <Stat label="Nyquist limit" value={fmt.hzFull(data.nyquist)} hint="sample rate ÷ 2" />
        </div>
      )}
    </ViewBody>
  )
}
