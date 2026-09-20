import { useState } from 'react'
import { AlertTriangle, Check, Play } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { ResampleData, SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { usePlayback } from '../state/PlaybackContext'
import { LineChart } from '../components/charts/LineChart'
import { Button, Card, Legend, Notice, Select, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ErrorState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

export function SamplingView({ signal }: { signal: SignalSummary | null }) {
  const [freq, setFreq] = useState(500)
  const [fs, setFs] = useState(800)
  const [duration, setDuration] = useState(0.05)

  const { data, loading, error, reload } = useAsyncData(
    () => api.samplingDemo(freq, fs, duration),
    [freq, fs, duration],
  )

  return (
    <ViewBody>
      <ViewHeader
        title="Sampling & Aliasing"
        description="Nyquist–Shannon: a signal band-limited to f_max reconstructs perfectly only when sampled at f_s ≥ 2·f_max. Below that, the true frequency folds back as a lower ghost."
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <Card
          title="Sample and reconstruct"
          subtitle="Original tone, the discrete samples taken from it, and the signal rebuilt from those samples"
        >
          {loading && !data && <ChartSkeleton height={300} />}
          {error && <ErrorState message={error} onRetry={reload} />}
          {data && (
            <>
              <LineChart
                height={300}
                series={[
                  {
                    x: data.continuous.t,
                    y: data.continuous.x,
                    color: 'muted',
                    label: 'Original tone',
                    width: 1.2,
                  },
                  {
                    x: data.reconstructed.t,
                    y: data.reconstructed.x,
                    color: 'freq',
                    label: 'Reconstructed',
                    dashed: true,
                    width: 1.8,
                  },
                  {
                    x: data.samples.t,
                    y: data.samples.x,
                    color: 'time',
                    label: 'Samples',
                    stems: true,
                  },
                ]}
                xLabel="Time (s)"
                yLabel="Amplitude"
                yRange={[-1.25, 1.25]}
                formatX={(v) => v.toFixed(3)}
                formatY={(v) => v.toFixed(1)}
                label={`Sampling demonstration at ${fs} hertz`}
              />
              <div className="mt-3">
                <Legend
                  items={[
                    { color: 'var(--axis)', label: 'Original tone' },
                    { color: 'var(--series-freq)', label: 'Reconstructed', dash: true },
                    { color: 'var(--series-time)', label: 'Samples' },
                  ]}
                />
              </div>
            </>
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Parameters">
            <div className="space-y-4">
              <Slider
                label="Tone frequency"
                value={freq}
                min={20}
                max={4000}
                step={10}
                onChange={setFreq}
                format={(v) => `${v} Hz`}
              />
              <Slider
                label="Sampling rate — Fs"
                value={fs}
                min={100}
                max={8000}
                step={50}
                onChange={setFs}
                format={(v) => `${fmt.int(v)} Hz`}
              />
              <Slider
                label="Duration"
                value={duration}
                min={0.02}
                max={0.2}
                step={0.005}
                onChange={setDuration}
                format={(v) => `${(v * 1000).toFixed(0)} ms`}
              />
            </div>
          </Card>

          {data && (
            <>
              {data.aliasing ? (
                <Notice tone="warn" title="Aliasing — Fs below 2·f_max">
                  <span className="flex items-start gap-2">
                    <AlertTriangle size={14} className="mt-0.5 shrink-0 text-accent" aria-hidden />
                    <span>
                      Nyquist sits at {fmt.hzFull(data.nyquist)} but the tone is {fmt.hzFull(freq)}.
                      After reconstruction it appears as a{' '}
                      <strong className="text-text">
                        {data.aliasedFrequency ? fmt.hzFull(data.aliasedFrequency) : '—'}
                      </strong>{' '}
                      alias, indistinguishable from a real tone at that frequency.
                    </span>
                  </span>
                </Notice>
              ) : (
                <Notice tone="success" title="No aliasing expected">
                  <span className="flex items-start gap-2">
                    <Check size={14} className="mt-0.5 shrink-0 text-success" aria-hidden />
                    <span>
                      Sampled above Nyquist ({fmt.hzFull(data.nyquist)}), so reconstruction should
                      track the original closely.
                    </span>
                  </span>
                </Notice>
              )}

              <div className="grid gap-3">
                <Stat label="Nyquist frequency" value={fmt.hzFull(data.nyquist)} tone="primary" />
                <Stat label="Reconstruction SNR" value={data.snr.toFixed(1)} unit="dB" />
                <Stat label="Correlation" value={data.correlation.toFixed(3)} />
                <Stat label="MSE" value={fmt.compact(data.mse)} />
              </div>
            </>
          )}
        </div>
      </div>

      <ResamplePanel signal={signal} />
    </ViewBody>
  )
}

/* -------------------------------------------------------------------------- */

function ResamplePanel({ signal }: { signal: SignalSummary | null }) {
  const { registerDerived } = useSignals()
  const playback = usePlayback()
  const [targetSr, setTargetSr] = useState('8000')
  const [result, setResult] = useState<ResampleData | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!signal) {
    return (
      <Card title="Apply to a real signal">
        <p className="text-[13px] text-muted">
          Load a signal from the dashboard to hear what sample-rate reduction actually does to it.
        </p>
      </Card>
    )
  }

  const options = ['2000', '4000', '8000', '11025', '16000', '22050'].filter(
    (v) => Number(v) < signal.sampleRate,
  )

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      const data = await api.resample(signal.id, Number(targetSr))
      setResult(data)
      registerDerived(data.result)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <Card
      title="Apply to the loaded signal"
      subtitle={`Downsample ${signal.name} then resample back up — the loss is what aliasing and lost bandwidth cost you`}
    >
      <div className="grid gap-4 md:grid-cols-[220px_auto_1fr] md:items-end">
        <Select
          label="Downsample to"
          value={targetSr}
          onChange={setTargetSr}
          options={options.map((v) => ({ value: v, label: `${fmt.int(Number(v))} Hz` }))}
        />
        <Button variant="primary" onClick={() => void run()} loading={running}>
          Run round trip
        </Button>
        {error && <p className="text-[13px] text-danger">{error}</p>}
      </div>

      {result && (
        <div className="mt-4 space-y-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="SNR vs original" value={result.snr.toFixed(1)} unit="dB" tone="primary" />
            <Stat label="Correlation" value={result.correlation.toFixed(3)} />
            <Stat label="MSE" value={fmt.compact(result.mse)} />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              icon={<Play size={13} />}
              onClick={() => playback.toggle(signal.id)}
            >
              Play original
            </Button>
            <Button
              size="sm"
              variant="primary"
              icon={<Play size={13} />}
              onClick={() => playback.toggle(result.result.id)}
            >
              Play round-tripped
            </Button>
            <span className="text-[12px] text-faint">
              Saved as “{result.result.name}” — available on the Compare view.
            </span>
          </div>
        </div>
      )}
    </Card>
  )
}
