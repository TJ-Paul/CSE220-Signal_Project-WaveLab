import { useState } from 'react'
import { Check, Zap } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { LineChart } from '../components/charts/LineChart'
import { WaveformChart } from '../components/charts/WaveformChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { Badge, Button, Card, Notice, Select, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ErrorState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

type Kind = 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'

export function FilterView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const nyquist = signal.sampleRate / 2

  const [kind, setKind] = useState<Kind>('lowpass')
  const [order, setOrder] = useState(6)
  const [cutoff, setCutoff] = useState(Math.min(1000, nyquist - 20))
  const [low, setLow] = useState(300)
  const [high, setHigh] = useState(Math.min(3400, nyquist - 20))
  const [applied, setApplied] = useState<SignalSummary | null>(null)
  const [applying, setApplying] = useState(false)
  const [applyError, setApplyError] = useState<string | null>(null)

  const isBand = kind === 'bandpass' || kind === 'bandstop'
  const body = isBand
    ? { kind, band: [low, high], order }
    : { kind, cutoff, order }

  const { data, loading, error, reload } = useAsyncData(
    () => api.filter(signal.id, body),
    [signal.id, kind, order, cutoff, low, high],
  )

  const appliedWave = useAsyncData(
    () => api.waveform(applied!.id, 1600),
    [applied?.id],
    { enabled: Boolean(applied) },
  )
  const originalWave = useAsyncData(() => api.waveform(signal.id, 1600), [signal.id])

  const originalHead = useSignalPlayhead(signal.id)
  const appliedHead = useSignalPlayhead(applied?.id ?? '')

  const apply = async () => {
    setApplying(true)
    setApplyError(null)
    try {
      const result = await api.filter(signal.id, { ...body, apply: true })
      if (result.result) {
        setApplied(result.result)
        registerDerived(result.result)
      }
    } catch (e) {
      setApplyError((e as Error).message)
    } finally {
      setApplying(false)
    }
  }

  return (
    <ViewBody>
      <ViewHeader
        title="Filtering"
        description="Design a Butterworth filter, inspect its magnitude response, then commit it to the signal."
        actions={
          <Button variant="primary" onClick={() => void apply()} loading={applying} icon={<Zap size={14} />}>
            Apply filter
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <Card title="Magnitude response" subtitle={data?.label ?? 'Designing…'}>
          {loading && !data && <ChartSkeleton height={290} />}
          {error && <ErrorState message={error} onRetry={reload} />}
          {data && (
            <LineChart
              height={290}
              series={[
                {
                  x: data.response.x,
                  y: data.response.y,
                  color: 'time',
                  label: 'Gain',
                  fill: true,
                  fillTo: 'bottom',
                  width: 1.8,
                },
              ]}
              xLabel="Frequency (Hz)"
              yLabel="Gain (dB)"
              xRange={[0, nyquist]}
              yRange={[Math.max(-100, Math.min(...data.response.y)), 5]}
              formatX={(v) => fmt.hz(v)}
              formatY={(v) => v.toFixed(0)}
              markers={data.cutoffs.map((c) => ({ x: c, label: `${fmt.hz(c)}Hz` }))}
              label={`${kind} filter magnitude response, order ${order}`}
            />
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Filter design">
            <div className="space-y-4">
              <Select
                label="Filter type"
                value={kind}
                onChange={(v) => setKind(v as Kind)}
                options={[
                  { value: 'lowpass', label: 'Low-pass' },
                  { value: 'highpass', label: 'High-pass' },
                  { value: 'bandpass', label: 'Band-pass' },
                  { value: 'bandstop', label: 'Band-stop' },
                ]}
              />
              {isBand ? (
                <>
                  <Slider
                    label="Low cutoff"
                    value={low}
                    min={20}
                    max={Math.max(40, high - 20)}
                    step={10}
                    onChange={setLow}
                    format={(v) => `${fmt.int(v)} Hz`}
                  />
                  <Slider
                    label="High cutoff"
                    value={high}
                    min={low + 20}
                    max={nyquist - 20}
                    step={10}
                    onChange={setHigh}
                    format={(v) => `${fmt.int(v)} Hz`}
                  />
                </>
              ) : (
                <Slider
                  label="Cutoff frequency"
                  value={cutoff}
                  min={20}
                  max={nyquist - 20}
                  step={10}
                  onChange={setCutoff}
                  format={(v) => `${fmt.int(v)} Hz`}
                />
              )}
              <Slider
                label="Filter order"
                value={order}
                min={2}
                max={12}
                step={1}
                onChange={setOrder}
                format={(v) => String(v)}
                hint="Higher order = steeper roll-off, longer ringing."
              />
            </div>
          </Card>

          <div className="grid gap-3">
            <Stat label="Nyquist limit" value={fmt.hzFull(nyquist)} />
            <Stat label="Roll-off" value={`${order * 6}`} unit="dB/oct" tone="primary" />
          </div>

          {applyError && <Notice tone="danger">{applyError}</Notice>}
        </div>
      </div>

      {applied && (
        <Card
          title="Before / after"
          subtitle="The filtered signal is stored in the session and can be compared precisely"
          actions={
            <Badge tone="success">
              <Check size={11} aria-hidden /> Applied
            </Badge>
          }
        >
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="min-w-0">
              <MiniPlayer signal={signal} label="Original" />
              {originalWave.data && (
                <WaveformChart
                  min={originalWave.data.min}
                  max={originalWave.data.max}
                  startTime={originalWave.data.startTime}
                  endTime={originalWave.data.endTime}
                  height={170}
                  {...originalHead}
                />
              )}
            </div>
            <div className="min-w-0">
              <MiniPlayer signal={applied} label="Filtered" tone="accent" />
              {appliedWave.data && (
                <WaveformChart
                  min={appliedWave.data.min}
                  max={appliedWave.data.max}
                  startTime={appliedWave.data.startTime}
                  endTime={appliedWave.data.endTime}
                  color="freq"
                  height={170}
                  {...appliedHead}
                />
              )}
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <a
              href={api.audioUrl(applied.id)}
              download={`${applied.name}.wav`}
              className="inline-flex h-8 cursor-pointer items-center rounded-lg border border-border bg-surface-2 px-3 text-xs font-medium text-text transition-colors hover:border-primary hover:text-primary"
            >
              Download WAV
            </a>
            <span className="text-[12px] text-faint">Saved as “{applied.name}”.</span>
          </div>
        </Card>
      )}
    </ViewBody>
  )
}
