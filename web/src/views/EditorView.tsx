import { useMemo, useState } from 'react'
import { Crop, Scissors, TrendingUp } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { LineChart } from '../components/charts/LineChart'
import { RegionWaveform } from '../components/charts/RegionWaveform'
import { MiniPlayer } from '../components/MiniPlayer'
import { ResultCard } from '../components/ResultCard'
import { Button, Card, Notice, Select, Slider, Stat, Toggle } from '../components/ui'
import { ChartSkeleton, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

const SHAPE_OPTIONS = [
  { value: 'scurve', label: 'S-curve (smoothest)' },
  { value: 'linear', label: 'Linear' },
  { value: 'logarithmic', label: 'Logarithmic (fast start)' },
  { value: 'exponential', label: 'Exponential (slow start)' },
  { value: 'equal_power', label: 'Equal power' },
]

export function EditorView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [selection, setSelection] = useState<{ start: number; end: number } | null>(null)
  const [result, setResult] = useState<{ signal: SignalSummary; label: string } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [fadeIn, setFadeIn] = useState(0.5)
  const [fadeOut, setFadeOut] = useState(0.5)
  const [shape, setShape] = useState('scurve')
  const [normalize, setNormalize] = useState(false)

  const wave = useAsyncData(() => api.waveform(signal.id, 2200), [signal.id])
  const head = useSignalPlayhead(signal.id)
  const shapes = useAsyncData(() => api.fadeShapes(), [])

  const run = async (label: string, fn: () => Promise<SignalSummary>) => {
    setBusy(label)
    setError(null)
    try {
      const produced = await fn()
      registerDerived(produced)
      setResult({ signal: produced, label })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(null)
    }
  }

  const applyTrim = (mode: 'keep' | 'cut') => {
    if (!selection) return
    void run(mode === 'keep' ? 'Trimmed to selection' : 'Selection removed', async () => {
      const data = await api.trim(signal.id, { start: selection.start, end: selection.end, mode })
      return data.result
    })
  }

  const applyFade = () =>
    void run('Fades applied', async () => {
      const data = await api.fade(signal.id, {
        fadeInS: fadeIn,
        fadeOutS: fadeOut,
        shape,
        normalize,
      })
      return data.result
    })

  const fadeCurve = useMemo(() => {
    const curve = shapes.data?.curves[shape]
    if (!curve || !shapes.data) return null
    return { x: shapes.data.t, y: curve }
  }, [shapes.data, shape])

  const selectionLength = selection ? selection.end - selection.start : 0
  const maxFade = Math.max(0.05, signal.duration / 2)

  return (
    <ViewBody>
      <ViewHeader
        title="Editor"
        description="Drag across the waveform to select a region, then keep it or remove it. Every cut is crossfaded, so no edit leaves a click behind."
      />

      <Card
        title="Timeline"
        subtitle={`${signal.name} · drag to select, drag the green edges to adjust, click outside to move the playhead`}
        actions={
          <>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setSelection({ start: 0, end: signal.duration })}
            >
              Select all
            </Button>
            <Button size="sm" variant="ghost" disabled={!selection} onClick={() => setSelection(null)}>
              Clear
            </Button>
          </>
        }
      >
        <MiniPlayer signal={signal} label="Source" />
        {wave.loading && <ChartSkeleton height={260} />}
        {wave.data && (
          <RegionWaveform
            min={wave.data.min}
            max={wave.data.max}
            startTime={wave.data.startTime}
            endTime={wave.data.endTime}
            selection={selection}
            onSelectionChange={setSelection}
            height={260}
            {...head}
          />
        )}
      </Card>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Source length" value={fmt.duration(signal.duration)} />
        <Stat
          label="Selection"
          value={selection ? fmt.duration(selectionLength) : '—'}
          tone={selection ? 'success' : 'default'}
          hint={selection ? `${selection.start.toFixed(2)} s → ${selection.end.toFixed(2)} s` : 'drag on the waveform'}
        />
        <Stat
          label="If kept"
          value={selection ? fmt.duration(selectionLength) : '—'}
          hint="trim to selection"
        />
        <Stat
          label="If removed"
          value={selection ? fmt.duration(Math.max(0, signal.duration - selectionLength)) : '—'}
          hint="cut selection out"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Trim & cut" subtitle="Both actions need a selection">
          <div className="flex flex-wrap gap-2">
            <Button
              variant="primary"
              disabled={!selection}
              loading={busy === 'Trimmed to selection'}
              onClick={() => applyTrim('keep')}
              icon={<Crop size={14} />}
            >
              Keep selection
            </Button>
            <Button
              variant="danger"
              disabled={!selection}
              loading={busy === 'Selection removed'}
              onClick={() => applyTrim('cut')}
              icon={<Scissors size={14} />}
            >
              Delete selection
            </Button>
          </div>
          <Notice tone="info" title="Why a cut needs a crossfade">
            Keeping a selection makes the cut edges the new file boundaries, so nothing has to be
            joined. Deleting one splices together material that was seconds apart — an instant jump
            in amplitude, which is broadband in the frequency domain and heard as a click. A 5 ms
            crossfade at the join removes it.
          </Notice>
        </Card>

        <Card title="Fade in & out" subtitle="Applied to the start and end of the clip">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-4">
              <Slider
                label="Fade in"
                value={fadeIn}
                min={0}
                max={maxFade}
                step={0.05}
                onChange={setFadeIn}
                format={(v) => `${v.toFixed(2)} s`}
              />
              <Slider
                label="Fade out"
                value={fadeOut}
                min={0}
                max={maxFade}
                step={0.05}
                onChange={setFadeOut}
                format={(v) => `${v.toFixed(2)} s`}
              />
              <Select label="Curve" value={shape} options={SHAPE_OPTIONS} onChange={setShape} />
              <Toggle
                label="Normalise after fading"
                checked={normalize}
                onChange={setNormalize}
                hint="Lift the peak to −1 dBFS"
              />
            </div>

            <div>
              <div className="mb-1.5 text-xs font-medium text-muted">Gain curve</div>
              {shapes.loading && <ChartSkeleton height={150} />}
              {fadeCurve && (
                <LineChart
                  series={[{ ...fadeCurve, color: 'detect', label: 'gain', fill: true }]}
                  height={150}
                  xRange={[0, 1]}
                  yRange={[0, 1.02]}
                  formatX={(v) => v.toFixed(1)}
                  formatY={(v) => v.toFixed(1)}
                  label={`Gain curve for the ${shape} fade shape`}
                />
              )}
              <p className="mt-1.5 text-[11px] text-faint">
                Loudness is heard logarithmically, so a curve that rises linearly in amplitude
                sounds fast at first and slow at the end. S-curve is flat at both ends and is the
                hardest to notice.
              </p>
            </div>
          </div>

          <div className="mt-4">
            <Button
              variant="primary"
              loading={busy === 'Fades applied'}
              disabled={fadeIn === 0 && fadeOut === 0}
              onClick={applyFade}
              icon={<TrendingUp size={14} />}
            >
              Apply fades
            </Button>
          </div>
        </Card>
      </div>

      {error && <Notice tone="danger">{error}</Notice>}

      {result && (
        <ResultCard
          signal={result.signal}
          title={result.label}
          subtitle={`${result.signal.name} · ${fmt.duration(result.signal.duration)}`}
        />
      )}
    </ViewBody>
  )
}
