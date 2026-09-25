import { useCallback, useEffect, useRef, useState } from 'react'
import { Eraser, Wand2 } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary, SilenceData } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { LineChart } from '../components/charts/LineChart'
import { WaveformChart } from '../components/charts/WaveformChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { ResultCard } from '../components/ResultCard'
import { Button, Card, Legend, Notice, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

export function SilenceView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [threshold, setThreshold] = useState<number | null>(null)
  const [minSilence, setMinSilence] = useState(0.35)
  const [pad, setPad] = useState(0.08)
  const [hysteresis, setHysteresis] = useState(6)
  const [analysis, setAnalysis] = useState<SilenceData | null>(null)
  const [result, setResult] = useState<SignalSummary | null>(null)
  const [removing, setRemoving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wave = useAsyncData(() => api.waveform(signal.id, 2000), [signal.id])
  const head = useSignalPlayhead(signal.id)

  // The first analysis runs with threshold=null, which asks the server to
  // choose one from the level distribution; the slider then adopts it.
  const adopted = useRef(false)
  const analyse = useCallback(
    async (thresholdDb: number | null) => {
      setError(null)
      try {
        const data = await api.silence(signal.id, {
          thresholdDb,
          minSilenceS: minSilence,
          padS: pad,
          hysteresisDb: hysteresis,
        })
        setAnalysis(data)
        if (!adopted.current) {
          adopted.current = true
          setThreshold(data.relativeThresholdDb)
        }
      } catch (e) {
        setError((e as Error).message)
      }
    },
    [signal.id, minSilence, pad, hysteresis],
  )

  useEffect(() => {
    void analyse(threshold)
  }, [analyse, threshold])

  const remove = async () => {
    setRemoving(true)
    setError(null)
    try {
      const data = await api.silence(signal.id, {
        thresholdDb: threshold,
        minSilenceS: minSilence,
        padS: pad,
        hysteresisDb: hysteresis,
        apply: true,
      })
      setAnalysis(data)
      if (data.result) {
        registerDerived(data.result)
        setResult(data.result)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRemoving(false)
    }
  }

  const reduction = analysis && analysis.totalDuration > 0
    ? analysis.removedDuration / analysis.totalDuration
    : 0

  return (
    <ViewBody>
      <ViewHeader
        title="Silence Remover"
        description="Find the dead air in a recording and close it, while leaving the natural pauses that carry meaning."
        actions={
          <Button
            variant="primary"
            loading={removing}
            disabled={!analysis || analysis.regionCount === 0}
            onClick={() => void remove()}
            icon={<Eraser size={14} />}
          >
            Remove {analysis?.regionCount ?? 0} silent {analysis?.regionCount === 1 ? 'region' : 'regions'}
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.3fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="Detected silence" subtitle="Shaded spans are what would be deleted">
            <MiniPlayer signal={signal} label="Source" />
            {wave.loading && <ChartSkeleton height={200} />}
            {wave.data && (
              <WaveformChart
                min={wave.data.min}
                max={wave.data.max}
                startTime={wave.data.startTime}
                endTime={wave.data.endTime}
                segments={analysis?.regions}
                height={200}
                {...head}
              />
            )}
            <div className="mt-2">
              <Legend items={[{ color: 'var(--series-detect)', label: 'Silence to be removed' }]} />
            </div>
          </Card>

          <Card title="Frame level and threshold" subtitle="RMS level of each 20 ms frame, in dBFS">
            {analysis ? (
              <LineChart
                series={[
                  {
                    x: analysis.levels.x,
                    y: analysis.levels.y,
                    color: 'time',
                    label: 'level',
                    fill: true,
                    fillTo: 'bottom',
                  },
                  {
                    x: [0, analysis.totalDuration],
                    y: [analysis.thresholdDb, analysis.thresholdDb],
                    color: 'freq',
                    label: 'threshold',
                    dashed: true,
                  },
                ]}
                height={190}
                xLabel="Time (s)"
                yLabel="Level (dBFS)"
                formatX={(v) => v.toFixed(1)}
                formatY={(v) => v.toFixed(0)}
                label="Frame level over time against the silence threshold"
              />
            ) : (
              <ChartSkeleton height={190} />
            )}
            <p className="mt-2 text-[11px] text-faint">
              Silence is a property of a span, not of a sample — every waveform crosses zero twice
              per cycle, so the decision is made on this frame-rate envelope rather than on the
              samples themselves.
            </p>
          </Card>
        </div>

        <div className="space-y-4">
          <Card
            title="Threshold"
            actions={
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setThreshold(analysis?.suggestedThresholdDb ?? -40)}
                icon={<Wand2 size={13} />}
              >
                Auto
              </Button>
            }
          >
            <Slider
              label="Silence below"
              value={threshold ?? -40}
              min={-70}
              max={-6}
              step={0.5}
              onChange={setThreshold}
              format={(v) => `${v.toFixed(1)} dB`}
              hint={
                analysis
                  ? `${analysis.thresholdDb.toFixed(1)} dBFS absolute · peak is ${analysis.peakDb.toFixed(1)} dBFS`
                  : 'relative to this signal’s own peak'
              }
            />
            {analysis && (
              <p className="mt-2 text-[11px] text-faint">
                Auto suggests {analysis.suggestedThresholdDb.toFixed(1)} dB, read from this
                recording’s own level distribution — a fixed offset is wrong for both a clean studio
                take and a noisy phone recording.
              </p>
            )}
          </Card>

          <Card title="Guards against over-editing">
            <div className="space-y-4">
              <Slider
                label="Minimum silence"
                value={minSilence}
                min={0.05}
                max={2}
                step={0.05}
                onChange={setMinSilence}
                format={(v) => `${v.toFixed(2)} s`}
                hint="Shorter gaps are the rhythm of speech, and are kept."
              />
              <Slider
                label="Keep padding"
                value={pad}
                min={0}
                max={0.5}
                step={0.01}
                onChange={setPad}
                format={(v) => `${(v * 1000).toFixed(0)} ms`}
                hint="Silence left at each end so words are not clipped."
              />
              <Slider
                label="Hysteresis"
                value={hysteresis}
                min={0}
                max={18}
                step={1}
                onChange={setHysteresis}
                format={(v) => `${v.toFixed(0)} dB`}
                hint="Gap between the on and off thresholds. At 0 the detector chatters."
              />
            </div>
          </Card>

          {error && <Notice tone="danger">{error}</Notice>}
          {analysis?.regionCount === 0 && (
            <Notice tone="warn" title="Nothing detected">
              No span is both quiet enough and long enough. Either this signal is continuous, or the
              threshold sits below its noise floor — raise it and watch the dashed line cross the
              level curve.
            </Notice>
          )}
        </div>
      </div>

      {analysis && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Silent regions" value={String(analysis.regionCount)} tone="primary" />
          <Stat label="Time removed" value={fmt.duration(analysis.removedDuration)} tone="accent" />
          <Stat label="New length" value={fmt.duration(analysis.keptDuration)} hint={`from ${fmt.duration(analysis.totalDuration)}`} />
          <Stat label="Reduction" value={(reduction * 100).toFixed(1)} unit="%" tone="success" />
        </div>
      )}

      {result && <ResultCard signal={result} title="Silence removed" tone="accent" />}
    </ViewBody>
  )
}
