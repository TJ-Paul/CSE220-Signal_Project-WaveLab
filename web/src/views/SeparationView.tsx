import { useState } from 'react'
import { Scissors } from 'lucide-react'
import { api } from '../lib/api'
import type { SeparateData, SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { WaveformChart } from '../components/charts/WaveformChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { Button, Card, Notice, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

export function SeparationView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [marginBackground, setMarginBackground] = useState(2)
  const [marginForeground, setMarginForeground] = useState(10)
  const [result, setResult] = useState<SeparateData | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mixture = useAsyncData(() => api.waveform(signal.id, 1600), [signal.id])
  const mixtureHead = useSignalPlayhead(signal.id)

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      const data = await api.separate(signal.id, { marginBackground, marginForeground })
      setResult(data)
      registerDerived(data.foreground)
      registerDerived(data.background)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <ViewBody>
      <ViewHeader
        title="Vocal / Instrumental Separation"
        description="Repetition-based masking: frames that repeat across the track are treated as accompaniment, leaving the non-repetitive part as the vocal-like foreground."
        actions={
          <Button variant="primary" onClick={() => void run()} loading={running} icon={<Scissors size={14} />}>
            Run separation
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <Card title="Input mixture" subtitle={signal.name}>
          <MiniPlayer signal={signal} label="Mixture" />
          {mixture.loading && <ChartSkeleton height={190} />}
          {mixture.data && (
            <WaveformChart
              min={mixture.data.min}
              max={mixture.data.max}
              startTime={mixture.data.startTime}
              endTime={mixture.data.endTime}
              height={190}
              {...mixtureHead}
            />
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Mask aggressiveness">
            <div className="space-y-4">
              <Slider
                label="Background margin"
                value={marginBackground}
                min={1}
                max={20}
                step={0.5}
                onChange={setMarginBackground}
                format={(v) => v.toFixed(1)}
                hint="Higher commits more energy to the instrumental stream."
              />
              <Slider
                label="Foreground margin"
                value={marginForeground}
                min={1}
                max={20}
                step={0.5}
                onChange={setMarginForeground}
                format={(v) => v.toFixed(1)}
                hint="Higher makes the vocal stream more exclusive."
              />
            </div>
          </Card>

          <Notice tone="info" title="What this method can and can't do">
            It separates <em>repetitive</em> from <em>non-repetitive</em> content, not "voice" from
            "instruments". On real music expect bleed; the synthetic song demo is the clean case.
          </Notice>

          {error && <Notice tone="danger">{error}</Notice>}
        </div>
      </div>

      {result && (
        <>
          {result.truth && (
            <div className="grid gap-3 sm:grid-cols-2">
              <Stat
                label="Foreground vs true melody"
                value={result.truth.foregroundCorrelation.toFixed(3)}
                hint="correlation, ground truth available for the song demo"
                tone="primary"
              />
              <Stat
                label="Background vs true accompaniment"
                value={result.truth.backgroundCorrelation.toFixed(3)}
                hint="correlation, ground truth available for the song demo"
                tone="accent"
              />
            </div>
          )}

          <div className="grid gap-4 lg:grid-cols-2">
            <StreamCard title="Foreground — vocal-like" signal={result.foreground} color="time" />
            <StreamCard title="Background — instrumental-like" signal={result.background} color="freq" />
          </div>
        </>
      )}
    </ViewBody>
  )
}

function StreamCard({
  title,
  signal,
  color,
}: {
  title: string
  signal: SignalSummary
  color: 'time' | 'freq'
}) {
  const wave = useAsyncData(() => api.waveform(signal.id, 1600), [signal.id])
  const head = useSignalPlayhead(signal.id)

  return (
    <Card
      title={title}
      actions={
        <a
          href={api.audioUrl(signal.id)}
          download={`${signal.name}.wav`}
          className="inline-flex h-8 cursor-pointer items-center rounded-lg border border-border bg-surface-2 px-3 text-xs font-medium text-text transition-colors hover:border-primary hover:text-primary"
        >
          WAV
        </a>
      }
    >
      <MiniPlayer signal={signal} tone={color === 'freq' ? 'accent' : 'primary'} />
      {wave.loading && <ChartSkeleton height={170} />}
      {wave.data && (
        <WaveformChart
          min={wave.data.min}
          max={wave.data.max}
          startTime={wave.data.startTime}
          endTime={wave.data.endTime}
          color={color}
          height={170}
          {...head}
        />
      )}
    </Card>
  )
}
