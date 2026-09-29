import { useState } from 'react'
import { Mic, Music4, Wand2 } from 'lucide-react'
import { api } from '../lib/api'
import type { Job, SignalSummary, VocalsData, VocalsEngine } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { WaveformChart } from '../components/charts/WaveformChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { ResultCard } from '../components/ResultCard'
import { Button, Card, Notice, SegmentedControl, Stat, Toggle } from '../components/ui'
import { ChartSkeleton, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

type Outputs = 'karaoke' | 'acapella' | 'both'

const OUTPUT_COPY: Record<Outputs, { title: string; blurb: string }> = {
  karaoke: {
    title: 'Karaoke',
    blurb: 'Keep the backing track, suppress the lead vocal.',
  },
  acapella: {
    title: 'A cappella',
    blurb: 'Keep the lead vocal, suppress the backing track.',
  },
  both: {
    title: 'Both stems',
    blurb: 'Produce the pair, for A/B listening against the mix.',
  },
}

const ENGINE_COPY: Record<VocalsEngine, string> = {
  ml: 'Pretrained Demucs (htdemucs) neural network. Has learned what a voice sounds like.',
  classical: 'Repetition-based spectral masking. No model: repeating content is treated as backing.',
}

export function VocalStudioView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [engine, setEngine] = useState<VocalsEngine>('ml')
  const [preset, setPreset] = useState('balanced')
  const [outputs, setOutputs] = useState<Outputs>('both')
  const [levelMatch, setLevelMatch] = useState(true)
  const [result, setResult] = useState<VocalsData | null>(null)
  const [running, setRunning] = useState(false)
  const [job, setJob] = useState<Job<VocalsData> | null>(null)
  const [error, setError] = useState<string | null>(null)

  const engines = useAsyncData(() => api.vocalsEngines(), [])
  const mlAvailable = engines.data?.ml ?? true

  const wave = useAsyncData(() => api.waveform(signal.id, 1800), [signal.id])
  const head = useSignalPlayhead(signal.id)

  const run = async () => {
    setRunning(true)
    setJob(null)
    setError(null)
    try {
      const data = await api.vocals(signal.id, { engine, preset, outputs, levelMatch }, setJob)
      if (data.karaoke) registerDerived(data.karaoke)
      if (data.acapella) registerDerived(data.acapella)
      setResult(data)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <ViewBody>
      <ViewHeader
        title="Karaoke & Vocals"
        description="Turn a mix into a backing track or an isolated vocal, and measure how cleanly the two came apart."
        actions={
          <Button variant="primary" loading={running} onClick={() => void run()} icon={<Wand2 size={14} />}>
            {OUTPUT_COPY[outputs].title === 'Both stems' ? 'Make both stems' : `Make ${OUTPUT_COPY[outputs].title.toLowerCase()}`}
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.2fr)_minmax(0,1fr)]">
        <Card title="Source mix" subtitle={signal.name}>
          <MiniPlayer signal={signal} label="Mix" />
          {wave.loading && <ChartSkeleton height={190} />}
          {wave.data && (
            <WaveformChart
              min={wave.data.min}
              max={wave.data.max}
              startTime={wave.data.startTime}
              endTime={wave.data.endTime}
              height={190}
              {...head}
            />
          )}
        </Card>

        <div className="space-y-4">
          <Card title="Output">
            <div className="space-y-4">
              <SegmentedControl
                label="Separation engine"
                value={engine}
                options={[
                  { value: 'ml', label: 'ML (Demucs)' },
                  { value: 'classical', label: 'Classical DSP' },
                ]}
                onChange={setEngine}
              />
              <p className="-mt-1 text-[11px] text-faint">{ENGINE_COPY[engine]}</p>
              {engine === 'ml' && !mlAvailable && (
                <Notice tone="warn">
                  Demucs is not installed. Run <code>pip install demucs</code> in the virtual
                  environment and restart the API.
                </Notice>
              )}

              <SegmentedControl
                label="What to produce"
                value={outputs}
                options={[
                  { value: 'karaoke', label: 'Karaoke' },
                  { value: 'acapella', label: 'Vocals' },
                  { value: 'both', label: 'Both' },
                ]}
                onChange={setOutputs}
              />
              <p className="-mt-1 text-[11px] text-faint">{OUTPUT_COPY[outputs].blurb}</p>

              {engine === 'classical' && (
                <>
                  <SegmentedControl
                    label="Separation strength"
                    value={preset}
                    options={[
                      { value: 'gentle', label: 'Gentle', hint: 'Least aggressive, most bleed, fewest artefacts' },
                      { value: 'balanced', label: 'Balanced' },
                      { value: 'aggressive', label: 'Aggressive', hint: 'Cleanest split on paper, most artefacts' },
                    ]}
                    onChange={setPreset}
                  />
                  <p className="-mt-1 text-[11px] text-faint">
                    Higher strength commits each time–frequency bin more decisively to one stream:
                    cleaner numbers, more audible artefacts.
                  </p>
                </>
              )}

              <Toggle
                label="Level-match the output"
                checked={levelMatch}
                onChange={setLevelMatch}
                hint="Scale stems back to the mix’s RMS so an A/B compares the processing, not the gain"
              />
            </div>
          </Card>

          {running && engine === 'ml' && job?.status === 'queued' && (
            <Notice tone="info" title="Waiting for the GPU">
              {job.ahead === 1 ? '1 job is' : `${job.ahead ?? 0} jobs are`} ahead of yours. It runs
              on the server, so you can leave this page and come back for the result.
            </Notice>
          )}
          {running && engine === 'ml' && job?.status !== 'queued' && (
            <Notice tone="info">
              Running the neural network on the server (on the Apple GPU when available). The first
              run also loads the model, and downloads it (about 85 MB) if it isn’t cached yet.
            </Notice>
          )}
          {!running && result && job?.reused && (
            <Notice tone="success">
              This song was already separated with these settings, so the saved result came straight
              back.
            </Notice>
          )}
          {error && <Notice tone="danger">{error}</Notice>}
        </div>
      </div>

      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <Stat
              label="Vocal-band suppression"
              value={result.metrics.vocalSuppressionDb.toFixed(2)}
              unit="dB"
              tone="primary"
              hint="energy the backing gives up over 300–3400 Hz"
            />
            <Stat
              label="Stem correlation"
              value={result.metrics.stemCorrelation.toFixed(3)}
              tone="success"
              hint="near 0 means the stems carry different content"
            />
            <Stat
              label="Vocal energy share"
              value={(result.metrics.vocalEnergyShare * 100).toFixed(1)}
              unit="%"
              tone="accent"
              hint="of total energy across both stems"
            />
          </div>

          {result.truth && (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Stat
                label="Vocals SI-SDR"
                value={result.truth.vocalsSdrDb.toFixed(1)}
                unit="dB"
                tone="primary"
                hint="separation quality vs the true melody, higher is better"
              />
              <Stat
                label="Backing SI-SDR"
                value={result.truth.instrumentalSdrDb.toFixed(1)}
                unit="dB"
                tone="accent"
                hint="separation quality vs the true accompaniment"
              />
              <Stat
                label="Vocals vs true melody"
                value={result.truth.vocalsCorrelation.toFixed(3)}
                tone="primary"
                hint="ground truth, available for the song demo"
              />
              <Stat
                label="Backing vs true accompaniment"
                value={result.truth.instrumentalCorrelation.toFixed(3)}
                tone="accent"
                hint="ground truth, available for the song demo"
              />
            </div>
          )}

          <div className={result.karaoke && result.acapella ? 'grid gap-4 lg:grid-cols-2' : ''}>
            {result.karaoke && (
              <ResultCard
                signal={result.karaoke}
                title="Karaoke — backing track"
                subtitle="Lead vocal suppressed"
                tone="accent"
              />
            )}
            {result.acapella && (
              <ResultCard
                signal={result.acapella}
                title="A cappella — isolated vocal"
                subtitle="Backing suppressed"
              />
            )}
          </div>

          {result.engine === 'ml' ? (
            <Notice tone="info" title="What this method can and cannot do">
              A pretrained Demucs model (Meta AI) predicts drums, bass, other and vocals from both
              the waveform and its STFT spectrogram; the backing track is drums + bass + other. We
              use it as a ready-made tool — no training happens here. It was trained on real
              singing, so it only finds <em>voices</em>: the synthetic song demo's melody is a
              generated tone, and the model correctly leaves it in the backing. Use a real song, or
              switch to Classical DSP for the demo.
            </Notice>
          ) : (
            <Notice tone="warn" title="What this method can and cannot do">
              The split is driven by <em>repetition</em>, not by any understanding of what a voice
              is: frames whose spectrum recurs elsewhere in the track are treated as backing, and
              what is left is treated as lead. That holds when the accompaniment loops and the
              vocal does not — common in pop, not universal. Expect bleed from rubato piano, sparse
              arrangements or a repeated chorus hook. The ML engine handles real songs far better.
            </Notice>
          )}
        </>
      )}

      {!result && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card title="Karaoke" subtitle="What you get">
            <div className="flex items-start gap-3">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent">
                <Music4 size={18} aria-hidden />
              </span>
              <p className="text-[13px] text-muted">
                The backing track — drums, bass, chords — with the lead vocal pulled out of it.
                Level-matched to the source so it drops straight into an A/B against the
                original.
              </p>
            </div>
          </Card>
          <Card title="A cappella" subtitle="What you get">
            <div className="flex items-start gap-3">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-primary-soft text-primary">
                <Mic size={18} aria-hidden />
              </span>
              <p className="text-[13px] text-muted">
                The complement: the isolated lead vocal. The two stems together reconstruct the
                mix.
              </p>
            </div>
          </Card>
        </div>
      )}
    </ViewBody>
  )
}
