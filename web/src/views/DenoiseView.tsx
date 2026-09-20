import { useState } from 'react'
import { ArrowRight, Sparkles } from 'lucide-react'
import { api } from '../lib/api'
import type { DenoiseData, SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { SpectrogramChart } from '../components/charts/SpectrogramChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { Button, Card, Notice, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

export function DenoiseView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [noiseDurationS, setNoiseDurationS] = useState(0.5)
  const [alpha, setAlpha] = useState(2)
  const [beta, setBeta] = useState(0.05)
  const [result, setResult] = useState<DenoiseData | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const noisySpec = useAsyncData(() => api.spectrogram(signal.id), [signal.id])
  const noisyHead = useSignalPlayhead(signal.id)
  const cleanHead = useSignalPlayhead(result?.result.id ?? '')
  // Pin the denoised plot to the noisy plot's dB window, otherwise each
  // auto-scales to its own maximum and the floor drop becomes invisible.
  const cleanSpec = useAsyncData(
    () =>
      api.spectrogram(result!.result.id, 2048, 512, 70, {
        dbMin: noisySpec.data!.dbMin,
        dbMax: noisySpec.data!.dbMax,
      }),
    [result?.result.id, noisySpec.data?.dbMin, noisySpec.data?.dbMax],
    { enabled: Boolean(result && noisySpec.data) },
  )

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      const data = await api.denoise(signal.id, { noiseDurationS, alpha, beta })
      setResult(data)
      registerDerived(data.result)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <ViewBody>
      <ViewHeader
        title="Noise Reduction"
        description="Spectral subtraction: estimate the noise floor from a quiet opening segment, then subtract that profile from every frame."
        actions={
          <Button variant="primary" onClick={() => void run()} loading={running} icon={<Sparkles size={14} />}>
            Run denoise
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="Noisy input" subtitle="Broadband noise shows as a lifted floor across all bins">
            <MiniPlayer signal={signal} label="Noisy" />
            {noisySpec.loading && <ChartSkeleton height={260} />}
            {noisySpec.data && <SpectrogramChart spec={noisySpec.data} height={260} {...noisyHead} />}
          </Card>

          <Card
            title="Denoised output"
            subtitle={result ? 'Same colour scale — compare the background level' : 'Run the denoiser to see the result'}
          >
            {result && <MiniPlayer signal={result.result} label="Denoised" tone="accent" />}
            {!result && (
              <div className="flex h-[260px] items-center justify-center rounded-xl border border-dashed border-border text-[13px] text-faint">
                Awaiting a run
              </div>
            )}
            {result && cleanSpec.loading && <ChartSkeleton height={260} />}
            {result && cleanSpec.data && <SpectrogramChart spec={cleanSpec.data} height={260} {...cleanHead} />}
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="Subtraction settings">
            <div className="space-y-4">
              <Slider
                label="Noise sample"
                value={noiseDurationS}
                min={0.1}
                max={Math.min(2, Math.max(0.2, signal.duration / 2))}
                step={0.05}
                onChange={setNoiseDurationS}
                format={(v) => `${v.toFixed(2)} s`}
                hint="Taken from the start — it must contain noise only."
              />
              <Slider
                label="Strength — α"
                value={alpha}
                min={0.5}
                max={5}
                step={0.1}
                onChange={setAlpha}
                format={(v) => v.toFixed(1)}
                hint="Over-subtraction factor. Too high introduces musical noise."
              />
              <Slider
                label="Spectral floor — β"
                value={beta}
                min={0}
                max={0.3}
                step={0.01}
                onChange={setBeta}
                format={(v) => v.toFixed(2)}
                hint="Keeps a little noise rather than gating to silence."
              />
            </div>
          </Card>

          {result && (
            <div className="grid gap-3">
              {result.snrBefore != null && result.snrAfter != null ? (
                <>
                  <div className="flex items-center gap-2 rounded-xl border border-border bg-surface-2 px-3 py-2.5">
                    <div className="flex-1">
                      <div className="text-[11px] uppercase tracking-[0.07em] text-faint">SNR before</div>
                      <div className="tnum text-[19px] font-semibold text-muted">
                        {result.snrBefore.toFixed(1)} dB
                      </div>
                    </div>
                    <ArrowRight size={16} className="text-faint" aria-hidden />
                    <div className="flex-1 text-right">
                      <div className="text-[11px] uppercase tracking-[0.07em] text-faint">After</div>
                      <div className="tnum text-[19px] font-semibold text-success">
                        {result.snrAfter.toFixed(1)} dB
                      </div>
                    </div>
                  </div>
                  {result.correlationAfter != null && (
                    <Stat
                      label="Correlation with clean reference"
                      value={result.correlationAfter.toFixed(4)}
                      tone="primary"
                    />
                  )}
                </>
              ) : (
                <Notice tone="info">
                  SNR needs a clean reference, which only the synthetic noisy-tone demo carries.
                  Compare by ear, or use the Compare view for relative metrics.
                </Notice>
              )}

              <a
                href={api.audioUrl(result.result.id)}
                download={`${result.result.name}.wav`}
                className="inline-flex h-8 w-fit cursor-pointer items-center rounded-lg border border-border bg-surface-2 px-3 text-xs font-medium text-text transition-colors hover:border-primary hover:text-primary"
              >
                Download WAV
              </a>
            </div>
          )}

          {error && <Notice tone="danger">{error}</Notice>}
        </div>
      </div>
    </ViewBody>
  )
}
