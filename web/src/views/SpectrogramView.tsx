import { useState } from 'react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { SpectrogramChart } from '../components/charts/SpectrogramChart'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { Card, Notice, Select, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ErrorState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

const FFT_SIZES = ['256', '512', '1024', '2048', '4096'] as const
const HOPS = ['64', '128', '256', '512', '1024'] as const

export function SpectrogramView({ signal }: { signal: SignalSummary }) {
  const [nFft, setNFft] = useState<(typeof FFT_SIZES)[number]>('2048')
  const [hop, setHop] = useState<(typeof HOPS)[number]>('512')
  const [rangeDb, setRangeDb] = useState(60)
  const head = useSignalPlayhead(signal.id)

  const { data, loading, error, reload } = useAsyncData(
    () => api.spectrogram(signal.id, Number(nFft), Number(hop), rangeDb),
    [signal.id, nFft, hop, rangeDb],
  )

  return (
    <ViewBody>
      <ViewHeader
        title="Spectrogram"
        description="Short-Time Fourier Transform — how the signal's frequency content evolves over time."
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <Card title="STFT magnitude" subtitle="Hover to read time, frequency and level">
          {loading && !data && <ChartSkeleton height={360} />}
          {error && <ErrorState message={error} onRetry={reload} />}
          {data && <SpectrogramChart spec={data} height={360} {...head} />}
        </Card>

        <div className="space-y-4">
          <Card title="Transform settings">
            <div className="space-y-3.5">
              <Select
                label="FFT size — frequency resolution"
                value={nFft}
                onChange={setNFft}
                options={FFT_SIZES.map((v) => ({ value: v, label: v }))}
              />
              <Select
                label="Hop length — time resolution"
                value={hop}
                onChange={setHop}
                options={HOPS.map((v) => ({ value: v, label: v }))}
              />
              <Slider
                label="Dynamic range"
                value={rangeDb}
                min={30}
                max={100}
                step={5}
                onChange={setRangeDb}
                format={(v) => `${v} dB`}
                hint="Narrower range lifts contrast by clipping the noise floor."
              />
            </div>
          </Card>

          {data && (
            <>
              <div className="grid gap-3">
                <Stat
                  label="Frequency resolution"
                  value={data.freqResolution.toFixed(1)}
                  unit="Hz"
                  tone="primary"
                />
                <Stat label="Time resolution" value={data.timeResolution.toFixed(1)} unit="ms" />
                <Stat
                  label="Level range"
                  value={`${data.dbMin.toFixed(0)} … ${data.dbMax.toFixed(0)}`}
                  unit="dB"
                />
              </div>

              <Notice tone="info" title="Time–frequency trade-off">
                A larger FFT size resolves closer frequencies but smears events in time; a smaller
                one does the opposite. At {nFft} samples each bin spans{' '}
                {data.freqResolution.toFixed(1)} Hz and each column{' '}
                {data.timeResolution.toFixed(1)} ms.
              </Notice>
            </>
          )}
        </div>
      </div>

      {data && (
        <p className="text-[12px] text-faint">
          Showing up to {fmt.hzFull(data.maxFrequency)} over {fmt.seconds(data.duration)}. Colour
          encodes magnitude in dB — see the scale on the right of the plot.
        </p>
      )}
    </ViewBody>
  )
}
