import { useState } from 'react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { LineChart } from '../components/charts/LineChart'
import { WaveformChart } from '../components/charts/WaveformChart'
import { useSignalPlayhead } from '../state/PlaybackContext'
import { Card, Legend, Slider, Stat } from '../components/ui'
import { ChartSkeleton, ErrorState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

export function VadView({ signal }: { signal: SignalSummary }) {
  const [frameMs, setFrameMs] = useState(25)
  const [hopMs, setHopMs] = useState(10)
  const [energyPercentile, setEnergyPercentile] = useState(40)
  const [bandRatioThresh, setBandRatioThresh] = useState(0.35)

  const wave = useAsyncData(() => api.waveform(signal.id, 2000), [signal.id])
  const head = useSignalPlayhead(signal.id)
  const vad = useAsyncData(
    () => api.vad(signal.id, { frameMs, hopMs, energyPercentile, bandRatioThresh }),
    [signal.id, frameMs, hopMs, energyPercentile, bandRatioThresh],
  )

  return (
    <ViewBody>
      <ViewHeader
        title="Voice Activity Detection"
        description="Frames are classified as speech only when short-time energy and the 300–3400 Hz band ratio both clear their thresholds — one time-domain and one frequency-domain test."
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="Detected speech regions" subtitle="Shaded bands mark frames classified as speech">
            {(wave.loading || vad.loading) && !wave.data && <ChartSkeleton height={230} />}
            {wave.error && <ErrorState message={wave.error} onRetry={wave.reload} />}
            {wave.data && (
              <>
                <WaveformChart
                  min={wave.data.min}
                  max={wave.data.max}
                  startTime={wave.data.startTime}
                  endTime={wave.data.endTime}
                  segments={vad.data?.segments}
                  height={230}
                  {...head}
                />
                <div className="mt-3">
                  <Legend
                    items={[
                      { color: 'var(--series-time)', label: 'Waveform' },
                      { color: 'var(--series-detect)', label: 'Speech region' },
                    ]}
                  />
                </div>
              </>
            )}
          </Card>

          <Card title="Per-frame features" subtitle="What the classifier actually thresholds">
            {vad.loading && !vad.data && <ChartSkeleton height={230} />}
            {vad.error && <ErrorState message={vad.error} onRetry={vad.reload} />}
            {vad.data && (
              <>
                <LineChart
                  height={230}
                  series={[
                    {
                      x: vad.data.frameTimes,
                      y: vad.data.energyDb,
                      color: 'freq',
                      label: 'Short-time energy (dB)',
                    },
                  ]}
                  xLabel="Time (s)"
                  yLabel="Energy (dB)"
                  formatX={(v) => v.toFixed(1)}
                  formatY={(v) => v.toFixed(0)}
                  label="Short-time energy per frame in decibels"
                />
                <LineChart
                  height={190}
                  series={[
                    {
                      x: vad.data.frameTimes,
                      y: vad.data.bandRatio,
                      color: 'detect',
                      label: 'Speech-band ratio',
                      fill: true,
                    },
                  ]}
                  xLabel="Time (s)"
                  yLabel="Band ratio"
                  yRange={[0, 1]}
                  formatX={(v) => v.toFixed(1)}
                  formatY={(v) => v.toFixed(1)}
                  label="Fraction of frame energy inside the 300 to 3400 hertz speech band"
                />
                <p className="mt-2 text-[12px] text-faint">
                  Threshold line: frames need a band ratio above {bandRatioThresh.toFixed(2)} and
                  energy above the {energyPercentile}th percentile.
                </p>
              </>
            )}
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="Detector settings">
            <div className="space-y-4">
              <Slider
                label="Frame size"
                value={frameMs}
                min={10}
                max={40}
                onChange={setFrameMs}
                format={(v) => `${v} ms`}
              />
              <Slider
                label="Hop size"
                value={hopMs}
                min={5}
                max={30}
                onChange={setHopMs}
                format={(v) => `${v} ms`}
              />
              <Slider
                label="Energy threshold"
                value={energyPercentile}
                min={10}
                max={90}
                onChange={setEnergyPercentile}
                format={(v) => `${v}th pct`}
                hint="Higher flags more frames as speech."
              />
              <Slider
                label="Speech-band ratio"
                value={bandRatioThresh}
                min={0}
                max={1}
                step={0.01}
                onChange={setBandRatioThresh}
                format={(v) => v.toFixed(2)}
                hint="Rejects loud broadband noise that energy alone would pass."
              />
            </div>
          </Card>

          {vad.data && (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
              <Stat
                label="Speech duration"
                value={fmt.seconds(vad.data.speechDuration)}
                tone="success"
              />
              <Stat label="Silence duration" value={fmt.seconds(vad.data.silenceDuration)} />
              <Stat label="Speech ratio" value={fmt.percent(vad.data.speechRatio)} tone="primary" />
              <Stat label="Segments found" value={String(vad.data.segmentCount)} />
            </div>
          )}
        </div>
      </div>

      {vad.data && vad.data.segments.length > 0 && (
        <Card title="Segment boundaries" padded={false}>
          <div className="max-h-56 overflow-y-auto">
            <table className="w-full text-left text-[13px]">
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-border text-[11px] uppercase tracking-[0.07em] text-faint">
                  <th scope="col" className="px-4 py-2 font-semibold">#</th>
                  <th scope="col" className="px-4 py-2 font-semibold">Start</th>
                  <th scope="col" className="px-4 py-2 font-semibold">End</th>
                  <th scope="col" className="px-4 py-2 text-right font-semibold">Length</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {vad.data.segments.map((seg, i) => (
                  <tr key={`${seg.start}-${i}`} className="transition-colors hover:bg-surface-2">
                    <td className="tnum px-4 py-1.5 text-faint">{i + 1}</td>
                    <td className="tnum px-4 py-1.5 text-text">{seg.start.toFixed(3)} s</td>
                    <td className="tnum px-4 py-1.5 text-text">{seg.end.toFixed(3)} s</td>
                    <td className="tnum px-4 py-1.5 text-right text-muted">
                      {(seg.end - seg.start).toFixed(3)} s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </ViewBody>
  )
}
