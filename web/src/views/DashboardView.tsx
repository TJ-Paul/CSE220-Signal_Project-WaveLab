import { useCallback, useRef, useState } from 'react'
import { Check, ChevronDown, Music3, Pause, Play, Radio, Upload, Waves } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SignalSummary, ViewId } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { usePlayback, useSignalPlayhead } from '../state/PlaybackContext'
import { WaveformChart } from '../components/charts/WaveformChart'
import { PlayerCard } from '../components/PlayerCard'
import { Badge, Button, Card, Stat, cx } from '../components/ui'
import { ChartSkeleton, EmptyState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

const DEMOS = [
  {
    kind: 'speech' as const,
    label: 'Speech-like',
    icon: Waves,
    blurb: 'Formant bursts, 6.0 s · voiced segments separated by silence — good for VAD and framing.',
  },
  {
    kind: 'song' as const,
    label: 'Song-like',
    icon: Music3,
    blurb: 'Melody + chords, 6.0 s · repetitive accompaniment — built for separation.',
  },
  {
    kind: 'noisy' as const,
    label: 'Noisy tone',
    icon: Radio,
    blurb: 'Tone + noise floor, 3.0 s · ships with a clean reference, so denoising SNR is measurable.',
  },
]

export function DashboardView({ onNavigate }: { onNavigate: (v: ViewId) => void }) {
  const { signals, active, setActiveId, loadDemo, upload, busy } = useSignals()
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (file) void upload(file)
    },
    [upload],
  )

  return (
    <ViewBody>
      <ViewHeader
        title="Signal Lab"
        description="Load a signal, then analyse, transform and compare it across the full DSP pipeline."
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)]">
        {/* Player takes the dropzone's place once a signal is loaded ----- */}
        {active ? (
          <PlayerCard signal={active} onUpload={upload} uploading={busy === 'upload'} />
        ) : (
          <Card padded={false} className="overflow-hidden">
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                handleFiles(e.dataTransfer.files)
              }}
              className={cx(
                'flex h-full flex-col items-center justify-center gap-3 px-6 py-9 text-center transition-colors duration-200',
                dragging ? 'bg-primary-soft' : 'bg-transparent',
              )}
            >
              <span
                className={cx(
                  'grid h-14 w-14 place-items-center rounded-2xl border border-dashed transition-colors duration-200',
                  dragging ? 'border-primary bg-primary-soft text-primary' : 'border-border-strong text-muted',
                )}
              >
                <Upload size={22} aria-hidden />
              </span>
              <div>
                <div className="text-[15px] font-semibold text-text">Drop an audio file</div>
                <p className="mt-1 text-[12px] text-muted">WAV or MP3 · up to 200 MB · decoded to mono float32</p>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept=".wav,.mp3,audio/wav,audio/mpeg"
                className="sr-only"
                onChange={(e) => handleFiles(e.target.files)}
              />
              <Button
                variant="primary"
                size="sm"
                loading={busy === 'upload'}
                onClick={() => inputRef.current?.click()}
                icon={<Upload size={14} />}
              >
                Browse files
              </Button>
            </div>
          </Card>
        )}

        {active ? (
          <SignalOverview signal={active} onNavigate={onNavigate} />
        ) : (
          <EmptyState
            title="No signal loaded yet"
            description="Drop a file, or load one of the demo signals below to unlock analysis and processing."
            action={
              <Button variant="primary" size="sm" onClick={() => void loadDemo('speech')}>
                Load the speech demo
              </Button>
            }
          />
        )}
      </div>

      {/* Demo signals — one compact strip ------------------------------- */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.09em] text-faint">
          Demo signals
        </span>
        {DEMOS.map((demo) => {
          const Icon = demo.icon
          const loading = busy === `demo:${demo.kind}`
          const isLoaded = active?.name.startsWith(demo.label) ?? false
          return (
            <button
              key={demo.kind}
              type="button"
              title={demo.blurb}
              disabled={loading}
              aria-pressed={isLoaded}
              onClick={() => void loadDemo(demo.kind)}
              className={cx(
                'inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-full border px-3',
                'text-[12px] font-medium transition-colors duration-150 disabled:opacity-50',
                isLoaded
                  ? 'border-accent bg-accent-soft text-accent'
                  : 'border-border bg-surface text-muted hover:border-accent hover:bg-accent-soft hover:text-accent',
              )}
            >
              <Icon size={13} aria-hidden />
              {demo.label}
            </button>
          )
        })}
      </div>

      {signals.length > 1 && (
        <SignalLibrary signals={signals} activeId={active?.id ?? null} onSelect={setActiveId} />
      )}
    </ViewBody>
  )
}

/* -------------------------------------------------------------------------- */

function SignalOverview({
  signal,
  onNavigate,
}: {
  signal: SignalSummary
  onNavigate: (v: ViewId) => void
}) {
  const head = useSignalPlayhead(signal.id)
  const { data, loading, error } = useAsyncData(
    () => api.waveform(signal.id, 1400),
    [signal.id],
  )

  return (
    <Card
      title={signal.name}
      subtitle={`${signal.origin === 'upload' ? 'Uploaded file' : signal.origin === 'demo' ? 'Demo signal' : 'Derived signal'} · ${fmt.int(signal.samples)} samples · click the waveform to move the playhead`}
      actions={
        <>
          <Badge tone="success">
            <Check size={11} aria-hidden /> Loaded
          </Badge>
          <Button size="sm" onClick={() => onNavigate('waveform')}>
            Open analysis
          </Button>
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Duration" value={fmt.duration(signal.duration)} />
        <Stat label="Sample rate" value={(signal.sampleRate / 1000).toFixed(1)} unit="kHz" />
        <Stat label="Peak level" value={signal.peakDb.toFixed(1)} unit="dBFS" tone="primary" />
        <Stat label="Crest factor" value={signal.crestFactor.toFixed(2)} hint="peak ÷ RMS" />
      </div>

      <div className="mt-4">
        {loading && <ChartSkeleton height={180} />}
        {error && <div className="text-[13px] text-danger">{error}</div>}
        {data && (
          <WaveformChart
            min={data.min}
            max={data.max}
            startTime={data.startTime}
            endTime={data.endTime}
            height={180}
            {...head}
          />
        )}
      </div>
    </Card>
  )
}

/* -------------------------------------------------------------------------- */

/** Compact, scroll-capped list — it should never dominate the dashboard. */
function SignalLibrary({
  signals,
  activeId,
  onSelect,
}: {
  signals: SignalSummary[]
  activeId: string | null
  onSelect: (id: string) => void
}) {
  const playback = usePlayback()
  const [open, setOpen] = useState(true)

  return (
    <section className="min-w-0 overflow-hidden rounded-[14px] border border-border bg-surface">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center gap-2 px-3.5 py-2.5 text-left transition-colors hover:bg-surface-2"
      >
        <ChevronDown
          size={14}
          aria-hidden
          className={cx('shrink-0 text-faint transition-transform duration-200', !open && '-rotate-90')}
        />
        <span className="text-[12px] font-semibold uppercase tracking-[0.07em] text-faint">
          Session signals
        </span>
        <span className="tnum rounded-full bg-surface-3 px-1.5 text-[11px] font-semibold text-muted">
          {signals.length}
        </span>
      </button>

      {open && (
        <ul className="max-h-[188px] overflow-y-auto border-t border-border">
          {signals.map((s) => {
            const isActive = s.id === activeId
            return (
              <li
                key={s.id}
                className={cx(
                  'group flex items-center gap-2 px-3.5 py-1.5 text-[13px] transition-colors',
                  isActive ? 'bg-primary-soft' : 'hover:bg-surface-2',
                )}
              >
                <button
                  type="button"
                  onClick={() => playback.toggle(s.id)}
                  aria-label={playback.isActive(s.id) ? `Pause ${s.name}` : `Play ${s.name}`}
                  className="grid h-6 w-6 shrink-0 cursor-pointer place-items-center rounded-full text-muted transition-colors hover:bg-surface-3 hover:text-primary"
                >
                  {playback.isActive(s.id) ? <Pause size={12} aria-hidden /> : <Play size={12} aria-hidden />}
                </button>

                <button
                  type="button"
                  onClick={() => onSelect(s.id)}
                  className={cx(
                    'min-w-0 flex-1 cursor-pointer truncate text-left',
                    isActive ? 'font-semibold text-primary' : 'text-text',
                  )}
                  title={s.name}
                >
                  {s.name}
                </button>

                <span title={s.origin} className="shrink-0">
                  <span
                    aria-hidden
                    className={cx(
                      'block h-1.5 w-1.5 rounded-full',
                      s.origin === 'derived'
                        ? 'bg-accent'
                        : s.origin === 'upload'
                          ? 'bg-primary'
                          : 'bg-faint',
                    )}
                  />
                  {/* Colour alone can't carry the origin */}
                  <span className="sr-only">{s.origin}</span>
                </span>
                <span className="tnum w-14 shrink-0 text-right text-[11px] text-faint">
                  {fmt.duration(s.duration)}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
