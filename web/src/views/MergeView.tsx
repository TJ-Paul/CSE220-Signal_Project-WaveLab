import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Combine, Plus, X } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { MergeData, SignalSummary } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { ResultCard } from '../components/ResultCard'
import { Button, Card, Legend, Notice, Select, Slider, Stat, Toggle, cx } from '../components/ui'
import { EmptyState, ViewBody, ViewHeader } from '../components/ViewShell'

export function MergeView({ signals, activeId }: { signals: SignalSummary[]; activeId: string | null }) {
  const { registerDerived } = useSignals()
  const [order, setOrder] = useState<string[]>(() => (activeId ? [activeId] : []))
  const [crossfade, setCrossfade] = useState(0)
  const [gap, setGap] = useState(0)
  const [law, setLaw] = useState('equal_power')
  const [normalize, setNormalize] = useState(true)
  const [result, setResult] = useState<MergeData | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const byId = useMemo(() => new Map(signals.map((s) => [s.id, s])), [signals])
  const chosen = order.map((id) => byId.get(id)).filter((s): s is SignalSummary => Boolean(s))
  const available = signals.filter((s) => !order.includes(s.id))

  const move = (index: number, delta: number) => {
    const next = [...order]
    const target = index + delta
    if (target < 0 || target >= next.length) return
    ;[next[index], next[target]] = [next[target], next[index]]
    setOrder(next)
  }

  const totalSource = chosen.reduce((sum, s) => sum + s.duration, 0)
  const seams = Math.max(0, chosen.length - 1)
  // A crossfade consumes material from both sides; a gap adds silence.
  const projected = gap > 0
    ? totalSource + seams * gap
    : Math.max(0, totalSource - seams * crossfade)

  const mixedRates = new Set(chosen.map((s) => s.sampleRate)).size > 1

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      const data = await api.merge({
        ids: order,
        crossfadeS: crossfade,
        gapS: gap,
        law,
        normalize,
      })
      registerDerived(data.result)
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
        title="Merge"
        description="Join signals end to end in the order you choose, with a crossfade or a gap at every seam."
        actions={
          <Button
            variant="primary"
            disabled={order.length < 2}
            loading={running}
            onClick={() => void run()}
            icon={<Combine size={14} />}
          >
            Merge {order.length || ''} {order.length === 1 ? 'clip' : 'clips'}
          </Button>
        }
      />

      {signals.length < 2 ? (
        <EmptyState
          title="Merging needs at least two signals"
          description="Load another file or demo signal, or produce one from any processing view — every result stays in the session."
        />
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <div className="space-y-4">
            <Card title="Merge order" subtitle="Clips are joined top to bottom">
              {chosen.length === 0 ? (
                <p className="py-6 text-center text-[13px] text-muted">
                  Add clips from the list on the right.
                </p>
              ) : (
                <ol className="space-y-1.5">
                  {chosen.map((clip, index) => (
                    <li
                      key={clip.id}
                      className="flex items-center gap-2 rounded-xl border border-border bg-surface-2 px-2.5 py-2"
                    >
                      <span className="tnum grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-primary-soft text-[11px] font-semibold text-primary">
                        {index + 1}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-[13px] text-text" title={clip.name}>
                        {clip.name}
                      </span>
                      <span className="tnum shrink-0 text-[11px] text-faint">
                        {fmt.duration(clip.duration)}
                      </span>
                      <span className="tnum shrink-0 text-[11px] text-faint">
                        {(clip.sampleRate / 1000).toFixed(1)}k
                      </span>
                      <div className="flex shrink-0 items-center">
                        <IconButton
                          label={`Move ${clip.name} up`}
                          disabled={index === 0}
                          onClick={() => move(index, -1)}
                        >
                          <ArrowUp size={13} aria-hidden />
                        </IconButton>
                        <IconButton
                          label={`Move ${clip.name} down`}
                          disabled={index === chosen.length - 1}
                          onClick={() => move(index, 1)}
                        >
                          <ArrowDown size={13} aria-hidden />
                        </IconButton>
                        <IconButton
                          label={`Remove ${clip.name}`}
                          onClick={() => setOrder(order.filter((id) => id !== clip.id))}
                        >
                          <X size={13} aria-hidden />
                        </IconButton>
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </Card>

            {available.length > 0 && (
              <Card title="Available signals">
                <div className="flex flex-wrap gap-2">
                  {available.map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => setOrder([...order, s.id])}
                      title={`Add ${s.name}`}
                      className={cx(
                        'inline-flex max-w-full cursor-pointer items-center gap-1.5 rounded-full border border-border',
                        'bg-surface px-3 py-1.5 text-[12px] font-medium text-muted transition-colors',
                        'hover:border-primary hover:bg-primary-soft hover:text-primary',
                      )}
                    >
                      <Plus size={12} aria-hidden className="shrink-0" />
                      <span className="truncate">{s.name}</span>
                      <span className="tnum shrink-0 text-faint">{fmt.duration(s.duration)}</span>
                    </button>
                  ))}
                </div>
              </Card>
            )}
          </div>

          <div className="space-y-4">
            <Card title="Seam treatment">
              <div className="space-y-4">
                <Slider
                  label="Crossfade"
                  value={crossfade}
                  min={0}
                  max={2}
                  step={0.05}
                  onChange={(v) => {
                    setCrossfade(v)
                    if (v > 0) setGap(0)
                  }}
                  format={(v) => (v === 0 ? 'off' : `${v.toFixed(2)} s`)}
                  hint="Neighbours overlap and fade through each other."
                />
                <Slider
                  label="Gap"
                  value={gap}
                  min={0}
                  max={2}
                  step={0.05}
                  onChange={(v) => {
                    setGap(v)
                    if (v > 0) setCrossfade(0)
                  }}
                  format={(v) => (v === 0 ? 'off' : `${v.toFixed(2)} s`)}
                  hint="Silence between clips. Excludes a crossfade."
                />
                <Select
                  label="Crossfade law"
                  value={law}
                  options={[
                    { value: 'equal_power', label: 'Equal power (different sources)' },
                    { value: 'linear', label: 'Linear (same source)' },
                  ]}
                  onChange={setLaw}
                />
                <Toggle
                  label="Normalise the result"
                  checked={normalize}
                  onChange={setNormalize}
                  hint="Lift the peak to −1 dBFS after joining"
                />
              </div>
            </Card>

            <Notice tone="info" title="Which law to pick">
              Unrelated clips add <em>incoherently</em> — their powers add, not their amplitudes —
              so a linear fade between them dips about 3 dB in the middle. Equal-power gains hold
              g<sub>out</sub>² + g<sub>in</sub>² = 1 and keep the level steady. Two pieces of the
              same take are coherent, and there linear is the correct choice.
            </Notice>

            {mixedRates && (
              <Notice tone="warn" title="Mixed sample rates">
                Clips will be resampled to {(chosen[0]?.sampleRate ?? 0) / 1000} kHz before joining —
                concatenating buffers of different rates would play the later ones at the wrong speed
                and pitch.
              </Notice>
            )}

            {error && <Notice tone="danger">{error}</Notice>}
          </div>
        </div>
      )}

      {chosen.length >= 2 && (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="Clips" value={String(chosen.length)} />
          <Stat label="Seams" value={String(seams)} hint={gap > 0 ? 'gapped' : crossfade > 0 ? 'crossfaded' : 'butt-joined'} />
          <Stat label="Source total" value={fmt.duration(totalSource)} />
          <Stat label="Projected length" value={fmt.duration(projected)} tone="primary" />
        </div>
      )}

      {result && (
        <>
          <MergeResult data={result} />
          {result.clips.some((c) => c.resampled) && (
            <Notice tone="info" title="Resampled on merge">
              {result.clips
                .filter((c) => c.resampled)
                .map((c) => `${c.name} (${(c.sourceSampleRate / 1000).toFixed(1)} kHz)`)
                .join(', ')}{' '}
              → {(result.sampleRate / 1000).toFixed(1)} kHz
            </Notice>
          )}
        </>
      )}
    </ViewBody>
  )
}

/** Alternating clips are shaded so every seam is visible in the output. */
function MergeResult({ data }: { data: MergeData }) {
  const edges = [0, ...data.boundaries, data.result.duration]
  const shaded = edges
    .slice(0, -1)
    .map((start, i) => ({ start, end: edges[i + 1] }))
    .filter((_, i) => i % 2 === 1)

  return (
    <>
      <ResultCard
        signal={data.result}
        title="Merged result"
        subtitle={`${data.clips.length} clips · ${fmt.duration(data.result.duration)} · ${(data.sampleRate / 1000).toFixed(1)} kHz`}
        height={200}
        segments={shaded}
      />
      {shaded.length > 0 && (
        <Legend items={[{ color: 'var(--series-detect)', label: 'Alternating clips, so seams are visible' }]} />
      )}
    </>
  )
}

function IconButton({
  children,
  label,
  onClick,
  disabled,
}: {
  children: React.ReactNode
  label: string
  onClick: () => void
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="grid h-6 w-6 cursor-pointer place-items-center rounded-md text-faint transition-colors hover:bg-surface-3 hover:text-text disabled:cursor-not-allowed disabled:opacity-30"
    >
      {children}
    </button>
  )
}
