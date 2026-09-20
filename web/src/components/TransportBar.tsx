import { Menu, Pause, Play, Repeat, Trash2 } from 'lucide-react'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { usePlayback } from '../state/PlaybackContext'
import { Button, cx } from './ui'

/**
 * Persistent transport: the same controls as the dashboard player, sized
 * down to a bar, so playback behaves identically on every page.
 */
export function TransportBar({
  signals,
  active,
  onSelect,
  onClear,
  onOpenNav,
}: {
  signals: SignalSummary[]
  active: SignalSummary | null
  onSelect: (id: string) => void
  onClear: () => void
  onOpenNav: () => void
}) {
  const playback = usePlayback()
  // The bar follows the audio: whatever is loaded in the player wins over the
  // analysis selection, so a derived track you hit play on shows up here.
  const loaded = signals.find((s) => s.id === playback.signalId) ?? null
  const shown = loaded ?? active
  const isCurrent = Boolean(shown && playback.signalId === shown.id)
  const playing = isCurrent && playback.playing
  const position = isCurrent ? playback.currentTime : 0
  const total = shown ? (isCurrent && playback.duration ? playback.duration : shown.duration) : 0
  const progress = total ? Math.min(1, position / total) : 0
  const analysingElsewhere = Boolean(loaded && active && loaded.id !== active.id)

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-surface/85 backdrop-blur-md">
      <div className="flex items-center gap-3 px-4 py-2">
        <button
          type="button"
          onClick={onOpenNav}
          aria-label="Open navigation"
          className="grid h-9 w-9 shrink-0 cursor-pointer place-items-center rounded-lg border border-border bg-surface-2 text-muted transition-colors hover:text-text lg:hidden"
        >
          <Menu size={17} aria-hidden />
        </button>

        {shown ? (
          <>
            <button
              type="button"
              onClick={() => playback.toggle(shown.id)}
              title={playing ? 'Pause' : 'Play'}
              className="grid h-9 w-9 shrink-0 cursor-pointer place-items-center rounded-full bg-text text-surface transition-transform duration-150 hover:scale-105 active:scale-95"
            >
              {playing ? (
                <Pause size={15} fill="currentColor" aria-hidden />
              ) : (
                <Play size={15} fill="currentColor" className="ml-0.5" aria-hidden />
              )}
              <span className="sr-only">{playing ? 'Pause' : 'Play'}</span>
            </button>

            <div className="flex min-w-0 flex-1 items-center gap-3">
              <div className="min-w-0 max-w-[260px] shrink-0">
                <label htmlFor="active-signal" className="sr-only">
                  Loaded signal
                </label>
                <select
                  id="active-signal"
                  value={shown.id}
                  onChange={(e) => {
                    onSelect(e.target.value)
                    playback.cue(e.target.value, 0)
                  }}
                  className="w-full cursor-pointer truncate rounded-md border border-transparent bg-transparent py-0.5 text-[13px] font-semibold text-text transition-colors hover:border-border hover:bg-surface-2"
                >
                  {signals.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))}
                </select>
                {/* The player can hold a different track than the one being
                    analysed — say so rather than letting the two look the same. */}
                {analysingElsewhere && active && (
                  <p className="truncate text-[10px] leading-tight text-faint">
                    Analysing: {active.name}
                  </p>
                )}
              </div>

              <span className="tnum hidden shrink-0 text-[11px] text-faint sm:inline">
                {fmt.duration(position)} / {fmt.duration(total)}
              </span>

              <label htmlFor="transport-seek" className="sr-only">
                Seek position
              </label>
              <input
                id="transport-seek"
                type="range"
                min={0}
                max={total || 1}
                step={0.01}
                value={position}
                onChange={(e) => {
                  const time = Number(e.target.value)
                  if (isCurrent) playback.seek(time)
                  else playback.cue(shown.id, time)
                }}
                style={{
                  background: `linear-gradient(to right, var(--primary) ${progress * 100}%, var(--surface-3) ${progress * 100}%)`,
                }}
                className="hidden h-1.5 min-w-[80px] flex-1 md:block"
              />
            </div>

            <div className="flex shrink-0 items-center gap-1">
              <button
                type="button"
                onClick={playback.toggleLoop}
                aria-pressed={playback.loop}
                title={playback.loop ? 'Looping on' : 'Looping off'}
                className={cx(
                  'grid h-8 w-8 cursor-pointer place-items-center rounded-full transition-colors',
                  playback.loop
                    ? 'bg-primary-soft text-primary'
                    : 'text-muted hover:bg-surface-2 hover:text-text',
                )}
              >
                <Repeat size={15} aria-hidden />
                <span className="sr-only">
                  {playback.loop ? 'Turn looping off' : 'Turn looping on'}
                </span>
              </button>

              <button
                type="button"
                onClick={playback.cycleRate}
                title="Playback speed"
                className={cx(
                  'tnum grid h-8 min-w-8 cursor-pointer place-items-center rounded-full border px-1.5 text-[11px] font-bold transition-colors',
                  playback.rate === 1
                    ? 'border-border-strong text-muted hover:border-primary hover:text-primary'
                    : 'border-primary bg-primary-soft text-primary',
                )}
              >
                {playback.rate}x
                <span className="sr-only">Playback speed, currently {playback.rate}x</span>
              </button>
            </div>
          </>
        ) : (
          <div className="flex-1 text-[13px] text-muted">No signal loaded</div>
        )}

        {signals.length > 0 && (
          <Button
            size="sm"
            variant="ghost"
            icon={<Trash2 size={14} />}
            onClick={onClear}
            title="Remove every signal from this session"
            className="shrink-0"
          >
            <span className="hidden sm:inline">Clear</span>
          </Button>
        )}
      </div>

      {/* Playback position stays visible even when the slider is hidden */}
      <div className="h-0.5 w-full bg-surface-3 md:hidden" aria-hidden>
        <div className="h-full bg-primary transition-[width]" style={{ width: `${progress * 100}%` }} />
      </div>
    </header>
  )
}
