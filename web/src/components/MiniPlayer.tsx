import { Pause, Play } from 'lucide-react'
import { fmt } from '../lib/format'
import type { SignalSummary } from '../lib/types'
import { usePlayback } from '../state/PlaybackContext'
import { cx } from './ui'

/**
 * Compact transport for a single signal, sized to sit directly above a
 * chart. Loop and speed stay in the global bar on purpose — they belong to
 * the player, not to one signal, so repeating them per chart would imply
 * they can differ.
 */
export function MiniPlayer({
  signal,
  label,
  tone = 'primary',
}: {
  signal: SignalSummary
  label?: string
  tone?: 'primary' | 'accent'
}) {
  const playback = usePlayback()
  const isCurrent = playback.signalId === signal.id
  const playing = isCurrent && playback.playing
  const position = isCurrent ? playback.currentTime : 0
  const total = isCurrent && playback.duration ? playback.duration : signal.duration
  const progress = total ? Math.min(1, position / total) : 0
  const fill = tone === 'accent' ? 'var(--accent)' : 'var(--primary)'

  return (
    <div className="mb-2.5 flex items-center gap-2.5 rounded-xl border border-border bg-surface-2 px-2.5 py-1.5">
      <button
        type="button"
        onClick={() => playback.toggle(signal.id)}
        title={playing ? `Pause ${label ?? signal.name}` : `Play ${label ?? signal.name}`}
        className={cx(
          'grid h-7 w-7 shrink-0 cursor-pointer place-items-center rounded-full transition-transform duration-150 hover:scale-105 active:scale-95',
          tone === 'accent' ? 'bg-accent text-white' : 'bg-primary text-white',
        )}
      >
        {playing ? (
          <Pause size={12} fill="currentColor" aria-hidden />
        ) : (
          <Play size={12} fill="currentColor" className="ml-0.5" aria-hidden />
        )}
        <span className="sr-only">
          {playing ? 'Pause' : 'Play'} {label ?? signal.name}
        </span>
      </button>

      {label && (
        <span className="shrink-0 text-[12px] font-semibold text-text">{label}</span>
      )}

      <span className="tnum hidden shrink-0 text-[11px] text-faint sm:inline">
        {fmt.duration(position)} / {fmt.duration(total)}
      </span>

      <label htmlFor={`mini-seek-${signal.id}`} className="sr-only">
        Seek {label ?? signal.name}
      </label>
      <input
        id={`mini-seek-${signal.id}`}
        type="range"
        min={0}
        max={total || 1}
        step={0.01}
        value={position}
        onChange={(e) => {
          const time = Number(e.target.value)
          if (isCurrent) playback.seek(time)
          else playback.cue(signal.id, time)
        }}
        style={{
          background: `linear-gradient(to right, ${fill} ${progress * 100}%, var(--surface-3) ${progress * 100}%)`,
        }}
        className="h-1.5 min-w-[60px] flex-1"
      />
    </div>
  )
}
