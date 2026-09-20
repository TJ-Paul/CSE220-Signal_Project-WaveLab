import { Pause, Play, Repeat, Upload } from 'lucide-react'
import type { ChangeEvent } from 'react'
import { useRef } from 'react'
import type { SignalSummary } from '../lib/types'
import { usePlayback } from '../state/PlaybackContext'
import { cx } from './ui'

/** mm:ss — player convention, distinct from the analytical mm:ss.SS. */
function clock(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds - m * 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

export function PlayerCard({
  signal,
  onUpload,
  uploading,
}: {
  signal: SignalSummary
  onUpload: (file: File) => void
  uploading: boolean
}) {
  const playback = usePlayback()
  const inputRef = useRef<HTMLInputElement>(null)

  const isCurrent = playback.signalId === signal.id
  const playing = isCurrent && playback.playing
  const position = isCurrent ? playback.currentTime : 0
  const total = isCurrent && playback.duration ? playback.duration : signal.duration
  const progress = total ? Math.min(1, position / total) : 0

  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onUpload(file)
    e.target.value = ''
  }

  return (
    <section className="relative flex min-w-0 flex-col items-center justify-center gap-5 rounded-[14px] border border-border bg-surface px-6 py-7 shadow-[0_1px_2px_rgb(0_0_0/0.16)]">
      <input
        ref={inputRef}
        type="file"
        accept=".wav,.mp3,audio/wav,audio/mpeg"
        className="sr-only"
        onChange={handleFile}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        title="Load a different audio file"
        className="absolute right-3 top-3 grid h-8 w-8 cursor-pointer place-items-center rounded-lg border border-border bg-surface-2 text-muted transition-colors hover:border-primary hover:text-primary disabled:opacity-50"
      >
        <Upload size={14} aria-hidden />
        <span className="sr-only">Load a different audio file</span>
      </button>

      <h2 className="max-w-full truncate px-8 text-center text-[19px] font-bold tracking-tight text-text">
        {signal.name}
      </h2>

      {/* Seek bar ------------------------------------------------------ */}
      <div className="w-full">
        <label htmlFor="player-seek" className="sr-only">
          Seek position
        </label>
        <input
          id="player-seek"
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
            background: `linear-gradient(to right, var(--primary) ${progress * 100}%, var(--surface-3) ${progress * 100}%)`,
          }}
          className="h-1.5"
        />
        <div className="mt-1.5 flex items-center justify-between">
          <span className="tnum text-[13px] font-semibold text-muted">{clock(position)}</span>
          <span className="tnum text-[13px] font-semibold text-muted">{clock(total)}</span>
        </div>
      </div>

      {/* Transport ----------------------------------------------------- */}
      <div className="flex items-center gap-7">
        <button
          type="button"
          onClick={playback.toggleLoop}
          aria-pressed={playback.loop}
          title={playback.loop ? 'Looping on' : 'Looping off'}
          className={cx(
            'grid h-10 w-10 cursor-pointer place-items-center rounded-full transition-colors',
            playback.loop
              ? 'bg-primary-soft text-primary'
              : 'text-muted hover:bg-surface-2 hover:text-text',
          )}
        >
          <Repeat size={19} aria-hidden />
          <span className="sr-only">{playback.loop ? 'Turn looping off' : 'Turn looping on'}</span>
        </button>

        <button
          type="button"
          onClick={() => playback.toggle(signal.id)}
          title={playing ? 'Pause' : 'Play'}
          className="grid h-16 w-16 cursor-pointer place-items-center rounded-full bg-text text-surface shadow-lg transition-transform duration-150 hover:scale-105 active:scale-95"
        >
          {playing ? (
            <Pause size={26} fill="currentColor" aria-hidden />
          ) : (
            <Play size={26} fill="currentColor" className="ml-1" aria-hidden />
          )}
          <span className="sr-only">{playing ? 'Pause' : 'Play'}</span>
        </button>

        <button
          type="button"
          onClick={playback.cycleRate}
          title="Playback speed"
          className={cx(
            'tnum grid h-10 w-10 cursor-pointer place-items-center rounded-full border text-[12px] font-bold transition-colors',
            playback.rate === 1
              ? 'border-border-strong text-muted hover:border-primary hover:text-primary'
              : 'border-primary bg-primary-soft text-primary',
          )}
        >
          {playback.rate}x<span className="sr-only">Playback speed, currently {playback.rate}x</span>
        </button>
      </div>
    </section>
  )
}
