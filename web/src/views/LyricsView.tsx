import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  AudioLines,
  Check,
  Copy,
  Cpu,
  Download,
  Expand,
  FileText,
  FileUp,
  ListMusic,
  Maximize2,
  Mic,
  Music4,
  Pause,
  Play,
  Shrink,
  SkipBack,
  SkipForward,
  Sparkles,
  Wand2,
  X,
} from 'lucide-react'
import { api } from '../lib/api'
import type {
  LyricLine,
  LyricsLanguage,
  LyricsMode,
  Job,
  SignalSummary,
  TranscribeData,
} from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { usePlayback, useSignalPlayhead } from '../state/PlaybackContext'
import { WaveformChart } from '../components/charts/WaveformChart'
import { MiniPlayer } from '../components/MiniPlayer'
import { Button, Card, Notice, SegmentedControl, Stat, cx } from '../components/ui'
import { ChartSkeleton, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'

const LANGUAGES: { code: LyricsLanguage; name: string; native: string; glyph: string }[] = [
  { code: 'en', name: 'English', native: 'English', glyph: 'Aa' },
  { code: 'hi', name: 'Hindi', native: 'हिन्दी', glyph: 'हि' },
  { code: 'bn', name: 'Bangla', native: 'বাংলা', glyph: 'বা' },
]

const DEVICE_LABEL: Record<string, string> = {
  mps: 'Apple GPU (MPS)',
  cuda: 'NVIDIA GPU (CUDA)',
  cpu: 'CPU',
}

/* Time formats ------------------------------------------------------------ */

const pad = (n: number, w = 2) => String(Math.floor(n)).padStart(w, '0')
const clock = (t: number) => `${pad(t / 60)}:${pad(t % 60)}`
// Work in whole milliseconds so 3.84 s prints as 3,840, not 3,839.
const srtTime = (t: number) => {
  const ms = Math.round(t * 1000)
  return `${pad(ms / 3_600_000)}:${pad((ms % 3_600_000) / 60_000)}:${pad((ms % 60_000) / 1000)},${pad(ms % 1000, 3)}`
}
const lrcTime = (t: number) => {
  const cs = Math.round(t * 100)
  return `${pad(cs / 6000)}:${pad((cs % 6000) / 100)}.${pad(cs % 100)}`
}

function toSrt(lines: LyricLine[]) {
  return lines
    .map((l, i) => `${i + 1}\n${srtTime(l.start)} --> ${srtTime(l.end)}\n${l.text}\n`)
    .join('\n')
}
const toLrc = (lines: LyricLine[]) => lines.map((l) => `[${lrcTime(l.start)}]${l.text}`).join('\n')
const toTxt = (lines: LyricLine[]) =>
  lines.map((l) => `[${clock(l.start)} – ${clock(l.end)}]  ${l.text}`).join('\n')

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  a.click()
  URL.revokeObjectURL(url)
}

/** Index of the line being sung at `t`: the last line that has started, as
 *  long as it hasn't ended more than a beat ago. */
function activeIndex(lines: LyricLine[], t: number) {
  let idx = -1
  for (let i = 0; i < lines.length && lines[i].start <= t; i++) idx = i
  if (idx >= 0 && t > lines[idx].end + 1.5) return -1
  return idx
}

/** A line whose words light up as they are sung, from Whisper's word times.
 *  Each word carries its own leading space from the tokenizer. */
function SungText({ line, time }: { line: LyricLine; time: number }) {
  if (!line.words.length) return <>{line.text}</>
  return (
    <>
      {line.words.map((w, i) => (
        <span key={i} className={cx('transition-colors duration-150', w.start <= time ? 'text-primary' : 'opacity-60')}>
          {i === 0 ? w.text.trimStart() : w.text}
        </span>
      ))}
    </>
  )
}

/* Lyrics input ------------------------------------------------------------ */

// Mirrors lyrics_align.parse_lyrics, so the counts match what gets synced.
const LRC_TAG = /^\s*(\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\])+/
const SECTION = /^\s*\[[^\]]*\]\s*$/
function lyricLines(text: string) {
  return text
    .split('\n')
    .map((l) => l.replace(LRC_TAG, '').trim())
    .filter((l) => l && !SECTION.test(l))
}

const lyricsKey = (id: string) => `signal-lab-lyrics-${id}`
function loadLyrics(id: string) {
  try {
    return sessionStorage.getItem(lyricsKey(id)) ?? ''
  } catch {
    return ''
  }
}

function LyricsInput({
  value,
  onChange,
  language,
}: {
  value: string
  onChange: (v: string) => void
  language: LyricsLanguage
}) {
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const lines = lyricLines(value)
  const words = lines.reduce((n, l) => n + l.split(/\s+/).length, 0)

  const load = async (file: File | undefined) => {
    if (file) onChange(await file.text())
  }

  return (
    <Card
      title="Your lyrics"
      subtitle="One sung line per line, in order — include every repeat of the chorus"
      actions={
        <>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.lrc,text/plain"
            className="hidden"
            onChange={(e) => {
              void load(e.target.files?.[0])
              e.target.value = ''
            }}
          />
          <Button size="sm" onClick={() => fileRef.current?.click()} icon={<FileUp size={13} />}>
            Open .txt / .lrc
          </Button>
          {value && (
            <Button size="sm" variant="ghost" onClick={() => onChange('')}>
              Clear
            </Button>
          )}
        </>
      }
    >
      <div
        className="relative"
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          void load(e.dataTransfer.files?.[0])
        }}
      >
        <label htmlFor="lyrics-input" className="sr-only">
          Lyrics
        </label>
        <textarea
          id="lyrics-input"
          lang={language}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={9}
          spellCheck={false}
          placeholder={'Paste the lyrics here, or drop a .txt / .lrc file.\n\n[Chorus] headers, blank lines and LRC timestamps are ignored.'}
          className={cx(
            'font-lyrics block w-full resize-y rounded-xl border bg-surface-2 px-3.5 py-3 text-[14px] leading-relaxed text-text',
            'placeholder:text-faint focus:border-primary focus:outline-none focus:ring-3 focus:ring-primary-soft',
            dragging ? 'border-primary' : 'border-border',
          )}
        />
        {dragging && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center rounded-xl border-2 border-dashed border-primary bg-primary-soft text-[13px] font-semibold text-primary">
            Drop the lyrics file
          </div>
        )}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 text-[11px] text-faint">
        <span className="tnum">
          {lines.length} line{lines.length === 1 ? '' : 's'} · {words} word{words === 1 ? '' : 's'}
        </span>
        <span>Your text is kept exactly; only the timing comes from the audio.</span>
      </div>
    </Card>
  )
}

/* View -------------------------------------------------------------------- */

export function LyricsView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [language, setLanguage] = useState<LyricsLanguage>('en')
  const [mode, setMode] = useState<LyricsMode>('song')
  const [lyricsSource, setLyricsSource] = useState<'auto' | 'mine'>(() =>
    loadLyrics(signal.id) ? 'mine' : 'auto',
  )
  const [lyrics, setLyrics] = useState(() => loadLyrics(signal.id))
  const [result, setResult] = useState<TranscribeData | null>(null)
  const [running, setRunning] = useState(false)
  const [job, setJob] = useState<Job<TranscribeData> | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const status = useAsyncData(() => api.transcribeStatus(), [])
  const wave = useAsyncData(() => api.waveform(signal.id, 1600), [signal.id])
  const head = useSignalPlayhead(signal.id)

  // Pasted lyrics survive switching tabs (per signal, this browser session only).
  useEffect(() => {
    try {
      if (lyrics) sessionStorage.setItem(lyricsKey(signal.id), lyrics)
      else sessionStorage.removeItem(lyricsKey(signal.id))
    } catch {
      /* storage unavailable: the text simply isn't remembered */
    }
  }, [lyrics, signal.id])

  const syncing = lyricsSource === 'mine'
  const lineCount = syncing ? lyricLines(lyrics).length : 0

  // Wall-clock timer while the request runs, so a long song visibly progresses.
  useEffect(() => {
    if (!running) return
    const started = performance.now()
    const id = window.setInterval(() => setElapsed((performance.now() - started) / 1000), 200)
    return () => window.clearInterval(id)
  }, [running])

  const run = async () => {
    setRunning(true)
    setJob(null)
    setElapsed(0)
    setError(null)
    try {
      const data = await api.transcribe(
        signal.id,
        { language, mode, ...(syncing ? { lyrics } : {}) },
        setJob,
      )
      if (data.vocals) registerDerived(data.vocals)
      setResult(data)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  const unavailable = status.data && !status.data.available
  const device = status.data?.device ? DEVICE_LABEL[status.data.device] ?? status.data.device : null

  return (
    <ViewBody>
      <ViewHeader
        title="Lyrics & Transcription"
        description="Isolate the singer with Demucs, then turn the voice into timestamped text with Whisper — or paste the lyrics you already have and sync them to the audio."
        actions={
          <Button
            variant="primary"
            loading={running}
            disabled={Boolean(unavailable) || (syncing && lineCount === 0)}
            onClick={() => void run()}
            icon={syncing ? <Wand2 size={14} /> : <FileText size={14} />}
          >
            {syncing ? 'Sync my lyrics' : mode === 'song' ? 'Extract lyrics' : 'Transcribe'}
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,2.2fr)_minmax(0,1fr)]">
        <Card title="Source" subtitle={signal.name}>
          <MiniPlayer signal={signal} label="Original" />
          {wave.loading && <ChartSkeleton height={170} />}
          {wave.data && (
            <WaveformChart
              min={wave.data.min}
              max={wave.data.max}
              startTime={wave.data.startTime}
              endTime={wave.data.endTime}
              height={170}
              {...head}
            />
          )}
        </Card>

        <Card title="Settings">
          <div className="space-y-4">
            <div>
              <div className="mb-1.5 text-xs font-medium text-muted">Language of the audio</div>
              <div role="radiogroup" aria-label="Language" className="grid grid-cols-3 gap-2">
                {LANGUAGES.map((l) => {
                  const active = l.code === language
                  return (
                    <button
                      key={l.code}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      onClick={() => setLanguage(l.code)}
                      className={cx(
                        'group flex cursor-pointer flex-col items-center gap-1 rounded-xl border px-2 py-2.5',
                        'transition-all duration-150',
                        active
                          ? 'border-primary bg-primary-soft shadow-[0_0_0_3px_var(--primary-soft)]'
                          : 'border-border bg-surface-2 hover:border-border-strong hover:bg-surface-3',
                      )}
                    >
                      <span
                        lang={l.code}
                        className={cx(
                          'font-lyrics grid h-9 w-9 place-items-center rounded-full text-[15px] font-semibold',
                          'transition-colors duration-150',
                          active ? 'bg-primary text-white' : 'bg-surface-3 text-muted group-hover:text-text',
                        )}
                      >
                        {l.glyph}
                      </span>
                      <span lang={l.code} className="font-lyrics text-[13px] font-semibold text-text">
                        {l.native}
                      </span>
                      <span className="text-[10px] uppercase tracking-[0.08em] text-faint">{l.name}</span>
                    </button>
                  )
                })}
              </div>
              <p className="mt-1.5 text-[11px] text-faint">
                Telling Whisper the language skips its detection pass and avoids a wrong guess on a
                sung intro.
              </p>
            </div>

            <SegmentedControl
              label="Lyrics"
              value={lyricsSource}
              options={[
                { value: 'auto', label: 'Transcribe for me', hint: 'Whisper writes the words' },
                { value: 'mine', label: 'Use my lyrics', hint: 'Sync lyrics you already have' },
              ]}
              onChange={setLyricsSource}
            />
            <p className="-mt-1 text-[11px] text-faint">
              {syncing
                ? 'Your exact text, timed word by word against the audio (forced alignment).'
                : 'Whisper writes the words it hears; expect some mistakes.'}
            </p>

            <SegmentedControl
              label="What is it?"
              value={mode}
              options={[
                { value: 'song', label: 'Song', hint: 'Isolate the vocals first' },
                { value: 'voice', label: 'Voice recording', hint: 'Speech only, no music' },
              ]}
              onChange={setMode}
            />
            <p className="-mt-1 text-[11px] text-faint">
              {mode === 'song'
                ? 'Demucs removes the instruments first, so Whisper hears only the singer.'
                : 'No music to remove, so the audio goes straight to Whisper.'}
            </p>

            {status.data?.available && (
              <div className="flex items-center gap-2 rounded-lg border border-border bg-surface-2 px-2.5 py-2 text-[11px] text-muted">
                <Cpu size={13} className="shrink-0 text-success" aria-hidden />
                <span>
                  Whisper <span className="font-mono text-text">{status.data.model}</span>
                  {device && <> on {device}</>}
                </span>
              </div>
            )}
          </div>
        </Card>
      </div>

      {unavailable && (
        <Notice tone="warn" title="Transcription is not installed">
          Run <code>pip install openai-whisper demucs</code> in the virtual environment and restart
          the API.
        </Notice>
      )}
      {syncing && <LyricsInput value={lyrics} onChange={setLyrics} language={language} />}

      {error && <Notice tone="danger">{error}</Notice>}

      {running && (
        <PipelineProgress
          mode={mode}
          syncing={syncing}
          elapsed={elapsed}
          ahead={job?.status === 'queued' ? job.ahead : null}
        />
      )}

      {result && !running && job?.reused && (
        <Notice tone="success">
          This song was already transcribed with these settings, so the saved result came straight
          back.
        </Notice>
      )}
      {result && !running && <LyricsResult signal={signal} result={result} />}

      {!result && !running && <PipelineIntro />}
    </ViewBody>
  )
}

/* Running ----------------------------------------------------------------- */

function PipelineProgress({
  mode,
  syncing,
  elapsed,
  ahead,
}: {
  mode: LyricsMode
  syncing: boolean
  elapsed: number
  /** Jobs ahead of this one while it waits for the GPU; null once running. */
  ahead: number | null
}) {
  const steps = [
    ...(mode === 'song' ? [{ icon: Music4, title: 'Isolating vocals', detail: 'Demucs · htdemucs' }] : []),
    { icon: AudioLines, title: 'Speech to text', detail: 'Whisper · 30 s windows' },
    syncing
      ? { icon: ListMusic, title: 'Aligning your lyrics', detail: 'forced alignment · DTW' }
      : { icon: Sparkles, title: 'Timestamping & clean-up', detail: 'energy gate on the vocal' },
  ]
  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-1 flex-wrap items-center gap-2">
          {steps.map((s, i) => (
            <div key={s.title} className="flex items-center gap-2">
              {i > 0 && <span className="hidden h-px w-6 bg-border-strong sm:block" aria-hidden />}
              <div className="flex items-center gap-2.5 rounded-xl border border-border bg-surface-2 px-3 py-2">
                <span className="relative grid h-7 w-7 place-items-center rounded-full bg-primary-soft text-primary">
                  <span
                    className="absolute inset-0 animate-ping rounded-full bg-primary/20"
                    style={{ animationDelay: `${i * 300}ms` }}
                    aria-hidden
                  />
                  <s.icon size={14} aria-hidden />
                </span>
                <div>
                  <div className="text-[12px] font-semibold text-text">{s.title}</div>
                  <div className="text-[10.5px] text-faint">{s.detail}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="text-right">
          <div className="tnum text-[22px] font-semibold text-text">{elapsed.toFixed(1)} s</div>
          <div className="text-[11px] text-faint">
            {ahead !== null
              ? `waiting for the GPU · ${ahead} ahead`
              : 'runs on the server · first run also loads the models'}
          </div>
        </div>
      </div>
    </Card>
  )
}

/* Result ------------------------------------------------------------------ */

function LyricsResult({ signal, result }: { signal: SignalSummary; result: TranscribeData }) {
  const playback = usePlayback()
  const [source, setSource] = useState<'mix' | 'vocals'>('mix')
  const [copied, setCopied] = useState(false)
  const [stageOpen, setStageOpen] = useState(false)
  const listRef = useRef<HTMLOListElement | null>(null)

  const player = source === 'vocals' && result.vocals ? result.vocals : signal
  const tracking = playback.signalId === signal.id || playback.signalId === result.vocals?.id
  const current = tracking ? activeIndex(result.lines, playback.currentTime) : -1

  const words = useMemo(
    () => result.lines.reduce((n, l) => n + l.text.split(/\s+/).filter(Boolean).length, 0),
    [result.lines],
  )
  const totalS =
    (result.timings.separateS ?? 0) + result.timings.transcribeS + (result.timings.alignS ?? 0)
  const speed = signal.duration / Math.max(totalS, 1e-3)

  // Keep the sung line in view, scrolling only the lyric sheet, never the page.
  useEffect(() => {
    const list = listRef.current
    if (!list || current < 0 || !playback.playing) return
    const el = list.children[current] as HTMLElement | undefined
    if (el) list.scrollTo({ top: el.offsetTop - list.clientHeight / 3, behavior: 'smooth' })
  }, [current, playback.playing])

  const seekTo = (t: number) => {
    playback.cue(player.id, t)
    playback.play(player.id)
  }

  const base = signal.name.replace(/\.[^.]+$/, '')
  const copy = async () => {
    const text = toTxt(result.lines)
    // navigator.clipboard exists only on https or localhost, not for LAN guests.
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text)
    } else {
      const area = document.createElement('textarea')
      area.value = text
      area.setAttribute('readonly', '')
      area.style.position = 'fixed'
      area.style.opacity = '0'
      document.body.appendChild(area)
      area.select()
      document.execCommand('copy')
      area.remove()
    }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Lines"
          value={String(result.lines.length)}
          tone="primary"
          hint={result.aligned ? 'from your lyrics' : 'timestamped segments'}
        />
        <Stat
          label="Words"
          value={String(words)}
          tone="accent"
          hint={result.aligned ? `each one timed · ${result.languageName}` : `${result.languageName} transcript`}
        />
        <Stat
          label="Processing time"
          value={totalS.toFixed(1)}
          unit="s"
          tone="success"
          hint={
            (speed >= 1 ? `${speed.toFixed(1)}× faster than real time` : 'slower than real time') +
            (result.timings.loadModelsS ? ` · +${result.timings.loadModelsS.toFixed(1)} s one-time model load` : '')
          }
        />
        <Stat
          label="Ran on"
          value={result.device.toUpperCase()}
          hint={
            result.mode === 'voice'
              ? `${DEVICE_LABEL[result.device] ?? result.device} · Whisper ${result.timings.transcribeS.toFixed(1)} s`
              : result.timings.stemReused
                ? 'vocal stem reused, Demucs skipped'
                : `Demucs ${result.timings.separateS?.toFixed(1)} s · Whisper ${result.timings.transcribeS.toFixed(1)} s` +
                  (result.timings.alignS ? ` · align ${result.timings.alignS.toFixed(1)} s` : '')
          }
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        <Card
          title={result.aligned ? 'Your lyrics, synced' : 'Transcript'}
          subtitle="Click a line to play from there"
          padded={false}
          actions={
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void copy()}
              disabled={!result.lines.length}
              icon={copied ? <Check size={13} /> : <Copy size={13} />}
            >
              {copied ? 'Copied' : 'Copy'}
            </Button>
          }
        >
          {result.lines.length === 0 ? (
            <p className="px-4 py-10 text-center text-[13px] text-muted">
              No words were recognised. Check the language, or try “Voice recording” if there is no
              music.
            </p>
          ) : (
            <ol
              ref={listRef}
              lang={result.language}
              className="relative max-h-[520px] space-y-1 overflow-y-auto p-3"
            >
              {result.lines.map((line, i) => {
                const active = i === current
                const past = current >= 0 && i < current
                return (
                  <li key={`${line.start}-${i}`}>
                    <button
                      type="button"
                      onClick={() => seekTo(line.start)}
                      className={cx(
                        'group flex w-full cursor-pointer items-start gap-3 rounded-xl border-l-[3px] px-3 py-2 text-left',
                        'transition-all duration-200',
                        active
                          ? 'border-l-primary bg-primary-soft'
                          : 'border-l-transparent hover:bg-surface-2',
                      )}
                    >
                      <span
                        className={cx(
                          'tnum mt-[3px] shrink-0 rounded-md px-1.5 py-0.5 font-mono text-[11px]',
                          active ? 'bg-primary text-white' : 'bg-surface-3 text-faint group-hover:text-muted',
                        )}
                      >
                        {clock(line.start)}
                      </span>
                      <span
                        className={cx(
                          'font-lyrics leading-relaxed transition-all duration-200',
                          active
                            ? 'text-[17px] font-semibold text-text'
                            : past
                              ? 'text-[15px] text-faint'
                              : 'text-[15px] text-muted group-hover:text-text',
                        )}
                      >
                        {active ? <SungText line={line} time={playback.currentTime} /> : line.text}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ol>
          )}
          <div className="flex flex-wrap items-center gap-2 border-t border-border px-4 py-3">
            <span className="mr-auto text-[11px] text-faint">
              Export · SRT for video subtitles, LRC for music players
            </span>
            {(
              [
                ['SRT', () => download(`${base}.srt`, toSrt(result.lines))],
                ['LRC', () => download(`${base}.lrc`, toLrc(result.lines))],
                ['TXT', () => download(`${base}.txt`, toTxt(result.lines))],
              ] as const
            ).map(([label, fn]) => (
              <Button key={label} size="sm" onClick={fn} icon={<Download size={13} />} disabled={!result.lines.length}>
                {label}
              </Button>
            ))}
          </div>
        </Card>

        <div className="space-y-4">
          <Card
            title="Listen along"
            actions={
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setStageOpen(true)}
                disabled={!result.lines.length}
                icon={<Maximize2 size={13} />}
                title="Open a full-screen player with live lyrics"
              >
                Pop out
              </Button>
            }
          >
            {result.vocals && (
              <div className="mb-3">
                <SegmentedControl
                  label="Play"
                  value={source}
                  options={[
                    { value: 'mix', label: 'Original' },
                    { value: 'vocals', label: 'Isolated vocals' },
                  ]}
                  onChange={setSource}
                />
              </div>
            )}
            <MiniPlayer
              signal={player}
              label={source === 'vocals' && result.vocals ? 'Vocals' : 'Original'}
              tone={source === 'vocals' ? 'accent' : 'primary'}
            />
            <NowSinging
              lines={result.lines}
              current={current}
              time={playback.currentTime}
              language={result.language}
            />
          </Card>

          {result.aligned && result.confidence != null && <SyncQuality result={result} />}
          {!result.aligned && result.recoveredLines > 0 && (
            <Notice tone="success" title={`${result.recoveredLines} line${result.recoveredLines > 1 ? 's' : ''} recovered`}>
              Whisper skipped part of the vocal (it tends to merge repeated lines, like a chorus).
              The vocal's energy showed someone was singing there, so those stretches were
              transcribed again on their own.
            </Notice>
          )}
          {result.droppedSilent > 0 && (
            <Notice tone="info" title={`${result.droppedSilent} line${result.droppedSilent > 1 ? 's' : ''} removed`}>
              Whisper produced text where the vocal was near-silent. Those lines were more than 35
              dB below the loudest singing, so they were dropped as hallucinations.
            </Notice>
          )}
        </div>
      </div>

      {stageOpen && (
        <LyricsStage
          signal={signal}
          result={result}
          source={source}
          onSource={setSource}
          onClose={() => setStageOpen(false)}
        />
      )}
    </>
  )
}

/** Confidence = mean probability the model gives your words, given the audio.
 *  Measured: correct lyrics 0.81–0.97, lyrics from another song 0.08–0.63. */
function SyncQuality({ result }: { result: TranscribeData }) {
  const confidence = result.confidence ?? 0
  const good = confidence >= 0.7
  return (
    <Notice
      tone={good ? 'success' : 'warn'}
      title={good ? `Synced · ${Math.round(confidence * 100)}% confidence` : 'These lyrics may not match this song'}
    >
      {good ? (
        <>
          Every word was timed against the audio by forced alignment, then trimmed to where the
          voice actually sounds. Whisper independently heard {Math.round((result.matchRate ?? 0) * 100)}%
          of your words; that only reflects how clearly it hears this language, not your lyrics.
        </>
      ) : (
        <>
          Alignment confidence is {Math.round(confidence * 100)}%; correct lyrics usually score above
          80%. Check that the lyrics belong to this song, are in singing order, and include every
          repeated line.
        </>
      )}
    </Notice>
  )
}

function NowSinging({
  lines,
  current,
  time,
  language,
}: {
  lines: LyricLine[]
  current: number
  time: number
  language: LyricsLanguage
}) {
  const line = current >= 0 ? lines[current] : null
  const next = lines[current + 1] ?? null
  return (
    <div
      lang={language}
      className="mt-1 flex min-h-[150px] flex-col justify-center rounded-xl border border-border px-4 py-4 text-center"
      style={{ background: 'linear-gradient(135deg, var(--primary-soft), var(--accent-soft))' }}
    >
      <div className="mb-2 text-[10.5px] font-semibold uppercase tracking-[0.1em] text-faint">
        {line ? 'Now' : 'Press play'}
      </div>
      <p key={current} className="font-lyrics animate-in text-[20px] font-semibold leading-snug text-text">
        {line ? <SungText line={line} time={time} /> : '♪'}
      </p>
      {next && (
        <p className="font-lyrics mt-2.5 text-[13px] leading-snug text-faint">{next.text}</p>
      )}
    </div>
  )
}

/* Pop-out player --------------------------------------------------------- */

/** Full-screen "listen along" mode: just the song and its lyrics, no
 *  metrics. It drives the same shared player as the rest of the app, so
 *  closing it leaves playback exactly where it was. */
function LyricsStage({
  signal,
  result,
  source,
  onSource,
  onClose,
}: {
  signal: SignalSummary
  result: TranscribeData
  source: 'mix' | 'vocals'
  onSource: (s: 'mix' | 'vocals') => void
  onClose: () => void
}) {
  const playback = usePlayback()
  const rootRef = useRef<HTMLDivElement | null>(null)
  const sheetRef = useRef<HTMLDivElement | null>(null)
  const [fullscreen, setFullscreen] = useState(false)

  const lines = result.lines
  const player = source === 'vocals' && result.vocals ? result.vocals : signal
  const tracking = playback.signalId === signal.id || playback.signalId === result.vocals?.id
  const isCurrent = playback.signalId === player.id
  const playing = isCurrent && playback.playing
  const time = tracking ? playback.currentTime : 0
  const total = isCurrent && playback.duration ? playback.duration : player.duration
  const progress = total ? Math.min(1, time / total) : 0
  const current = tracking ? activeIndex(lines, time) : -1

  const seekTo = (t: number, andPlay = false) => {
    if (isCurrent) playback.seek(t)
    else playback.cue(player.id, t)
    if (andPlay) playback.play(player.id)
  }

  // Previous / next jump by lyric line rather than by a fixed number of seconds.
  const lastStarted = (() => {
    let i = -1
    while (i + 1 < lines.length && lines[i + 1].start <= time) i++
    return i
  })()
  const prevLine = () => {
    // Like a music player: first press restarts the line, a quick second one goes back.
    const i = lastStarted >= 0 && time - lines[lastStarted].start < 1.5 ? lastStarted - 1 : lastStarted
    seekTo(i >= 0 ? lines[i].start : 0, playing)
  }
  const nextLine = () => {
    const next = lines[lastStarted + 1]
    if (next) seekTo(next.start, playing)
  }

  // Switching between the mix and the vocal keeps the position (and playback).
  const switchSource = (next: 'mix' | 'vocals') => {
    const target = next === 'vocals' && result.vocals ? result.vocals : signal
    const wasPlaying = tracking && playback.playing
    onSource(next)
    if (!tracking || target.id === playback.signalId) return
    playback.cue(target.id, playback.currentTime)
    if (wasPlaying) playback.play(target.id)
  }

  // Lock the page behind, take focus, and give it back on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    rootRef.current?.focus()
    return () => {
      document.body.style.overflow = overflow
      if (document.fullscreenElement) void document.exitFullscreen().catch(() => {})
      opener?.focus()
    }
  }, [])

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === rootRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen().catch(() => {})
    else void rootRef.current?.requestFullscreen().catch(() => {})
  }

  // Space plays/pauses, arrows skip 5 s, Escape closes. Keys aimed at a
  // focused control keep their native behaviour.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const onControl = e.target instanceof HTMLButtonElement || e.target instanceof HTMLInputElement
      if (e.key === 'Escape') onClose()
      else if (onControl) return
      else if (e.key === ' ') {
        e.preventDefault()
        playback.toggle(player.id)
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        const t = time + (e.key === 'ArrowLeft' ? -5 : 5)
        seekTo(Math.min(Math.max(0, t), total))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // Keep the sung line centred. Jump straight there on open, glide after that.
  const firstScroll = useRef(true)
  useEffect(() => {
    const sheet = sheetRef.current
    if (!sheet || current < 0) return
    const el = sheet.querySelector<HTMLElement>(`[data-line="${current}"]`)
    if (!el) return
    sheet.scrollTo({
      top: el.offsetTop - sheet.clientHeight / 2 + el.offsetHeight / 2,
      behavior: firstScroll.current ? 'auto' : 'smooth',
    })
    firstScroll.current = false
  }, [current])

  const title = signal.name.replace(/\.[^.]+$/, '')

  return createPortal(
    <div
      ref={rootRef}
      role="dialog"
      aria-modal="true"
      aria-label={`Lyrics player · ${title}`}
      tabIndex={-1}
      className="animate-in fixed inset-0 z-50 flex flex-col bg-bg text-text outline-none"
      style={{
        backgroundImage:
          'radial-gradient(ellipse 80% 60% at 15% 0%, var(--primary-soft), transparent 70%),' +
          'radial-gradient(ellipse 70% 60% at 90% 100%, var(--accent-soft), transparent 70%)',
      }}
    >
      <header className="flex items-center gap-3 px-4 py-3 sm:px-6 sm:py-4">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-primary-soft text-primary">
          <Music4 size={19} aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] font-semibold">{title}</div>
          <div className="text-[12px] text-faint">{result.languageName}</div>
        </div>

        {result.vocals && (
          <div role="radiogroup" aria-label="Play" className="hidden rounded-full bg-surface-2 p-1 sm:flex">
            {(
              [
                ['mix', 'Original'],
                ['vocals', 'Vocals only'],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={source === value}
                onClick={() => switchSource(value)}
                className={cx(
                  'cursor-pointer rounded-full px-3.5 py-1.5 text-[12px] font-semibold transition-colors duration-150',
                  source === value ? 'bg-primary text-white' : 'text-muted hover:text-text',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          onClick={toggleFullscreen}
          title={fullscreen ? 'Exit full screen' : 'Full screen'}
          className="grid h-10 w-10 cursor-pointer place-items-center rounded-full text-muted transition-colors hover:bg-surface-2 hover:text-text"
        >
          {fullscreen ? <Shrink size={18} aria-hidden /> : <Expand size={18} aria-hidden />}
          <span className="sr-only">{fullscreen ? 'Exit full screen' : 'Full screen'}</span>
        </button>
        <button
          type="button"
          onClick={onClose}
          title="Close (Esc)"
          className="grid h-10 w-10 cursor-pointer place-items-center rounded-full text-muted transition-colors hover:bg-surface-2 hover:text-text"
        >
          <X size={20} aria-hidden />
          <span className="sr-only">Close lyrics player</span>
        </button>
      </header>

      <div
        ref={sheetRef}
        lang={result.language}
        className="relative flex-1 overflow-y-auto px-5 sm:px-10"
        style={{
          maskImage: 'linear-gradient(to bottom, transparent, #000 14%, #000 86%, transparent)',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent, #000 14%, #000 86%, transparent)',
        }}
      >
        <ol className="mx-auto max-w-4xl py-[40vh] text-center">
          {lines.map((line, i) => {
            const active = i === current
            const past = current >= 0 && i < current
            return (
              <li key={`${line.start}-${i}`} data-line={i}>
                <button
                  type="button"
                  onClick={() => seekTo(line.start, true)}
                  className={cx(
                    'font-lyrics w-full cursor-pointer rounded-2xl px-3 py-2.5 leading-snug',
                    'text-[clamp(22px,3.4vw,40px)] font-bold transition-all duration-300',
                    active
                      ? 'scale-100 text-text'
                      : cx('scale-[0.94] hover:opacity-80', past ? 'text-faint opacity-40' : 'text-muted opacity-55'),
                  )}
                >
                  {active ? <SungText line={line} time={time} /> : line.text}
                </button>
              </li>
            )
          })}
        </ol>
      </div>

      <footer className="mx-auto w-full max-w-3xl px-5 pb-6 pt-3 sm:pb-8">
        <label htmlFor="stage-seek" className="sr-only">
          Seek
        </label>
        <input
          id="stage-seek"
          type="range"
          min={0}
          max={total || 1}
          step={0.01}
          value={time}
          onChange={(e) => seekTo(Number(e.target.value))}
          style={{
            background: `linear-gradient(to right, var(--primary) ${progress * 100}%, var(--surface-3) ${progress * 100}%)`,
          }}
          className="h-1.5 w-full"
        />
        <div className="tnum mt-1.5 flex justify-between font-mono text-[11px] text-faint">
          <span>{clock(time)}</span>
          <span>{clock(total)}</span>
        </div>
        <div className="mt-2 flex items-center justify-center gap-5">
          <button
            type="button"
            onClick={prevLine}
            title="Previous line"
            className="grid h-11 w-11 cursor-pointer place-items-center rounded-full text-muted transition-colors hover:bg-surface-2 hover:text-text"
          >
            <SkipBack size={20} fill="currentColor" aria-hidden />
            <span className="sr-only">Previous line</span>
          </button>
          <button
            type="button"
            onClick={() => playback.toggle(player.id)}
            title={playing ? 'Pause (Space)' : 'Play (Space)'}
            className="grid h-16 w-16 cursor-pointer place-items-center rounded-full bg-primary text-white shadow-lg transition-transform duration-150 hover:scale-105 active:scale-95"
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
            onClick={nextLine}
            disabled={lastStarted + 1 >= lines.length}
            title="Next line"
            className="grid h-11 w-11 cursor-pointer place-items-center rounded-full text-muted transition-colors hover:bg-surface-2 hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SkipForward size={20} fill="currentColor" aria-hidden />
            <span className="sr-only">Next line</span>
          </button>
        </div>
      </footer>
    </div>,
    document.body,
  )
}

/* Intro ------------------------------------------------------------------- */

function PipelineIntro() {
  const steps = [
    {
      icon: Music4,
      title: '1 · Separate',
      body: 'Demucs removes drums, bass and chords, leaving the vocal stem. Skipped for voice recordings.',
    },
    {
      icon: Mic,
      title: '2 · Transcribe',
      body: 'Whisper reads the log-mel spectrogram of the vocal in 30-second windows and writes the words with start and end times.',
    },
    {
      icon: Sparkles,
      title: '3 · Clean up',
      body: 'Word times are trimmed to where the voice actually sounds. Have the lyrics already? Choose “Use my lyrics” and your exact text is synced instead.',
    },
  ]
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {steps.map((s) => (
        <Card key={s.title}>
          <div className="flex items-start gap-3">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-primary-soft text-primary">
              <s.icon size={18} aria-hidden />
            </span>
            <div>
              <div className="text-[13px] font-semibold text-text">{s.title}</div>
              <p className="mt-1 text-[12.5px] text-muted">{s.body}</p>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}
