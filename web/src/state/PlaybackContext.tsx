import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { api } from '../lib/api'

export const PLAYBACK_RATES = [1, 1.25, 1.5, 2] as const

interface PlaybackState {
  signalId: string | null
  playing: boolean
  currentTime: number
  duration: number
  loop: boolean
  rate: number
  toggle: (signalId: string) => void
  play: (signalId: string) => void
  pause: () => void
  seek: (time: number) => void
  /** Point the player at a signal and a position without starting playback. */
  cue: (signalId: string, time: number) => void
  toggleLoop: () => void
  cycleRate: () => void
  isActive: (signalId: string) => boolean
}

const Ctx = createContext<PlaybackState | null>(null)

/** One shared <audio> element for the whole app: playing a signal anywhere
 *  stops whatever was playing before, and every view can show the playhead. */
export function PlaybackProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const pendingSeekRef = useRef<number | null>(null)
  const [signalId, setSignalId] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [loop, setLoop] = useState(false)
  const [rate, setRate] = useState(1)

  if (!audioRef.current && typeof Audio !== 'undefined') {
    audioRef.current = new Audio()
    // Buffer without waiting for playback, so clicking a waveform can seek
    // a track that has never been played.
    audioRef.current.preload = 'auto'
  }

  // Loop and rate live on the element, so they survive track changes.
  useEffect(() => {
    if (audioRef.current) audioRef.current.loop = loop
  }, [loop])

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = rate
  }, [rate])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    // While a seek is in flight the element still reports the old position,
    // so ignore timeupdate until it lands — otherwise the playhead snaps back.
    const onTime = () => {
      if (pendingSeekRef.current != null) return
      setCurrentTime(audio.currentTime)
    }
    const onMeta = () => setDuration(audio.duration || 0)
    const onEnd = () => {
      setPlaying(false)
      setCurrentTime(0)
    }
    audio.addEventListener('timeupdate', onTime)
    audio.addEventListener('loadedmetadata', onMeta)
    audio.addEventListener('ended', onEnd)
    return () => {
      audio.removeEventListener('timeupdate', onTime)
      audio.removeEventListener('loadedmetadata', onMeta)
      audio.removeEventListener('ended', onEnd)
    }
  }, [])

  // `timeupdate` only fires a few times a second, which makes a playhead
  // visibly step. Drive it from rAF while playing instead, capped at ~30fps
  // so charts aren't redrawn more often than the eye can use.
  useEffect(() => {
    if (!playing) return
    let frame = 0
    let last = 0
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick)
      if (now - last < 33) return
      last = now
      const audio = audioRef.current
      if (audio && pendingSeekRef.current == null) setCurrentTime(audio.currentTime)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing])

  const play = useCallback(
    (id: string) => {
      const audio = audioRef.current
      if (!audio) return
      if (signalId !== id) {
        audio.src = api.audioUrl(id)
        audio.playbackRate = rate
        audio.loop = loop
        setSignalId(id)
        setCurrentTime(0)
      }
      void audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false))
    },
    [signalId, rate, loop],
  )

  /** Move the playhead, optimistically in the UI and for real on the element
   *  as soon as it has metadata. `pendingSeekRef` keeps timeupdate from
   *  overwriting the requested position while the seek is in flight. */
  const applySeek = useCallback((time: number) => {
    const audio = audioRef.current
    if (!audio) return

    pendingSeekRef.current = time
    setCurrentTime(time)

    const settle = () => {
      pendingSeekRef.current = null
      audio.removeEventListener('seeked', settle)
    }
    audio.addEventListener('seeked', settle)
    // Safety net: never leave the display frozen if no 'seeked' arrives.
    window.setTimeout(settle, 2000)

    const commit = () => {
      try {
        audio.currentTime = time
      } catch {
        settle()
      }
    }
    if (audio.readyState >= 1 /* HAVE_METADATA */) commit()
    else audio.addEventListener('loadedmetadata', commit, { once: true })
  }, [])

  /** Load a signal and park the playhead at `time` without playing, so a
   *  click on any waveform moves the playhead whether or not audio is running. */
  const cue = useCallback(
    (id: string, time: number) => {
      const audio = audioRef.current
      if (!audio) return
      if (signalId !== id) {
        audio.src = api.audioUrl(id)
        audio.playbackRate = rate
        audio.loop = loop
        setSignalId(id)
      }
      setCurrentTime(time)
      applySeek(time)
    },
    [signalId, rate, loop, applySeek],
  )

  const pause = useCallback(() => {
    audioRef.current?.pause()
    setPlaying(false)
  }, [])

  const toggle = useCallback(
    (id: string) => {
      if (signalId === id && playing) pause()
      else play(id)
    },
    [signalId, playing, play, pause],
  )

  const seek = useCallback(
    (time: number) => {
      setCurrentTime(time)
      applySeek(time)
    },
    [applySeek],
  )

  const toggleLoop = useCallback(() => setLoop((l) => !l), [])

  const cycleRate = useCallback(
    () =>
      setRate((r) => {
        const i = PLAYBACK_RATES.indexOf(r as (typeof PLAYBACK_RATES)[number])
        return PLAYBACK_RATES[(i + 1) % PLAYBACK_RATES.length]
      }),
    [],
  )

  const value = useMemo<PlaybackState>(
    () => ({
      signalId,
      playing,
      currentTime,
      duration,
      loop,
      rate,
      toggle,
      play,
      pause,
      seek,
      cue,
      toggleLoop,
      cycleRate,
      isActive: (id: string) => signalId === id && playing,
    }),
    [signalId, playing, currentTime, duration, loop, rate, toggle, play, pause, seek, cue, toggleLoop, cycleRate],
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function usePlayback(): PlaybackState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('usePlayback must be used inside <PlaybackProvider>')
  return ctx
}

/**
 * Playhead wiring for a chart of one signal: the live position when this
 * signal is the one loaded, and a seek handler that works either way.
 */
export function useSignalPlayhead(signalId: string) {
  const playback = usePlayback()
  const isCurrent = playback.signalId === signalId
  return {
    playhead: isCurrent ? playback.currentTime : null,
    onSeek: (time: number) =>
      isCurrent ? playback.seek(time) : playback.cue(signalId, time),
  }
}
