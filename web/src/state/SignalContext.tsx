import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { api } from '../lib/api'
import type { SignalSummary } from '../lib/types'

interface SignalState {
  signals: SignalSummary[]
  active: SignalSummary | null
  activeId: string | null
  setActiveId: (id: string | null) => void
  busy: string | null
  error: string | null
  clearError: () => void
  refresh: () => Promise<SignalSummary[]>
  loadDemo: (kind: 'speech' | 'song' | 'noisy') => Promise<void>
  upload: (file: File) => Promise<void>
  clearAll: () => Promise<void>
  registerDerived: (signal: SignalSummary, makeActive?: boolean) => void
}

const Ctx = createContext<SignalState | null>(null)

export function SignalProvider({ children }: { children: ReactNode }) {
  const [signals, setSignals] = useState<SignalSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    const { signals: next } = await api.listSignals()
    setSignals(next)
    return next
  }, [])

  useEffect(() => {
    refresh().catch((e: Error) => setError(e.message))
  }, [refresh])

  const run = useCallback(
    async (label: string, fn: () => Promise<SignalSummary | void>) => {
      setBusy(label)
      setError(null)
      try {
        const result = await fn()
        const next = await refresh()
        if (result && 'id' in result) setActiveId(result.id)
        else if (!next.length) setActiveId(null)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy(null)
      }
    },
    [refresh],
  )

  const loadDemo = useCallback(
    (kind: 'speech' | 'song' | 'noisy') => run(`demo:${kind}`, () => api.loadDemo(kind)),
    [run],
  )

  const upload = useCallback(
    (file: File) => run('upload', () => api.uploadSignal(file)),
    [run],
  )

  const clearAll = useCallback(
    () =>
      run('clear', async () => {
        await api.clearSignals()
        setActiveId(null)
      }),
    [run],
  )

  // Processing views create new signals; surface them without a full reload.
  const registerDerived = useCallback(
    (signal: SignalSummary, makeActive = false) => {
      setSignals((prev) => (prev.some((s) => s.id === signal.id) ? prev : [...prev, signal]))
      if (makeActive) setActiveId(signal.id)
    },
    [],
  )

  const active = useMemo(
    () => signals.find((s) => s.id === activeId) ?? null,
    [signals, activeId],
  )

  // Keep a sensible selection when the active signal disappears.
  useEffect(() => {
    if (activeId && !signals.some((s) => s.id === activeId)) {
      setActiveId(signals.length ? signals[0].id : null)
    }
  }, [signals, activeId])

  const value: SignalState = {
    signals,
    active,
    activeId,
    setActiveId,
    busy,
    error,
    clearError: () => setError(null),
    refresh,
    loadDemo,
    upload,
    clearAll,
    registerDerived,
  }

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useSignals(): SignalState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useSignals must be used inside <SignalProvider>')
  return ctx
}
