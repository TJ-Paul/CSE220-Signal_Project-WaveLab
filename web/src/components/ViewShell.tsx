import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { AlertTriangle, AudioLines, Loader2 } from 'lucide-react'
import { Button } from './ui'

/** Fetch-on-deps with loading/error state and a manual reload. */
export function useAsyncData<T>(
  fn: () => Promise<T>,
  deps: unknown[],
  options: { enabled?: boolean } = {},
) {
  const enabled = options.enabled ?? true
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(enabled)
  const [error, setError] = useState<string | null>(null)
  const fnRef = useRef(fn)
  fnRef.current = fn
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    if (!enabled) {
      setLoading(false)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    fnRef
      .current()
      .then((result) => {
        if (!cancelled) setData(result)
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  return { data, loading, error, reload }
}

export function ViewHeader({
  title,
  description,
  actions,
}: {
  title: string
  description?: string
  actions?: ReactNode
}) {
  return (
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-[22px] font-semibold tracking-tight text-text">{title}</h1>
        {description && <p className="mt-0.5 max-w-2xl text-[13px] text-muted">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  )
}

export function ChartSkeleton({ height = 220 }: { height?: number }) {
  return (
    <div
      className="flex animate-pulse items-center justify-center rounded-xl border border-border bg-surface-2"
      style={{ height }}
    >
      <Loader2 size={18} className="animate-spin text-faint" aria-hidden />
      <span className="sr-only">Loading chart data</span>
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-danger/35 bg-danger-soft px-6 py-8 text-center">
      <AlertTriangle size={22} className="text-danger" aria-hidden />
      <div>
        <div className="text-[14px] font-semibold text-text">Something went wrong</div>
        <p className="mt-1 max-w-md text-[13px] text-muted">{message}</p>
      </div>
      {onRetry && (
        <Button size="sm" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border bg-surface px-6 py-14 text-center">
      <span className="grid h-12 w-12 place-items-center rounded-2xl bg-primary-soft text-primary">
        <AudioLines size={22} aria-hidden />
      </span>
      <div>
        <div className="text-[15px] font-semibold text-text">{title}</div>
        <p className="mx-auto mt-1 max-w-sm text-[13px] text-muted">{description}</p>
      </div>
      {action}
    </div>
  )
}

/** Vertical rhythm + entrance animation shared by every view. */
export function ViewBody({ children }: { children: ReactNode }) {
  return <div className="animate-in space-y-4">{children}</div>
}
