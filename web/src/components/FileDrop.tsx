import { useCallback, useRef, useState, type ReactNode } from 'react'
import { Upload, X } from 'lucide-react'
import { fmt } from '../lib/format'
import { Button, cx } from './ui'

/** Drag-and-drop or browse, for one file. */
export function FileDrop({
  accept,
  label,
  hint,
  file,
  onFile,
  icon,
  compact = false,
}: {
  accept: string
  label: string
  hint: string
  file: File | null
  onFile: (file: File | null) => void
  icon?: ReactNode
  compact?: boolean
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const take = useCallback(
    (files: FileList | null) => {
      const next = files?.[0]
      if (next) onFile(next)
    },
    [onFile],
  )

  if (file) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-primary/40 bg-primary-soft px-3 py-2.5">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-primary/15 text-primary">
          {icon ?? <Upload size={15} aria-hidden />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[13px] font-medium text-text" title={file.name}>
            {file.name}
          </div>
          <div className="tnum text-[11px] text-muted">{fmt.bytes(file.size)}</div>
        </div>
        <button
          type="button"
          onClick={() => {
            onFile(null)
            if (inputRef.current) inputRef.current.value = ''
          }}
          aria-label={`Remove ${file.name}`}
          className="grid h-7 w-7 shrink-0 cursor-pointer place-items-center rounded-md text-muted transition-colors hover:bg-surface-3 hover:text-text"
        >
          <X size={14} aria-hidden />
        </button>
      </div>
    )
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        take(e.dataTransfer.files)
      }}
      className={cx(
        'flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed text-center transition-colors duration-200',
        compact ? 'px-4 py-5' : 'px-5 py-8',
        dragging ? 'border-primary bg-primary-soft' : 'border-border-strong bg-surface-2',
      )}
    >
      <span
        className={cx(
          'grid place-items-center rounded-xl transition-colors duration-200',
          compact ? 'h-9 w-9' : 'h-11 w-11',
          dragging ? 'bg-primary/15 text-primary' : 'bg-surface-3 text-muted',
        )}
      >
        {icon ?? <Upload size={compact ? 16 : 19} aria-hidden />}
      </span>
      <div>
        <div className="text-[13px] font-semibold text-text">{label}</div>
        <p className="mt-0.5 text-[11px] text-muted">{hint}</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={(e) => take(e.target.files)}
      />
      <Button size="sm" onClick={() => inputRef.current?.click()}>
        Browse
      </Button>
    </div>
  )
}
