import { useCallback, useEffect, useState } from 'react'
import { ArrowRight, Download, FileAudio, FileImage, File as FileIcon, FolderOpen, Trash2 } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { SharedFile, ViewId } from '../lib/types'
import { useIsHost } from '../lib/useIsHost'
import { FileDrop } from '../components/FileDrop'
import { Badge, Button, Card, Notice } from '../components/ui'
import { EmptyState, ErrorState, ViewBody, ViewHeader } from '../components/ViewShell'

const POLL_MS = 3000

const KIND_ICON = { image: FileImage, audio: FileAudio, file: FileIcon } as const

/** Which tab opens each kind of shared file. */
const OPENS_IN: Partial<Record<SharedFile['kind'], { view: ViewId; label: string }>> = {
  image: { view: 'vault', label: 'Secure Vault → Recover' },
  audio: { view: 'note', label: 'Hidden Note → Read a note' },
}

function ago(seconds: number): string {
  const diff = Math.max(0, Date.now() / 1000 - seconds)
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`
  return new Date(seconds * 1000).toLocaleDateString()
}

export function SharedFilesView({ onNavigate }: { onNavigate: (v: ViewId) => void }) {
  const isHost = useIsHost()
  const [files, setFiles] = useState<SharedFile[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const data = await api.listShared()
      setFiles(data.files)
      setLoadError(null)
    } catch (e) {
      setLoadError((e as Error).message)
    }
  }, [])

  // New files appear on every phone without anyone reloading the page.
  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => window.clearInterval(timer)
  }, [refresh])

  const upload = async (file: File | null) => {
    if (!file) return
    setUploading(true)
    setActionError(null)
    try {
      await api.uploadShared(file)
      await refresh()
    } catch (e) {
      setActionError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const remove = async (id: string) => {
    setActionError(null)
    try {
      await api.deleteShared(id)
      await refresh()
    } catch (e) {
      setActionError((e as Error).message)
    }
  }

  return (
    <ViewBody>
      <ViewHeader
        title="Shared Files"
        description="Files published to everyone on this network. Download one, then decrypt it yourself with the password."
      />

      {actionError && <Notice tone="danger">{actionError}</Notice>}

      <Card title="Available to download" subtitle="Updates automatically" padded={false}>
        {loadError && !files ? (
          <div className="p-4">
            <ErrorState message={loadError} onRetry={() => void refresh()} />
          </div>
        ) : files === null ? (
          <p className="p-4 text-[13px] text-muted">Loading…</p>
        ) : files.length === 0 ? (
          <div className="p-4">
            <EmptyState
              title="Nothing shared yet"
              description="Files appear here as soon as the presenter shares them — no need to reload."
            />
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {files.map((file) => {
              const Icon = KIND_ICON[file.kind]
              const opens = OPENS_IN[file.kind]
              return (
                <li key={file.id} className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center">
                  <div className="flex min-w-0 flex-1 items-center gap-3">
                    <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-primary-soft text-primary">
                      <Icon size={18} aria-hidden />
                    </span>
                    <div className="min-w-0">
                      <div className="truncate text-[14px] font-medium text-text" title={file.name}>
                        {file.name}
                      </div>
                      <div className="tnum text-[11px] text-muted">
                        {fmt.bytes(file.size)} · {ago(file.sharedAt)}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <a
                      href={api.sharedFileUrl(file.id)}
                      download={file.name}
                      className="inline-flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-primary bg-primary px-3.5 text-[13px] font-medium text-white transition-colors hover:bg-primary-strong"
                    >
                      <Download size={14} aria-hidden />
                      Download
                    </a>
                    {opens && (
                      <Button
                        size="sm"
                        className="h-9"
                        icon={<ArrowRight size={13} />}
                        onClick={() => onNavigate(opens.view)}
                      >
                        {opens.label}
                      </Button>
                    )}
                    {isHost && (
                      <Button
                        size="sm"
                        variant="danger"
                        className="h-9"
                        icon={<Trash2 size={13} />}
                        onClick={() => void remove(file.id)}
                        aria-label={`Remove ${file.name}`}
                      />
                    )}
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Notice tone="warn" title="On a phone, keep the file exactly as downloaded">
          After downloading, pick it again with <b>Browse → Choose File</b> (the Files app or
          Downloads), not from Photos. Photos, screenshots and chat apps re-compress images and
          audio, and that alone wipes out the hidden bits.
        </Notice>

        <Card title="Share a file" subtitle="Anyone on the network can add one">
          {uploading ? (
            <p className="flex items-center gap-2 text-[13px] text-muted">
              <FolderOpen size={15} aria-hidden /> Uploading…
            </p>
          ) : (
            <FileDrop
              accept=""
              label="Drop a file to share"
              hint="An encrypted PNG or a WAV with a hidden note"
              file={null}
              onFile={(f) => void upload(f)}
              compact
            />
          )}
          {isHost && (
            <div className="mt-3">
              <Badge tone="primary">Presenter</Badge>
              <span className="ml-2 text-[11px] text-muted">Only this computer can remove files.</span>
            </div>
          )}
        </Card>
      </div>
    </ViewBody>
  )
}
