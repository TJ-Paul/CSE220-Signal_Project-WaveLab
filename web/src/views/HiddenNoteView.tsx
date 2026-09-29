import { useMemo, useState } from 'react'
import { Download, EyeOff, FileAudio, KeyRound, Lock, MessageSquareLock, Unlock } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { HideNoteData, NoteSampleRow, RevealNoteData, SignalSummary, ViewId } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { FileDrop } from '../components/FileDrop'
import { MiniPlayer } from '../components/MiniPlayer'
import { ShareWithRoom } from '../components/ShareWithRoom'
import { Badge, Button, Card, Meter, Notice, PasswordInput, Stat, cx } from '../components/ui'
import { EmptyState, ViewBody, ViewHeader, useAsyncData } from '../components/ViewShell'
import { InfoGrid, TimingBars } from '../components/vault/shared'

type Mode = 'hide' | 'reveal'

const MODES: { id: Mode; label: string; blurb: string; icon: typeof Lock }[] = [
  {
    id: 'hide',
    label: 'Hide a note',
    blurb: 'Text → AES-256-GCM → password-chosen samples → LSB ±1 → WAV',
    icon: Lock,
  },
  {
    id: 'reveal',
    label: 'Read a note',
    blurb: 'WAV → password-chosen samples → LSB → AES-256-GCM → text',
    icon: Unlock,
  },
]

/** The file most recently hidden in this session, so it can be read back
 *  without a download and re-upload. */
interface SessionFile {
  id: string
  filename: string
}

export function HiddenNoteView({
  signal,
  onNavigate,
}: {
  signal: SignalSummary | null
  onNavigate: (v: ViewId) => void
}) {
  const [mode, setMode] = useState<Mode>('hide')
  const [sessionFile, setSessionFile] = useState<SessionFile | null>(null)

  return (
    <ViewBody>
      <ViewHeader
        title="Hidden Note"
        description="Encrypt a short text and hide it in the samples of a song. The WAV plays exactly like the original — only the password can tell a note is there, or read it."
      />

      <div role="tablist" aria-label="Hidden note mode" className="grid gap-2 sm:grid-cols-2">
        {MODES.map((item) => {
          const active = mode === item.id
          const Icon = item.icon
          return (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setMode(item.id)}
              className={cx(
                'flex min-w-0 cursor-pointer items-center gap-3 rounded-[14px] border px-4 py-3 text-left',
                'transition-colors duration-150',
                active
                  ? 'border-primary bg-primary-soft'
                  : 'border-border bg-surface hover:border-border-strong hover:bg-surface-2',
              )}
            >
              <span
                className={cx(
                  'grid h-9 w-9 shrink-0 place-items-center rounded-xl transition-colors',
                  active ? 'bg-primary text-white' : 'bg-surface-2 text-muted',
                )}
              >
                <Icon size={17} aria-hidden />
              </span>
              <span className="min-w-0">
                <span
                  className={cx('block text-[14px] font-semibold', active ? 'text-primary' : 'text-text')}
                >
                  {item.label}
                </span>
                <span className="block truncate text-[11px] text-muted">{item.blurb}</span>
              </span>
            </button>
          )
        })}
      </div>

      {/* Both panels stay mounted so switching modes keeps each one's result. */}
      <div hidden={mode !== 'hide'}>
        {signal ? (
          <HidePanel
            key={signal.id}
            signal={signal}
            onHidden={setSessionFile}
            onReadBack={() => setMode('reveal')}
          />
        ) : (
          <EmptyState
            title="Load a song to hide a note in"
            description="Any loaded signal can carry a note. Reading one back needs no signal — switch to Read a note and drop the WAV."
            action={
              <Button variant="primary" size="sm" onClick={() => onNavigate('dashboard')}>
                Go to dashboard
              </Button>
            }
          />
        )}
      </div>
      <div hidden={mode !== 'reveal'}>
        {/* Keyed so a freshly hidden file starts a clean read-back. */}
        <RevealPanel key={sessionFile?.id ?? 'none'} sessionFile={sessionFile} />
      </div>
    </ViewBody>
  )
}

/* -------------------------------------------------------------------------- */
/* Hide                                                                        */
/* -------------------------------------------------------------------------- */

function HidePanel({
  signal,
  onHidden,
  onReadBack,
}: {
  signal: SignalSummary
  onHidden: (file: SessionFile) => void
  onReadBack: () => void
}) {
  const { registerDerived } = useSignals()
  const capacity = useAsyncData(() => api.noteCapacity(signal.id), [signal.id])
  const [message, setMessage] = useState('')
  const [password, setPassword] = useState('')
  const [result, setResult] = useState<HideNoteData | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Capacity is in UTF-8 bytes, which is what the server counts.
  const bytes = useMemo(() => new TextEncoder().encode(message).length, [message])
  const cap = capacity.data?.capacityBytes ?? 0
  const channels = capacity.data?.channels ?? 1
  const over = bytes > cap
  const canRun = Boolean(capacity.data) && bytes > 0 && !over && password !== ''

  const run = async () => {
    if (!canRun) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const data = await api.hideNote(signal.id, { message, password })
      setResult(data)
      registerDerived(data.signal)
      onHidden({ id: data.file.id, filename: data.file.filename })
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <Card title="1 · Your note & password">
          <div className="space-y-4">
            <div>
              <label htmlFor="hidden-note-input" className="text-xs font-medium text-muted">
                Note
              </label>
              <textarea
                id="hidden-note-input"
                dir="auto"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={5}
                placeholder="Type the text to hide. English, Bangla and Hindi all work."
                className={cx(
                  'mt-1.5 block w-full resize-y rounded-xl border bg-surface-2 px-3.5 py-3 text-[14px] leading-relaxed text-text',
                  'placeholder:text-faint focus:border-primary focus:outline-none focus:ring-3 focus:ring-primary-soft',
                  over ? 'border-danger' : 'border-border',
                )}
              />
            </div>

            {capacity.data && (
              <Meter
                label="Capacity used"
                value={cap ? bytes / cap : 1}
                tone={over ? 'danger' : 'primary'}
                caption={`${bytes.toLocaleString('en-US')} of ${cap.toLocaleString('en-US')} bytes · an English letter is 1 byte, a Bangla or Hindi letter 3`}
              />
            )}
            {capacity.error && <Notice tone="danger">{capacity.error}</Notice>}
            {capacity.data && cap === 0 && (
              <Notice tone="warn">This clip is too short to carry a note. Load a longer track.</Notice>
            )}

            <PasswordInput
              label="Password"
              value={password}
              onChange={setPassword}
              onSubmit={() => void run()}
              hint="Chooses where the bits go and encrypts them. Without it, the note cannot be found."
            />

            <Button
              variant="primary"
              className="w-full"
              disabled={!canRun}
              loading={busy}
              onClick={() => void run()}
              icon={<EyeOff size={14} />}
            >
              Encrypt & hide in the song
            </Button>

            {error && (
              <Notice tone="danger" title="Could not hide the note">
                {error}
              </Notice>
            )}
          </div>
        </Card>

        <div className="space-y-4">
          <Card title="Carrier" subtitle="The audio the note is hidden in">
            <MiniPlayer signal={signal} label="Original" />
            <InfoGrid
              rows={[
                { label: 'Name', value: signal.name },
                { label: 'Duration', value: fmt.duration(signal.duration) },
                { label: 'Sample rate', value: `${(signal.sampleRate / 1000).toFixed(1)} kHz` },
                { label: 'Channels', value: String(channels) },
                { label: 'Samples', value: (signal.samples * channels).toLocaleString('en-US') },
                { label: 'Output', value: '16-bit WAV' },
              ]}
            />
          </Card>

          {signal.isLossy && (
            <Notice tone="warn" title="The source is lossy — the output is not">
              An MP3 is fine as a carrier: the note goes into a lossless WAV. Just never convert that
              WAV back to MP3, which rewrites every sample.
            </Notice>
          )}

          {!result && (
            <Notice tone="info" title="How it's hidden">
              <span className="block">
                <b>1 · Encrypt.</b> AES-256-GCM under a scrypt key — the same code as the Secure Vault.
              </span>
              <span className="mt-1 block">
                <b>2 · Scatter.</b> The password seeds a random generator that picks which samples
                carry the bits, spread over the whole song. There is no marker to find.
              </span>
              <span className="mt-1 block">
                <b>3 · LSB matching.</b> A sample whose lowest bit is wrong is nudged by ±1 at random
                — 1/32768 of full scale, about −90 dB, far below hearing.
              </span>
            </Notice>
          )}
        </div>
      </div>

      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Stat
              label="Samples changed"
              value={result.stats.samplesChanged.toLocaleString('en-US')}
              hint={`${(result.stats.changeFraction * 100).toFixed(3)}% of ${result.stats.totalSamples.toLocaleString('en-US')}`}
              tone="primary"
            />
            <Stat
              label="Largest change"
              value={`±${result.stats.maxAmplitudeChange}`}
              unit="LSB"
              hint="1/32768 of full scale ≈ −90 dB"
            />
            <Stat
              label="PSNR"
              value={result.stats.psnrDb.toFixed(1)}
              unit="dB"
              hint="original vs. with note — higher is closer"
              tone="success"
            />
            <Stat
              label="Hide time"
              value={Object.values(result.timingsMs).reduce((a, b) => a + b, 0).toFixed(0)}
              unit="ms"
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
            <Card
              title="2 · Song with the note"
              subtitle="Listen for a difference — there isn't one"
              actions={
                <a
                  href={api.vaultFileUrl(result.file.id)}
                  download={result.file.filename}
                  className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-primary bg-primary px-3 text-xs font-medium text-white transition-colors hover:bg-primary-strong"
                >
                  <Download size={13} aria-hidden />
                  Download WAV
                </a>
              }
            >
              <MiniPlayer signal={signal} label="Original" />
              <MiniPlayer signal={result.signal} label="With hidden note" tone="accent" />
              <InfoGrid
                rows={[
                  { label: 'File', value: result.file.filename },
                  { label: 'Size', value: fmt.bytes(result.file.size) },
                  { label: 'Note', value: `${result.stats.messageBytes} bytes` },
                  { label: 'Bits written', value: result.stats.samplesUsed.toLocaleString('en-US') },
                ]}
              />
              <p className="mt-2 text-[11px] text-faint">
                The file is named like the song, never like a secret. Both versions are in the
                session, so the Compare tab can measure the difference.
              </p>
              <div className="mt-3 flex flex-wrap items-start gap-2">
                <Button size="sm" icon={<KeyRound size={13} />} onClick={onReadBack}>
                  Read it back
                </Button>
                <ShareWithRoom key={result.file.id} artifactId={result.file.id} />
              </div>
            </Card>

            <Card
              title="3 · Inside the samples"
              subtitle="A few of the samples that changed, as 16-bit integers"
            >
              <SampleTable
                rows={result.changedSamples}
                channels={result.stats.channels}
                sampleRate={result.stats.sampleRate}
              />
              <p className="mt-3 text-[11px] text-faint">
                Only the <em>parity</em> of each chosen sample carries a bit. About half already have
                the right parity and are left alone; the rest move by one step, up or down at random.
                Overwriting the bit instead (LSB replacement) would pair up values in the histogram —
                a known fingerprint that matching avoids.
              </p>
            </Card>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Timing">
              <TimingBars timings={result.timingsMs} />
            </Card>
            <Notice tone="warn" title="What destroys the note">
              Anything that changes sample values: MP3/AAC encoding, resampling, normalising,
              trimming or editing. Lossless copies — the WAV itself, or a FLAC made from it — keep it.
            </Notice>
          </div>
        </>
      )}
    </div>
  )
}

/** Two's-complement low byte, with the lowest bit — the carrier — set apart. */
function LowBits({ value, changed }: { value: number; changed: boolean }) {
  const bits = (value & 0xff).toString(2).padStart(8, '0')
  return (
    <span className="tnum">
      <span className="text-faint">…{bits.slice(0, 7)}</span>
      <span
        className={cx(
          'rounded px-0.5 font-bold',
          changed ? 'bg-accent-soft text-accent' : 'bg-surface-3 text-muted',
        )}
      >
        {bits.slice(7)}
      </span>
    </span>
  )
}

function SampleTable({
  rows,
  channels,
  sampleRate,
}: {
  rows: NoteSampleRow[]
  channels: number
  sampleRate: number
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[12px]">
        <thead>
          <tr className="border-b border-border text-left text-[11px] uppercase tracking-[0.07em] text-faint">
            <th className="py-1.5 pr-3 font-medium">Time</th>
            {channels > 1 && <th className="py-1.5 pr-3 font-medium">Ch</th>}
            <th className="py-1.5 pr-3 text-right font-medium">Before</th>
            <th className="py-1.5 pr-3 text-right font-medium">After</th>
            <th className="py-1.5 pr-3 font-medium">Low bits (after)</th>
            <th className="py-1.5 text-right font-medium">Δ</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {rows.map((row) => {
            // Indices are into the interleaved stream: L R L R …
            const frame = Math.floor(row.index / channels)
            const delta = row.after - row.before
            return (
              <tr key={row.index}>
                <td className="tnum py-1.5 pr-3 text-muted">{fmt.seconds(frame / sampleRate)}</td>
                {channels > 1 && (
                  <td className="py-1.5 pr-3 text-muted">{row.index % channels === 0 ? 'L' : 'R'}</td>
                )}
                <td className="tnum py-1.5 pr-3 text-right text-muted">{row.before}</td>
                <td className="tnum py-1.5 pr-3 text-right font-semibold text-text">{row.after}</td>
                <td className="py-1.5 pr-3">
                  <LowBits value={row.after} changed={delta !== 0} />
                </td>
                <td className="tnum py-1.5 text-right text-accent">
                  {delta > 0 ? '+' : ''}
                  {delta}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Reveal                                                                      */
/* -------------------------------------------------------------------------- */

function RevealPanel({ sessionFile }: { sessionFile: SessionFile | null }) {
  const [upload, setUpload] = useState<File | null>(null)
  const [password, setPassword] = useState('')
  const [result, setResult] = useState<RevealNoteData | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const hasSource = Boolean(upload || sessionFile)
  const canRun = hasSource && password !== ''

  const onUpload = (file: File | null) => {
    setUpload(file)
    setResult(null)
    setError(null)
  }

  const run = async () => {
    if (!canRun) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      setResult(
        await api.revealNote({ password, file: upload, fileId: upload ? null : sessionFile?.id }),
      )
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
      <Card title="1 · Audio file & password">
        <div className="space-y-4">
          {sessionFile && !upload && (
            <div className="flex items-center gap-3 rounded-xl border border-accent/40 bg-accent-soft px-3 py-2.5">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-accent/15 text-accent">
                <MessageSquareLock size={15} aria-hidden />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-[13px] font-medium text-text">
                  {sessionFile.filename}
                </span>
                <span className="block text-[11px] text-muted">
                  Hidden in this session — or drop another file below
                </span>
              </span>
            </div>
          )}

          <FileDrop
            accept=".wav,.flac,audio/wav,audio/x-wav,audio/flac"
            label="Drop a WAV or FLAC"
            hint="Lossless only — an MP3 copy has already lost the note"
            file={upload}
            onFile={onUpload}
            icon={<FileAudio size={18} aria-hidden />}
          />

          <PasswordInput
            label="Password"
            value={password}
            onChange={setPassword}
            onSubmit={() => void run()}
            hint="Try a wrong one too — the answer looks the same as 'no note'"
          />

          <Button
            variant="primary"
            className="w-full"
            disabled={!canRun}
            loading={busy}
            onClick={() => void run()}
            icon={<Unlock size={14} />}
          >
            Find & decrypt the note
          </Button>

          {error && (
            <Notice tone="danger" title="No note revealed">
              {error}
            </Notice>
          )}
        </div>
      </Card>

      <div className="space-y-4">
        {result ? (
          <>
            <Card
              title="2 · The note"
              subtitle="Decrypted and authenticated — not one bit altered"
              actions={
                <Badge tone="success">
                  <KeyRound size={11} aria-hidden /> GCM tag verified
                </Badge>
              }
            >
              <p
                dir="auto"
                className="whitespace-pre-wrap break-words rounded-xl border border-success/40 bg-success-soft px-3.5 py-3 text-[15px] leading-relaxed text-text"
              >
                {result.message}
              </p>
              <div className="mt-3">
                <InfoGrid
                  rows={[
                    { label: 'Note size', value: `${result.stats.messageBytes} bytes` },
                    ...(result.stats.format ? [{ label: 'Format', value: result.stats.format }] : []),
                    {
                      label: 'Sample rate',
                      value: `${(result.stats.sampleRate / 1000).toFixed(1)} kHz`,
                    },
                    { label: 'Channels', value: String(result.stats.channels) },
                  ]}
                />
              </div>
            </Card>
            <Card title="Timing">
              <TimingBars timings={result.timingsMs} />
            </Card>
          </>
        ) : (
          <Notice tone="info" title="Why every failure looks the same">
            A wrong password, a file with no note and a damaged file all give one answer: no hidden
            note found. The decoder never admits a note exists unless it can also open it — the
            password picks the positions, so a wrong one reads random bits, and the AES-GCM tag
            rejects them. Each guess also costs a full scrypt derivation, by design.
          </Notice>
        )}
      </div>
    </div>
  )
}
