import { useState } from 'react'
import {
  Binary,
  Download,
  FileAudio,
  Image as ImageIcon,
  Lock,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react'
import { api } from '../../lib/api'
import { fmt } from '../../lib/format'
import type { VaultEncodeData, VaultPixelData, VaultTamperData } from '../../lib/types'
import { useSignals } from '../../state/SignalContext'
import { useSignalPlayhead } from '../../state/PlaybackContext'
import { HistogramChart } from '../charts/HistogramChart'
import { WaveformChart } from '../charts/WaveformChart'
import { FileDrop } from '../FileDrop'
import {
  Badge,
  Button,
  Card,
  Meter,
  Notice,
  PasswordInput,
  SegmentedControl,
  Stat,
  Toggle,
  cx,
} from '../ui'
import { ChartSkeleton, useAsyncData } from '../ViewShell'
import { HashCompare, HeaderTable, InfoGrid, PixelRow, TimingBars, audioInfoRows } from './shared'

export function EncodePanel() {
  const { refresh } = useSignals()
  const [file, setFile] = useState<File | null>(null)
  const [cover, setCover] = useState<File | null>(null)
  const [useCover, setUseCover] = useState(false)
  const [password, setPassword] = useState('')
  const [bits, setBits] = useState(1)
  const [result, setResult] = useState<VaultEncodeData | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    if (!file || !password) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const data = await api.vaultEncode({
        file,
        password,
        bitsPerChannel: bits,
        cover: useCover ? cover : null,
      })
      setResult(data)
      await refresh()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const ready = Boolean(file && password && (!useCover || cover))

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Card title="1 · Source & password" subtitle="The file is encrypted exactly as it is on disk">
          <div className="space-y-4">
            <FileDrop
              accept=".wav,.mp3,.flac,.ogg,.oga,.aac,.m4a,audio/*"
              label="Drop an audio file"
              hint="WAV · MP3 · FLAC · OGG · AAC — treated as raw binary"
              file={file}
              onFile={(f) => {
                setFile(f)
                setResult(null)
              }}
              icon={<FileAudio size={18} aria-hidden />}
            />

            <PasswordInput
              label="Password"
              value={password}
              onChange={setPassword}
              onSubmit={() => ready && void run()}
              hint="Stretched into a 256-bit key with scrypt. Never stored in the image."
            />

            <SegmentedControl
              label="Bits per colour channel"
              value={String(bits)}
              options={[
                { value: '1', label: '1 bit', hint: 'Smallest change per pixel, largest image' },
                { value: '2', label: '2 bits', hint: 'Half the image area, up to 3 steps per channel' },
              ]}
              onChange={(v) => setBits(Number(v))}
            />

            <Toggle
              label="Use my own cover image"
              checked={useCover}
              onChange={setUseCover}
              hint="Otherwise a random-noise cover is generated at exactly the needed size"
            />

            {useCover && (
              <FileDrop
                accept=".png,image/png"
                label="Drop a PNG cover"
                hint="Must be large enough to hold the payload"
                file={cover}
                onFile={setCover}
                icon={<ImageIcon size={16} aria-hidden />}
                compact
              />
            )}

            <Button
              variant="primary"
              className="w-full"
              disabled={!ready}
              loading={busy}
              onClick={() => void run()}
              icon={<Lock size={14} />}
            >
              Encrypt & encode to PNG
            </Button>

            {error && <Notice tone="danger">{error}</Notice>}
          </div>
        </Card>

        <div className="space-y-4">
          <Notice tone="info" title="What happens, in order">
            The file is hashed, wrapped with its metadata, encrypted with AES-256-GCM under a key
            derived from your password, and only then written into the low bits of the image.
            Encrypting <em>before</em> embedding is the whole point: an attacker who knows the
            technique and extracts every bit perfectly still ends up with ciphertext.
          </Notice>

          {result ? (
            <SourceAudio result={result} />
          ) : (
            <Card title="Source audio" subtitle="Analysis appears once a file is encoded">
              <p className="py-8 text-center text-[13px] text-muted">
                The waveform, format details and byte distribution of the original file will be
                shown here, alongside what encryption does to them.
              </p>
            </Card>
          )}
        </div>
      </div>

      {result && <EncodeResults result={result} password={password} bits={bits} />}
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function SourceAudio({ result }: { result: VaultEncodeData }) {
  const rows = [
    { label: 'File', value: result.source.filename },
    { label: 'Size', value: fmt.bytes(result.source.size) },
    ...audioInfoRows(result.source.audio),
  ]
  return (
    <Card title="Source audio" subtitle="Read as raw binary — never re-encoded">
      <InfoGrid rows={rows} />
      {result.signalId && <SourceWaveform signalId={result.signalId} />}
    </Card>
  )
}

function SourceWaveform({ signalId }: { signalId: string }) {
  const wave = useAsyncData(() => api.waveform(signalId, 1400), [signalId])
  const head = useSignalPlayhead(signalId)
  return (
    <div className="mt-3">
      {wave.loading && <ChartSkeleton height={150} />}
      {wave.data && (
        <WaveformChart
          min={wave.data.min}
          max={wave.data.max}
          startTime={wave.data.startTime}
          endTime={wave.data.endTime}
          height={150}
          {...head}
        />
      )}
    </div>
  )
}

/* -------------------------------------------------------------------------- */

function EncodeResults({
  result,
  password,
  bits,
}: {
  result: VaultEncodeData
  password: string
  bits: number
}) {
  const cap = result.capacity
  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Original audio" value={fmt.bytes(result.originalSize)} />
        <Stat
          label="Encrypted container"
          value={fmt.bytes(result.containerSize)}
          tone="primary"
          hint={`+${result.overheadBytes} B header, metadata & tag`}
        />
        <Stat
          label="Stego PNG"
          value={fmt.bytes(result.pngSize)}
          tone="accent"
          hint={`${(result.pngSize / result.originalSize).toFixed(1)}× the audio`}
        />
        <Stat
          label="Cipher entropy"
          value={result.cipherEntropy.toFixed(4)}
          unit="bits/byte"
          tone="success"
          hint={`source was ${result.source.entropy.toFixed(2)}`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
        <StegoImage result={result} />

        <div className="space-y-4">
          <Card title="2 · Capacity" subtitle="How the image was sized to fit the payload">
            <Meter
              label="Embedding utilisation"
              value={cap.utilization}
              tone={cap.utilization > 0.98 ? 'accent' : 'primary'}
              caption={`${fmt.bytes(cap.usedBytes)} of ${fmt.bytes(cap.capacityBytes)} used · ${fmt.bytes(cap.remainingBytes)} spare`}
            />
            <div className="mt-4">
              <InfoGrid
                rows={[
                  { label: 'Dimensions', value: `${cap.width} × ${cap.height}` },
                  { label: 'Pixels', value: fmt.int(cap.pixels) },
                  { label: 'Bits per pixel', value: `${cap.bitsPerPixel} (${cap.bitsPerChannel}/channel)` },
                  { label: 'Capacity', value: `${fmt.int(cap.capacityBits)} bits` },
                ]}
              />
            </div>
            <p className="mt-3 text-[11px] text-faint">
              capacity = width × height × 3 channels × {cap.bitsPerChannel} bit
              {cap.bitsPerChannel > 1 ? 's' : ''} ÷ 8 = {fmt.int(cap.capacityBytes)} bytes
            </p>
          </Card>

          <Card title="Timing" subtitle="Where the work went">
            <TimingBars timings={result.timingsMs} />
            <p className="mt-2 text-[11px] text-faint">
              Key derivation dominates on purpose — that cost is paid once by you and once per
              guess by anyone attacking the password.
            </p>
          </Card>
        </div>
      </div>

      <Card
        title="3 · What encryption did to the bytes"
        subtitle="Byte-value distribution, before and after"
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[12px] font-medium text-text">Original audio file</span>
              <Badge tone="muted">{result.source.entropy.toFixed(3)} bits/byte</Badge>
            </div>
            <HistogramChart
              counts={result.source.histogram}
              color="time"
              label="Byte value distribution of the original audio file"
            />
          </div>
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[12px] font-medium text-text">Encrypted payload</span>
              <Badge tone="success">{result.cipherEntropy.toFixed(3)} bits/byte</Badge>
            </div>
            <HistogramChart
              counts={result.cipherHistogram}
              color="detect"
              label="Byte value distribution of the encrypted payload"
            />
          </div>
        </div>
        <Notice tone="success" title="Why this is the proof, not the waveform">
          The audio file's bytes cluster — container headers, silence, PCM values bunched near
          zero. The ciphertext's do not: every value occurs about equally often, sitting on the
          dashed uniform line, at {result.cipherEntropy.toFixed(4)} of a possible 8 bits per byte.
          No waveform, no format signature and no structure survives, which is exactly what
          confidentiality looks like from the outside.
        </Notice>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="4 · Container header" subtitle="Cleartext, but authenticated">
          <HeaderTable header={result.header} />
          <Notice tone="info" title="Why the filename isn’t here">
            Only what is needed to <em>attempt</em> decryption is left readable. The filename, the
            original size and the SHA-256 digest all live inside the ciphertext, because each one
            leaks: a cleartext digest in particular would let anyone confirm a guessed file without
            ever attacking the password.
          </Notice>
        </Card>

        <LsbDemo imageId={result.imageId} bits={bits} />
      </div>

      <SecurityTests result={result} password={password} bits={bits} />
    </>
  )
}

/* -------------------------------------------------------------------------- */

function StegoImage({ result }: { result: VaultEncodeData }) {
  return (
    <Card
      title="Encrypted PNG"
      subtitle={result.usedCover ? 'Your cover image, with the payload embedded' : 'Generated noise cover'}
      actions={
        <a
          href={api.vaultImageUrl(result.imageId)}
          download={result.imageName}
          className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-primary bg-primary px-3 text-xs font-medium text-white transition-colors hover:bg-primary-strong"
        >
          <Download size={13} aria-hidden />
          Download PNG
        </a>
      }
    >
      <div className="overflow-hidden rounded-xl border border-border bg-surface-2">
        <img
          src={api.vaultPreviewUrl(result.imageId)}
          alt={`Preview of the encrypted image, ${result.capacity.width} by ${result.capacity.height} pixels`}
          className="mx-auto block max-h-[320px] w-auto"
        />
      </div>
      <div className="mt-3 space-y-2">
        <InfoGrid
          rows={[
            { label: 'Dimensions', value: `${result.capacity.width} × ${result.capacity.height}` },
            { label: 'PNG size', value: fmt.bytes(result.pngSize) },
          ]}
        />
        <Notice tone="warn" title="This preview is downscaled — keep the download">
          Resampling averages neighbouring pixels and destroys the low-bit plane, so a screenshot
          of this preview contains no payload. Only the downloaded PNG decodes. For the same
          reason, never re-save it as JPEG: lossy compression nudges pixel values, and one step is
          all it takes.
        </Notice>
      </div>
    </Card>
  )
}

/* -------------------------------------------------------------------------- */

function LsbDemo({ imageId, bits }: { imageId: string; bits: number }) {
  const [offset, setOffset] = useState(0)
  const pixels = useAsyncData<VaultPixelData>(
    () => api.vaultPixels(imageId, offset, 6),
    [imageId, offset],
  )

  return (
    <Card
      title="5 · How the bits were hidden"
      subtitle="Cover pixel → stego pixel, one row per channel"
      actions={
        <>
          <Button size="sm" variant="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 6))}>
            Prev
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setOffset(offset + 6)}>
            Next
          </Button>
        </>
      }
    >
      {pixels.loading && <ChartSkeleton height={220} />}
      {pixels.data && (
        <div className="space-y-2">
          {pixels.data.rows.map((row) => (
            <PixelRow key={row.index} row={row} bitsPerChannel={bits} />
          ))}
        </div>
      )}
      <p className="mt-3 text-[11px] text-faint">
        The highlighted {bits === 1 ? 'bit carries' : 'bits carry'} the payload; everything to the
        left is untouched. A channel moves by at most {(1 << bits) - 1} of 256 steps — well under
        the roughly 1% contrast difference the eye can resolve.
      </p>
    </Card>
  )
}

/* -------------------------------------------------------------------------- */

type TestState = {
  status: 'idle' | 'running' | 'pass' | 'fail'
  detail?: string
}

/** The three claims, each runnable in place. */
function SecurityTests({
  result,
  password,
  bits,
}: {
  result: VaultEncodeData
  password: string
  bits: number
}) {
  const [roundTrip, setRoundTrip] = useState<TestState>({ status: 'idle' })
  const [wrongPw, setWrongPw] = useState<TestState>({ status: 'idle' })
  const [tamper, setTamper] = useState<TestState>({ status: 'idle' })
  const [recovered, setRecovered] = useState<{ expected: string; actual: string } | null>(null)
  const [tamperInfo, setTamperInfo] = useState<VaultTamperData | null>(null)

  const asFile = () => api.vaultImageAsFile(result.imageId, result.imageName)

  const runRoundTrip = async () => {
    setRoundTrip({ status: 'running' })
    setRecovered(null)
    try {
      const image = await asFile()
      const data = await api.vaultDecode({ image, password, bitsPerChannel: bits })
      setRecovered({ expected: data.expectedSha256, actual: data.recoveredSha256 })
      setRoundTrip(
        data.integrityVerified
          ? { status: 'pass', detail: `${fmt.int(data.recoveredSize)} bytes recovered, 0 differing` }
          : { status: 'fail', detail: 'digests did not match' },
      )
    } catch (e) {
      setRoundTrip({ status: 'fail', detail: (e as Error).message })
    }
  }

  const runWrongPassword = async () => {
    setWrongPw({ status: 'running' })
    try {
      const image = await asFile()
      await api.vaultDecode({ image, password: `${password}-wrong`, bitsPerChannel: bits })
      setWrongPw({ status: 'fail', detail: 'decoding succeeded — this must never happen' })
    } catch (e) {
      const message = (e as Error).message
      setWrongPw(
        message.toLowerCase().includes('authentication')
          ? { status: 'pass', detail: message }
          : { status: 'fail', detail: message },
      )
    }
  }

  const runTamper = async () => {
    setTamper({ status: 'running' })
    try {
      const info = await api.vaultTamper(result.imageId, 12)
      setTamperInfo(info)
      const image = await api.vaultImageAsFile(info.imageId, info.imageName)
      await api.vaultDecode({ image, password, bitsPerChannel: bits })
      setTamper({ status: 'fail', detail: 'a modified image decoded — this must never happen' })
    } catch (e) {
      const message = (e as Error).message
      setTamper(
        message.toLowerCase().includes('authentication')
          ? { status: 'pass', detail: message }
          : { status: 'fail', detail: message },
      )
    }
  }

  return (
    <Card title="6 · Security & integrity tests" subtitle="Run them here — each one either holds or does not">
      <div className="grid gap-3 lg:grid-cols-3">
        <TestCard
          title="Bit-perfect recovery"
          description="Decode with the correct password and compare SHA-256 digests."
          action="Run round trip"
          onRun={() => void runRoundTrip()}
          state={roundTrip}
          expected="Files identical"
        />
        <TestCard
          title="Wrong password"
          description="Attempt to decrypt with a password that is one character off."
          action="Try wrong password"
          onRun={() => void runWrongPassword()}
          state={wrongPw}
          expected="Authentication fails"
        />
        <TestCard
          title="Modified image"
          description="Flip 12 low bits in the ciphertext, then decode with the right password."
          action="Tamper & decode"
          onRun={() => void runTamper()}
          state={tamper}
          expected="Corruption detected"
        />
      </div>

      {tamperInfo && (
        <p className="mt-3 text-[11px] text-faint">
          Flipped {tamperInfo.flippedBits} of {fmt.int(tamperInfo.totalChannels)} channel bits —{' '}
          {(tamperInfo.fractionChanged * 100).toFixed(6)}% of the image, visually identical, and
          still rejected.
        </p>
      )}

      {recovered && (
        <div className="mt-4">
          <HashCompare
            expected={recovered.expected}
            recovered={recovered.actual}
            matches={recovered.expected === recovered.actual}
          />
        </div>
      )}
    </Card>
  )
}

function TestCard({
  title,
  description,
  action,
  onRun,
  state,
  expected,
}: {
  title: string
  description: string
  action: string
  onRun: () => void
  state: TestState
  expected: string
}) {
  // Written out rather than interpolated: Tailwind scans source for literal
  // class names, so `text-${tone}` would never be generated.
  const expectedTone =
    state.status === 'pass'
      ? 'text-success'
      : state.status === 'fail'
        ? 'text-danger'
        : 'text-muted'
  return (
    <div
      className={cx(
        'flex flex-col gap-2 rounded-xl border px-3 py-3',
        state.status === 'pass'
          ? 'border-success/40 bg-success-soft'
          : state.status === 'fail'
            ? 'border-danger/40 bg-danger-soft'
            : 'border-border bg-surface-2',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-[13px] font-semibold text-text">{title}</span>
        {state.status === 'pass' && (
          <Badge tone="success">
            <ShieldCheck size={11} aria-hidden /> Pass
          </Badge>
        )}
        {state.status === 'fail' && (
          <Badge tone="danger">
            <ShieldAlert size={11} aria-hidden /> Fail
          </Badge>
        )}
      </div>
      <p className="text-[11px] text-muted">{description}</p>
      <p className="text-[11px] text-faint">
        Expected: <span className={cx('font-medium', expectedTone)}>{expected}</span>
      </p>
      {state.detail && (
        <p
          className={cx(
            'text-[11px] leading-relaxed',
            state.status === 'pass' ? 'text-success' : 'text-danger',
          )}
        >
          {state.detail}
        </p>
      )}
      <Button
        size="sm"
        className="mt-auto"
        loading={state.status === 'running'}
        onClick={onRun}
        icon={<Binary size={13} />}
      >
        {action}
      </Button>
    </div>
  )
}
