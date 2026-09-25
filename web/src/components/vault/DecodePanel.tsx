import { useCallback, useState } from 'react'
import { Download, FileCheck2, Image as ImageIcon, ShieldAlert, Unlock } from 'lucide-react'
import { api } from '../../lib/api'
import { fmt } from '../../lib/format'
import type { VaultDecodeData, VaultInspectData } from '../../lib/types'
import { useSignals } from '../../state/SignalContext'
import { useSignalPlayhead } from '../../state/PlaybackContext'
import { WaveformChart } from '../charts/WaveformChart'
import { MiniPlayer } from '../MiniPlayer'
import { FileDrop } from '../FileDrop'
import { Badge, Button, Card, Notice, PasswordInput, SegmentedControl, Stat } from '../ui'
import { ChartSkeleton, useAsyncData } from '../ViewShell'
import { HashCompare, HeaderTable, InfoGrid, TimingBars, audioInfoRows } from './shared'

export function DecodePanel() {
  const { refresh, signals, setActiveId } = useSignals()
  const [image, setImage] = useState<File | null>(null)
  const [password, setPassword] = useState('')
  const [bits, setBits] = useState(1)
  const [inspection, setInspection] = useState<VaultInspectData | null>(null)
  const [inspectError, setInspectError] = useState<string | null>(null)
  const [result, setResult] = useState<VaultDecodeData | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /** Header detection runs on upload, with no password — it is the check
   *  that tells the user they have the right file before they type one. */
  const inspect = useCallback(async (file: File | null, bitsPerChannel: number) => {
    setInspection(null)
    setInspectError(null)
    setResult(null)
    setError(null)
    if (!file) return
    try {
      setInspection(await api.vaultInspect(file, bitsPerChannel))
    } catch (e) {
      setInspectError((e as Error).message)
    }
  }, [])

  const onImage = (file: File | null) => {
    setImage(file)
    void inspect(file, bits)
  }

  const onBits = (value: number) => {
    setBits(value)
    void inspect(image, value)
  }

  const run = async () => {
    if (!image || !password) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const data = await api.vaultDecode({ image, password, bitsPerChannel: bits })
      setResult(data)
      await refresh()
      if (data.signalId) setActiveId(data.signalId)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const recoveredSignal = result?.signalId
    ? signals.find((s) => s.id === result.signalId) ?? null
    : null

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Card title="1 · Encrypted image & password">
          <div className="space-y-4">
            <FileDrop
              accept=".png,image/png"
              label="Drop the encrypted PNG"
              hint="Must be the original file — not a screenshot or a re-saved copy"
              file={image}
              onFile={onImage}
              icon={<ImageIcon size={18} aria-hidden />}
            />

            <PasswordInput
              label="Password"
              value={password}
              onChange={setPassword}
              onSubmit={() => image && password && void run()}
              hint="Must match the password used to encode"
            />

            <SegmentedControl
              label="Bits per colour channel"
              value={String(bits)}
              options={[
                { value: '1', label: '1 bit' },
                { value: '2', label: '2 bits' },
              ]}
              onChange={(v) => onBits(Number(v))}
            />

            <Button
              variant="primary"
              className="w-full"
              disabled={!image || !password || !inspection}
              loading={busy}
              onClick={() => void run()}
              icon={<Unlock size={14} />}
            >
              Decrypt & recover audio
            </Button>

            {error && (
              <Notice tone="danger" title="Decryption failed">
                {error}
                <span className="mt-1 block text-[11px]">
                  Nothing is returned on failure — no partial or corrupted audio is produced.
                </span>
              </Notice>
            )}
          </div>
        </Card>

        <div className="space-y-4">
          {inspectError && (
            <Notice tone="danger" title="Not a vault container">
              {inspectError}
              <span className="mt-1 block text-[11px]">
                The magic number is checked before a password is ever requested, so an ordinary
                image is rejected immediately rather than being treated as encrypted audio.
              </span>
            </Notice>
          )}

          {inspection ? (
            <Card
              title="2 · Container detected"
              subtitle="Read from the header — no password needed"
              actions={
                <Badge tone="success">
                  <FileCheck2 size={11} aria-hidden /> Valid
                </Badge>
              }
            >
              <HeaderTable header={inspection.header} />
              <div className="mt-3">
                <InfoGrid
                  rows={[
                    {
                      label: 'Image',
                      value: `${inspection.capacity.width} × ${inspection.capacity.height}`,
                    },
                    { label: 'PNG size', value: fmt.bytes(inspection.pngSize) },
                    {
                      label: 'Declared payload',
                      value: fmt.bytes(inspection.header.payloadLength),
                    },
                    {
                      label: 'Capacity used',
                      value: `${(inspection.capacity.utilization * 100).toFixed(1)}%`,
                    },
                  ]}
                />
              </div>
            </Card>
          ) : (
            !inspectError && (
              <Card title="2 · Container detection">
                <p className="py-8 text-center text-[13px] text-muted">
                  Drop an encrypted PNG and its header is parsed immediately — magic number,
                  version, key-derivation parameters, salt, nonce and payload length, all without
                  a password.
                </p>
              </Card>
            )
          )}

          {!result && !error && (
            <Notice tone="info" title="Why a wrong password cannot produce audio">
              AES-GCM verifies a 128-bit authentication tag before releasing any plaintext. A wrong
              key, a flipped bit, a cropped image or a re-saved JPEG all fail that check, and the
              decoder returns nothing at all rather than plausible-sounding noise.
            </Notice>
          )}
        </div>
      </div>

      {result && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Stat label="Recovered file" value={result.filename} />
            <Stat
              label="Size"
              value={fmt.bytes(result.recoveredSize)}
              hint={result.sizeMatches ? 'matches the declared size' : 'size mismatch'}
              tone={result.sizeMatches ? 'success' : 'accent'}
            />
            <Stat
              label="Integrity"
              value={result.integrityVerified ? 'VERIFIED' : 'FAILED'}
              tone={result.integrityVerified ? 'success' : 'accent'}
              hint="SHA-256, independent of the GCM tag"
            />
            <Stat
              label="Recovery time"
              value={Object.values(result.timingsMs).reduce((a, b) => a + b, 0).toFixed(0)}
              unit="ms"
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
            <Card
              title="3 · Recovered audio"
              subtitle="Reconstructed under its original filename"
              actions={
                <a
                  href={api.vaultFileUrl(result.fileId)}
                  download={result.filename}
                  className="inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-lg border border-primary bg-primary px-3 text-xs font-medium text-white transition-colors hover:bg-primary-strong"
                >
                  <Download size={13} aria-hidden />
                  Download {result.extension || 'file'}
                </a>
              }
            >
              <InfoGrid
                rows={[
                  { label: 'Filename', value: result.filename },
                  { label: 'Size', value: fmt.bytes(result.recoveredSize) },
                  ...audioInfoRows(result.audio),
                ]}
              />

              {recoveredSignal ? (
                <div className="mt-3">
                  <MiniPlayer signal={recoveredSignal} label="Recovered" />
                  <RecoveredWaveform signalId={recoveredSignal.id} />
                  <p className="mt-2 text-[11px] text-faint">
                    Also loaded as the active signal, so it can be analysed with the rest of the
                    workbench.
                  </p>
                </div>
              ) : (
                <Notice tone="info" title="Playback unavailable">
                  The bytes were recovered and verified, but this build cannot decode that format
                  for playback. The download is still the exact original file.
                </Notice>
              )}
            </Card>

            <div className="space-y-4">
              <Card title="4 · Bit-for-bit verification">
                <HashCompare
                  expected={result.expectedSha256}
                  recovered={result.recoveredSha256}
                  matches={result.integrityVerified}
                />
                <p className="mt-3 text-[11px] text-faint">
                  The GCM tag already proved the ciphertext was unaltered. This second, independent
                  SHA-256 comparison is what confirms the bytes were also <em>reassembled</em>
                  correctly — a different class of failure, and the one that would catch a bug in
                  the container framing rather than an attacker.
                </p>
              </Card>

              <Card title="Timing">
                <TimingBars timings={result.timingsMs} />
              </Card>
            </div>
          </div>

          {!result.integrityVerified && (
            <Notice tone="danger" title="Integrity check failed">
              <span className="flex items-start gap-2">
                <ShieldAlert size={15} className="mt-0.5 shrink-0" aria-hidden />
                The payload authenticated but the digests disagree. Do not treat this output as the
                original file.
              </span>
            </Notice>
          )}
        </>
      )}
    </div>
  )
}

function RecoveredWaveform({ signalId }: { signalId: string }) {
  const wave = useAsyncData(() => api.waveform(signalId, 1400), [signalId])
  const head = useSignalPlayhead(signalId)
  return (
    <>
      {wave.loading && <ChartSkeleton height={160} />}
      {wave.data && (
        <WaveformChart
          min={wave.data.min}
          max={wave.data.max}
          startTime={wave.data.startTime}
          endTime={wave.data.endTime}
          color="detect"
          height={160}
          {...head}
        />
      )}
    </>
  )
}
