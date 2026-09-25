import { Check, X } from 'lucide-react'
import { fmt } from '../../lib/format'
import type { VaultHeader, VaultPixelRow } from '../../lib/types'
import { cx } from '../ui'

/* -------------------------------------------------------------------------- */
/* Hash comparison                                                             */
/* -------------------------------------------------------------------------- */

/** SHA-256 digests shown in full — a truncated hash proves nothing. */
export function HashCompare({
  expected,
  recovered,
  matches,
}: {
  expected: string
  recovered: string
  matches: boolean
}) {
  return (
    <div
      className={cx(
        'rounded-xl border px-3.5 py-3',
        matches ? 'border-success/40 bg-success-soft' : 'border-danger/40 bg-danger-soft',
      )}
    >
      <div className="mb-2.5 flex items-center gap-2">
        <span
          className={cx(
            'grid h-6 w-6 shrink-0 place-items-center rounded-full',
            matches ? 'bg-success text-white' : 'bg-danger text-white',
          )}
        >
          {matches ? <Check size={14} aria-hidden /> : <X size={14} aria-hidden />}
        </span>
        <span className="text-[13px] font-semibold text-text">
          {matches ? 'Files are identical — bit-for-bit' : 'Digests differ — recovery failed'}
        </span>
      </div>
      <dl className="space-y-1.5">
        <HashRow label="Original" value={expected} />
        <HashRow label="Recovered" value={recovered} />
      </dl>
    </div>
  )
}

function HashRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-3">
      <dt className="w-20 shrink-0 text-[11px] uppercase tracking-[0.07em] text-faint">{label}</dt>
      <dd className="tnum min-w-0 break-all text-[11px] leading-relaxed text-text">{value}</dd>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Header table                                                                */
/* -------------------------------------------------------------------------- */

/** The cleartext preamble — everything readable without the password. */
export function HeaderTable({ header }: { header: VaultHeader }) {
  const rows: { label: string; value: string; hint?: string }[] = [
    { label: 'Magic', value: header.magic, hint: 'identifies the container format' },
    { label: 'Version', value: String(header.version) },
    {
      label: 'Key derivation',
      value: `${header.kdf} · N=${header.scryptN.toLocaleString('en-US')}, r=${header.scryptR}, p=${header.scryptP}`,
      hint: `memory-hard, about ${Math.round((128 * header.scryptN * header.scryptR) / 1e6)} MB per guess`,
    },
    { label: 'Salt', value: header.saltHex, hint: '128-bit, random per encryption' },
    { label: 'Nonce', value: header.nonceHex, hint: '96-bit, GCM’s native width' },
    {
      label: 'Payload length',
      value: `${header.payloadLength.toLocaleString('en-US')} bytes`,
      hint: `includes the ${header.tagSize}-byte authentication tag`,
    },
  ]

  return (
    <dl className="divide-y divide-border">
      {rows.map((row) => (
        <div key={row.label} className="flex flex-col gap-0.5 py-2 sm:flex-row sm:gap-3">
          <dt className="w-32 shrink-0">
            <span className="text-[12px] text-muted">{row.label}</span>
          </dt>
          <dd className="min-w-0 flex-1">
            <span className="tnum block break-all text-[12px] text-text">{row.value}</span>
            {row.hint && <span className="block text-[11px] text-faint">{row.hint}</span>}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/* -------------------------------------------------------------------------- */
/* LSB pixel demonstration                                                     */
/* -------------------------------------------------------------------------- */

const CHANNEL_NAMES = ['R', 'G', 'B'] as const
const CHANNEL_TONE = ['text-danger', 'text-success', 'text-primary'] as const

/**
 * One pixel, before and after embedding, with the modified bits marked.
 *
 * The whole technique is visible in a single row: the leading bits are
 * untouched, the trailing bit carries the payload, and the decimal value
 * moves by at most one step out of 256.
 */
export function PixelRow({ row, bitsPerChannel }: { row: VaultPixelRow; bitsPerChannel: number }) {
  return (
    <div className="rounded-xl border border-border bg-surface-2 px-3 py-2.5">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <span className="tnum text-[11px] font-semibold text-muted">
          pixel #{row.index.toLocaleString('en-US')}
        </span>
        <span className="tnum text-[11px] text-faint">
          x={row.x}, y={row.y}
        </span>
      </div>

      <div className="space-y-1.5">
        {CHANNEL_NAMES.map((name, c) => {
          const before = row.beforeBits[c]
          const after = row.afterBits[c]
          const keep = 8 - bitsPerChannel
          return (
            <div key={name} className="flex items-center gap-2 text-[11px]">
              <span className={cx('tnum w-3 shrink-0 font-bold', CHANNEL_TONE[c])}>{name}</span>
              <span className="tnum w-8 shrink-0 text-right text-faint">{row.before[c]}</span>
              <span className="tnum text-muted">
                {before.slice(0, keep)}
                <span className="text-faint">{before.slice(keep)}</span>
              </span>
              <span className="shrink-0 text-faint">→</span>
              <span className="tnum text-text">
                {after.slice(0, keep)}
                <span
                  className={cx(
                    'rounded px-0.5 font-bold',
                    row.changed[c] ? 'bg-accent-soft text-accent' : 'bg-surface-3 text-muted',
                  )}
                >
                  {after.slice(keep)}
                </span>
              </span>
              <span className="tnum w-8 shrink-0 text-right font-semibold text-text">
                {row.after[c]}
              </span>
              <span className="tnum w-7 shrink-0 text-right text-faint">
                {row.after[c] === row.before[c] ? '—' : `${row.after[c] > row.before[c] ? '+' : ''}${row.after[c] - row.before[c]}`}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Timing breakdown                                                            */
/* -------------------------------------------------------------------------- */

const STAGE_LABELS: Record<string, string> = {
  hash: 'SHA-256 of source',
  keyDerivation: 'Key derivation (scrypt)',
  encryption: 'AES-256-GCM encrypt',
  coverPreparation: 'Cover image',
  embedding: 'LSB embedding',
  pngEncode: 'PNG encode',
  extraction: 'LSB extraction',
  decryption: 'AES-256-GCM decrypt',
  verification: 'Integrity check',
}

/** Where the time actually went — scrypt should dominate, by design. */
export function TimingBars({ timings }: { timings: Record<string, number> }) {
  const entries = Object.entries(timings)
  const total = entries.reduce((sum, [, v]) => sum + v, 0) || 1
  const peak = Math.max(...entries.map(([, v]) => v), 0.01)

  return (
    <div className="space-y-2">
      {entries.map(([stage, ms]) => (
        <div key={stage}>
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-[12px] text-muted">{STAGE_LABELS[stage] ?? stage}</span>
            <span className="tnum text-[12px] font-semibold text-text">{ms.toFixed(1)} ms</span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-3">
            <div
              className={cx('h-full rounded-full', stage === 'keyDerivation' ? 'bg-accent' : 'bg-primary')}
              style={{ width: `${(ms / peak) * 100}%` }}
            />
          </div>
        </div>
      ))}
      <div className="flex items-baseline justify-between border-t border-border pt-2">
        <span className="text-[12px] font-medium text-text">Total</span>
        <span className="tnum text-[13px] font-semibold text-text">{total.toFixed(1)} ms</span>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */

export function InfoGrid({ rows }: { rows: { label: string; value: string }[] }) {
  return (
    <dl className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
      {rows.map((row) => (
        <div key={row.label} className="flex items-baseline justify-between gap-2 border-b border-border pb-1.5">
          <dt className="text-[12px] text-muted">{row.label}</dt>
          <dd className="tnum text-[12px] font-medium text-text">{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function audioInfoRows(audio: {
  sampleRate?: number
  channels?: number
  durationSeconds?: number
  format?: string
  bitDepth?: number | null
  bitrateKbps?: number
  isLossy?: boolean
}): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = []
  if (audio.format) rows.push({ label: 'Format', value: audio.format })
  if (audio.sampleRate) rows.push({ label: 'Sample rate', value: `${(audio.sampleRate / 1000).toFixed(1)} kHz` })
  if (audio.channels) rows.push({ label: 'Channels', value: String(audio.channels) })
  if (audio.durationSeconds) rows.push({ label: 'Duration', value: fmt.duration(audio.durationSeconds) })
  if (audio.bitDepth) rows.push({ label: 'Bit depth', value: `${audio.bitDepth}-bit` })
  if (audio.bitrateKbps) {
    rows.push({
      label: 'Bitrate',
      value: `${audio.bitrateKbps.toFixed(0)} kbps${audio.isLossy ? ' (lossy)' : ''}`,
    })
  }
  return rows
}
