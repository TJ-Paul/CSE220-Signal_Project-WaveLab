import { useState } from 'react'
import { Lock, Unlock } from 'lucide-react'
import { DecodePanel } from '../components/vault/DecodePanel'
import { EncodePanel } from '../components/vault/EncodePanel'
import { cx } from '../components/ui'
import { ViewBody, ViewHeader } from '../components/ViewShell'

type Mode = 'encode' | 'decode'

const MODES: { id: Mode; label: string; blurb: string; icon: typeof Lock }[] = [
  {
    id: 'encode',
    label: 'Encrypt & hide',
    blurb: 'Audio file → AES-256-GCM → LSB steganography → PNG',
    icon: Lock,
  },
  {
    id: 'decode',
    label: 'Recover',
    blurb: 'PNG → LSB extraction → AES-256-GCM → the original file',
    icon: Unlock,
  },
]

export function VaultView() {
  const [mode, setMode] = useState<Mode>('encode')

  return (
    <ViewBody>
      <ViewHeader
        title="Secure Vault"
        description="Encrypt an audio file and hide the ciphertext in the low bits of a PNG — then recover it byte for byte."
      />

      <div
        role="tablist"
        aria-label="Vault mode"
        className="grid gap-2 sm:grid-cols-2"
      >
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
                'flex cursor-pointer items-center gap-3 rounded-[14px] border px-4 py-3 text-left',
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
                  className={cx(
                    'block text-[14px] font-semibold',
                    active ? 'text-primary' : 'text-text',
                  )}
                >
                  {item.label}
                </span>
                <span className="block truncate text-[11px] text-muted">{item.blurb}</span>
              </span>
            </button>
          )
        })}
      </div>

      {mode === 'encode' ? <EncodePanel /> : <DecodePanel />}
    </ViewBody>
  )
}
