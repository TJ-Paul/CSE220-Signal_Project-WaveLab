import { useId, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { Eye, EyeOff, Loader2 } from 'lucide-react'

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

/* -------------------------------------------------------------------------- */
/* Card                                                                        */
/* -------------------------------------------------------------------------- */

interface CardProps {
  children: ReactNode
  className?: string
  title?: string
  subtitle?: string
  actions?: ReactNode
  padded?: boolean
}

export function Card({ children, className, title, subtitle, actions, padded = true }: CardProps) {
  return (
    <section
      className={cx(
        // min-w-0: a card inside a grid/flex track must never widen it —
        // grid items default to min-width:auto and would otherwise size to
        // their min-content width.
        'min-w-0 rounded-[14px] border border-border bg-surface shadow-[0_1px_2px_rgb(0_0_0/0.16)]',
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold tracking-wide text-text">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={padded ? 'p-4' : ''}>{children}</div>
    </section>
  )
}

/* -------------------------------------------------------------------------- */
/* Button                                                                      */
/* -------------------------------------------------------------------------- */

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  icon?: ReactNode
  loading?: boolean
}

export function Button({
  variant = 'secondary',
  size = 'md',
  icon,
  loading,
  children,
  className,
  disabled,
  ...rest
}: ButtonProps) {
  const variants = {
    primary:
      'bg-primary text-white border-primary hover:bg-primary-strong hover:border-primary-strong',
    secondary:
      'bg-surface-2 text-text border-border hover:border-border-strong hover:bg-surface-3',
    ghost: 'bg-transparent text-muted border-transparent hover:bg-surface-2 hover:text-text',
    danger: 'bg-danger-soft text-danger border-transparent hover:border-danger',
  }
  return (
    <button
      {...rest}
      disabled={disabled || loading}
      className={cx(
        'inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg border font-medium',
        'transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50',
        size === 'sm' ? 'h-8 px-3 text-xs' : 'h-10 px-4 text-sm',
        variants[variant],
        className,
      )}
    >
      {loading ? <Loader2 size={15} className="animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  )
}

/* -------------------------------------------------------------------------- */
/* Stat                                                                        */
/* -------------------------------------------------------------------------- */

export function Stat({
  label,
  value,
  unit,
  hint,
  tone = 'default',
}: {
  label: string
  value: string
  unit?: string
  hint?: string
  tone?: 'default' | 'primary' | 'accent' | 'success'
}) {
  const tones = {
    default: 'text-text',
    primary: 'text-primary',
    accent: 'text-accent',
    success: 'text-success',
  }
  return (
    <div className="rounded-xl border border-border bg-surface-2 px-3 py-2.5">
      <div className="text-[11px] font-medium uppercase tracking-[0.07em] text-faint">{label}</div>
      <div className={cx('tnum mt-1 text-[19px] font-semibold leading-tight', tones[tone])}>
        {value}
        {unit && <span className="ml-1 text-[12px] font-normal text-muted">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-faint">{hint}</div>}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Slider                                                                      */
/* -------------------------------------------------------------------------- */

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
  hint?: string
}) {
  const id = `slider-${label.replace(/\W+/g, '-').toLowerCase()}`
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-xs font-medium text-muted">
          {label}
        </label>
        <span className="tnum text-xs font-semibold text-text">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2"
      />
      {hint && <p className="mt-1 text-[11px] text-faint">{hint}</p>}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Select                                                                      */
/* -------------------------------------------------------------------------- */

export function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  const id = `select-${label.replace(/\W+/g, '-').toLowerCase()}`
  return (
    <div>
      <label htmlFor={id} className="text-xs font-medium text-muted">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className={cx(
          'mt-1.5 h-9 w-full cursor-pointer rounded-lg border border-border bg-surface-2 px-2.5',
          'text-sm text-text transition-colors hover:border-border-strong',
        )}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Badge / Legend / Notice                                                     */
/* -------------------------------------------------------------------------- */

export function Badge({
  children,
  tone = 'muted',
}: {
  children: ReactNode
  tone?: 'muted' | 'primary' | 'accent' | 'success' | 'danger'
}) {
  const tones = {
    muted: 'bg-surface-3 text-muted',
    primary: 'bg-primary-soft text-primary',
    accent: 'bg-accent-soft text-accent',
    success: 'bg-success-soft text-success',
    danger: 'bg-danger-soft text-danger',
  }
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-semibold',
        tones[tone],
      )}
    >
      {children}
    </span>
  )
}

/** Legend entries carry a shape as well as a colour, so series stay
 *  distinguishable without relying on hue. */
export function Legend({
  items,
}: {
  items: { color: string; label: string; dash?: boolean }[]
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5 text-[11px] text-muted">
          <span
            aria-hidden
            className="h-0.5 w-4 rounded-full"
            style={
              it.dash
                ? { backgroundImage: `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)` }
                : { background: it.color }
            }
          />
          {it.label}
        </span>
      ))}
    </div>
  )
}

export function Notice({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warn' | 'success' | 'danger'
  title?: string
  children: ReactNode
}) {
  const tones = {
    info: 'border-primary/35 bg-primary-soft text-text',
    warn: 'border-accent/40 bg-accent-soft text-text',
    success: 'border-success/35 bg-success-soft text-text',
    danger: 'border-danger/40 bg-danger-soft text-text',
  }
  return (
    <div className={cx('rounded-xl border px-3.5 py-2.5 text-[13px]', tones[tone])}>
      {title && <div className="mb-0.5 font-semibold">{title}</div>}
      <div className="text-muted">{children}</div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* SegmentedControl / Toggle                                                   */
/* -------------------------------------------------------------------------- */

/** A small set of mutually exclusive choices, all visible at once — used
 *  where a <select> would hide the alternatives behind a click. */
export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string; hint?: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium text-muted">{label}</div>
      <div role="group" aria-label={label} className="flex gap-1 rounded-lg bg-surface-2 p-1">
        {options.map((o) => {
          const active = o.value === value
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              aria-pressed={active}
              title={o.hint}
              className={cx(
                'flex-1 cursor-pointer rounded-md px-2 py-1.5 text-[12px] font-medium',
                'transition-colors duration-150',
                active
                  ? 'bg-primary text-white shadow-[0_1px_3px_rgb(0_0_0/0.2)]'
                  : 'text-muted hover:bg-surface-3 hover:text-text',
              )}
            >
              {o.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export function Toggle({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  hint?: string
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx(
          'mt-0.5 h-[18px] w-8 shrink-0 cursor-pointer rounded-full p-0.5 transition-colors duration-150',
          checked ? 'bg-primary' : 'bg-surface-3',
        )}
      >
        <span
          aria-hidden
          className={cx(
            'block h-[14px] w-[14px] rounded-full bg-white transition-transform duration-150',
            checked && 'translate-x-[14px]',
          )}
        />
      </button>
      <span className="min-w-0">
        <span className="block text-xs font-medium text-text">{label}</span>
        {hint && <span className="block text-[11px] text-faint">{hint}</span>}
      </span>
    </label>
  )
}

/* -------------------------------------------------------------------------- */
/* PasswordInput / Meter                                                       */
/* -------------------------------------------------------------------------- */

export function PasswordInput({
  label,
  value,
  onChange,
  hint,
  placeholder = 'Enter a password',
  autoComplete = 'off',
  onSubmit,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  hint?: string
  placeholder?: string
  autoComplete?: string
  onSubmit?: () => void
}) {
  const id = useId()
  const [visible, setVisible] = useState(false)
  return (
    <div>
      <label htmlFor={id} className="text-xs font-medium text-muted">
        {label}
      </label>
      <div className="relative mt-1.5">
        <input
          id={id}
          type={visible ? 'text' : 'password'}
          value={value}
          autoComplete={autoComplete}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && onSubmit) onSubmit()
          }}
          className={cx(
            'h-10 w-full rounded-lg border border-border bg-surface-2 pl-3 pr-10',
            'font-mono text-sm text-text transition-colors',
            'placeholder:font-sans placeholder:text-faint hover:border-border-strong',
          )}
        />
        <button
          type="button"
          onClick={() => setVisible((v) => !v)}
          aria-label={visible ? 'Hide password' : 'Show password'}
          className="absolute right-1 top-1 grid h-8 w-8 cursor-pointer place-items-center rounded-md text-faint transition-colors hover:bg-surface-3 hover:text-text"
        >
          {visible ? <EyeOff size={15} aria-hidden /> : <Eye size={15} aria-hidden />}
        </button>
      </div>
      {hint && <p className="mt-1 text-[11px] text-faint">{hint}</p>}
    </div>
  )
}

/** Proportion bar — used for embedding capacity utilisation. */
export function Meter({
  label,
  value,
  caption,
  tone = 'primary',
}: {
  label: string
  value: number
  caption?: string
  tone?: 'primary' | 'accent' | 'success' | 'danger'
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const fills = {
    primary: 'bg-primary',
    accent: 'bg-accent',
    success: 'bg-success',
    danger: 'bg-danger',
  }
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs font-medium text-muted">{label}</span>
        <span className="tnum text-xs font-semibold text-text">{pct.toFixed(1)}%</span>
      </div>
      <div
        className="mt-1.5 h-2 overflow-hidden rounded-full bg-surface-3"
        role="progressbar"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div className={cx('h-full rounded-full transition-all duration-300', fills[tone])} style={{ width: `${pct}%` }} />
      </div>
      {caption && <p className="mt-1 text-[11px] text-faint">{caption}</p>}
    </div>
  )
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium text-muted">{label}</div>
      {children}
    </div>
  )
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h3 className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.09em] text-faint">
      {children}
    </h3>
  )
}
