import {
  Activity,
  AudioLines,
  BarChart3,
  Combine,
  Eraser,
  Filter,
  Gauge,
  GitCompare,
  Grid2x2,
  LayoutDashboard,
  Mic2,
  Moon,
  Scissors,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Split,
  Sun,
  Waves,
} from 'lucide-react'
import type { ViewId } from '../lib/types'
import { cx } from './ui'

interface NavItem {
  id: ViewId
  label: string
  icon: typeof Waves
  requiresSignal?: boolean
}

const NAV: { group: string; items: NavItem[] }[] = [
  {
    group: 'Workspace',
    items: [{ id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard }],
  },
  {
    group: 'Analysis',
    items: [
      { id: 'waveform', label: 'Waveform', icon: Waves, requiresSignal: true },
      { id: 'spectrum', label: 'FFT Spectrum', icon: BarChart3, requiresSignal: true },
      { id: 'spectrogram', label: 'Spectrogram', icon: Grid2x2, requiresSignal: true },
      { id: 'vad', label: 'Voice Activity', icon: Activity, requiresSignal: true },
      { id: 'sampling', label: 'Sampling & Aliasing', icon: Gauge },
    ],
  },
  {
    group: 'Editing',
    items: [
      { id: 'editor', label: 'Trim, Cut & Fade', icon: Scissors, requiresSignal: true },
      { id: 'silence', label: 'Silence Remover', icon: Eraser, requiresSignal: true },
      { id: 'merge', label: 'Merge', icon: Combine },
      { id: 'timepitch', label: 'Speed & Pitch', icon: SlidersHorizontal, requiresSignal: true },
    ],
  },
  {
    group: 'Processing',
    items: [
      { id: 'filter', label: 'Filtering', icon: Filter, requiresSignal: true },
      { id: 'denoise', label: 'Noise Reduction', icon: Sparkles, requiresSignal: true },
      { id: 'vocals', label: 'Karaoke & Vocals', icon: Mic2, requiresSignal: true },
      { id: 'separation', label: 'Separation (analysis)', icon: Split, requiresSignal: true },
    ],
  },
  {
    group: 'Security',
    items: [{ id: 'vault', label: 'Secure Vault', icon: ShieldCheck }],
  },
  {
    group: 'Output',
    items: [{ id: 'compare', label: 'Compare', icon: GitCompare, requiresSignal: true }],
  },
]

export function Sidebar({
  view,
  onNavigate,
  hasSignal,
  signalCount,
  theme,
  onToggleTheme,
  open,
  onClose,
}: {
  view: ViewId
  onNavigate: (v: ViewId) => void
  hasSignal: boolean
  signalCount: number
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  open: boolean
  onClose: () => void
}) {
  return (
    <>
      {/* Backdrop only exists while the drawer is open on small screens */}
      {open && (
        <button
          type="button"
          aria-label="Close navigation"
          onClick={onClose}
          className="fixed inset-0 z-30 cursor-default bg-black/55 backdrop-blur-[2px] lg:hidden"
        />
      )}
      <aside
        className={cx(
          'flex h-full w-[236px] shrink-0 flex-col border-r border-border bg-surface',
          'fixed inset-y-0 left-0 z-40 transition-transform duration-200 lg:static lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
      <div className="flex items-center gap-2.5 px-4 py-4">
        <span className="grid h-9 w-9 place-items-center rounded-xl bg-primary-soft text-primary">
          <AudioLines size={19} aria-hidden />
        </span>
        <div className="min-w-0">
          <div className="text-[15px] font-semibold leading-tight text-text">Signal Lab</div>
          <div className="text-[11px] text-faint">DSP Workbench</div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-2.5 pb-3" aria-label="Main">
        {NAV.map((section) => (
          <div key={section.group} className="mb-4">
            <div className="px-2.5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.11em] text-faint">
              {section.group}
            </div>
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const disabled = Boolean(item.requiresSignal) && !hasSignal
                const active = view === item.id
                const Icon = item.icon
                return (
                  <li key={item.id}>
                    <button
                      type="button"
                      onClick={() => !disabled && onNavigate(item.id)}
                      disabled={disabled}
                      aria-current={active ? 'page' : undefined}
                      title={disabled ? 'Load a signal first' : undefined}
                      className={cx(
                        'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium',
                        'transition-colors duration-150',
                        active
                          ? 'bg-primary-soft text-primary'
                          : disabled
                            ? 'cursor-not-allowed text-faint/55'
                            : 'cursor-pointer text-muted hover:bg-surface-2 hover:text-text',
                      )}
                    >
                      <Icon size={16} aria-hidden className="shrink-0" />
                      <span className="truncate">{item.label}</span>
                      {active && (
                        <span className="ml-auto h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
                      )}
                    </button>
                  </li>
                )
              })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="border-t border-border p-2.5">
        <div className="mb-2 flex items-center justify-between rounded-lg bg-surface-2 px-2.5 py-2">
          <span className="text-[11px] text-muted">Signals in session</span>
          <span className="tnum text-[13px] font-semibold text-text">{signalCount}</span>
        </div>
        <button
          type="button"
          onClick={onToggleTheme}
          className="flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium text-muted transition-colors hover:bg-surface-2 hover:text-text"
        >
          {theme === 'dark' ? <Sun size={16} aria-hidden /> : <Moon size={16} aria-hidden />}
          {theme === 'dark' ? 'Light theme' : 'Dark theme'}
        </button>
        </div>
      </aside>
    </>
  )
}
