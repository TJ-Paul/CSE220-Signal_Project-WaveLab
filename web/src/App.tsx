import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
import { TransportBar } from './components/TransportBar'
import { EmptyState, ViewBody, ViewHeader } from './components/ViewShell'
import { Button } from './components/ui'
import type { ViewId } from './lib/types'
import { PlaybackProvider } from './state/PlaybackContext'
import { SignalProvider, useSignals } from './state/SignalContext'
import { CompareView } from './views/CompareView'
import { DashboardView } from './views/DashboardView'
import { DenoiseView } from './views/DenoiseView'
import { EditorView } from './views/EditorView'
import { FilterView } from './views/FilterView'
import { MergeView } from './views/MergeView'
import { SamplingView } from './views/SamplingView'
import { SeparationView } from './views/SeparationView'
import { SilenceView } from './views/SilenceView'
import { SpectrogramView } from './views/SpectrogramView'
import { SpectrumView } from './views/SpectrumView'
import { TimePitchView } from './views/TimePitchView'
import { VadView } from './views/VadView'
import { VaultView } from './views/VaultView'
import { VocalStudioView } from './views/VocalStudioView'
import { WaveformView } from './views/WaveformView'

type Theme = 'dark' | 'light'

/** Views that stand on their own: the dashboard loads signals, the sampling
 *  demo synthesises its own, and merge works across the whole session. */
const SIGNAL_FREE_VIEWS: ViewId[] = ['dashboard', 'sampling', 'merge', 'vault']

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem('signal-lab-theme') as Theme) || 'dark',
  )
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('signal-lab-theme', theme)
  }, [theme])
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))]
}

function Workspace() {
  const { signals, active, setActiveId, clearAll, error, clearError } = useSignals()
  const [view, setView] = useState<ViewId>('dashboard')
  const [theme, toggleTheme] = useTheme()
  const [navOpen, setNavOpen] = useState(false)

  // Escape closes the mobile drawer, matching every other dismissible surface.
  useEffect(() => {
    if (!navOpen) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setNavOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navOpen])

  // Views that need a signal fall back to the dashboard if it disappears.
  useEffect(() => {
    if (!active && !SIGNAL_FREE_VIEWS.includes(view)) setView('dashboard')
  }, [active, view])

  const navigate = useCallback((next: ViewId) => {
    setView(next)
    setNavOpen(false)
    document.getElementById('main-content')?.scrollTo({ top: 0 })
  }, [])

  return (
    <div className="flex h-screen overflow-hidden bg-bg">
      <Sidebar
        view={view}
        onNavigate={navigate}
        hasSignal={Boolean(active)}
        signalCount={signals.length}
        theme={theme}
        onToggleTheme={toggleTheme}
        open={navOpen}
        onClose={() => setNavOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TransportBar
          signals={signals}
          active={active}
          onSelect={setActiveId}
          onClear={() => void clearAll()}
          onOpenNav={() => setNavOpen(true)}
        />

        {error && (
          <div className="flex items-start gap-3 border-b border-danger/35 bg-danger-soft px-5 py-2.5">
            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-danger" aria-hidden />
            <p className="flex-1 text-[13px] text-text">{error}</p>
            <button
              type="button"
              onClick={clearError}
              aria-label="Dismiss error"
              className="cursor-pointer rounded p-0.5 text-muted transition-colors hover:text-text"
            >
              <X size={15} aria-hidden />
            </button>
          </div>
        )}

        <main id="main-content" className="flex-1 overflow-y-auto px-5 py-5" tabIndex={-1}>
          <div className="mx-auto max-w-[1400px]">
            {view === 'dashboard' && <DashboardView onNavigate={navigate} />}
            {view === 'sampling' && <SamplingView signal={active} />}
            {view === 'merge' && <MergeView signals={signals} activeId={active?.id ?? null} />}
            {view === 'vault' && <VaultView />}

            {active ? (
              <>
                {view === 'waveform' && <WaveformView key={active.id} signal={active} />}
                {view === 'spectrum' && <SpectrumView key={active.id} signal={active} />}
                {view === 'spectrogram' && <SpectrogramView key={active.id} signal={active} />}
                {view === 'vad' && <VadView key={active.id} signal={active} />}
                {view === 'filter' && <FilterView key={active.id} signal={active} />}
                {view === 'separation' && <SeparationView key={active.id} signal={active} />}
                {view === 'denoise' && <DenoiseView key={active.id} signal={active} />}
                {view === 'editor' && <EditorView key={active.id} signal={active} />}
                {view === 'silence' && <SilenceView key={active.id} signal={active} />}
                {view === 'timepitch' && <TimePitchView key={active.id} signal={active} />}
                {view === 'vocals' && <VocalStudioView key={active.id} signal={active} />}
                {view === 'compare' && <CompareView signals={signals} activeId={active.id} />}
              </>
            ) : (
              !SIGNAL_FREE_VIEWS.includes(view) && (
                <ViewBody>
                  <ViewHeader title="No signal loaded" />
                  <EmptyState
                    title="This view needs a signal"
                    description="Load an audio file or a demo signal from the dashboard to continue."
                    action={
                      <Button variant="primary" size="sm" onClick={() => navigate('dashboard')}>
                        Go to dashboard
                      </Button>
                    }
                  />
                </ViewBody>
              )
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <SignalProvider>
      <PlaybackProvider>
        <Workspace />
      </PlaybackProvider>
    </SignalProvider>
  )
}
