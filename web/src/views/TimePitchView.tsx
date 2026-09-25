import { useState } from 'react'
import { CheckCircle2, HelpCircle, Play } from 'lucide-react'
import { api } from '../lib/api'
import { fmt } from '../lib/format'
import type { PitchData, SignalSummary, SpeedData, Verification } from '../lib/types'
import { useSignals } from '../state/SignalContext'
import { MiniPlayer } from '../components/MiniPlayer'
import { ResultCard } from '../components/ResultCard'
import { Badge, Button, Card, Notice, SegmentedControl, Slider, Stat, Toggle, cx } from '../components/ui'
import { ViewBody, ViewHeader } from '../components/ViewShell'

/** Below this the shift is inaudible on a sustained tone, so the transform
 *  did what it claimed. Roughly 5–10 cents is the threshold of hearing. */
const VERIFIED_TOLERANCE_CENTS = 25

const SEMITONE_NAMES = ['unison', 'minor 2nd', 'major 2nd', 'minor 3rd', 'major 3rd', 'perfect 4th',
  'tritone', 'perfect 5th', 'minor 6th', 'major 6th', 'minor 7th', 'major 7th', 'octave']

/** Signed cents with no negative zero — a measured −0.04 should read as
 *  "0.0 cents", not as a shift that happens to be downward. */
function signedCents(value: number, digits = 1): string {
  const rounded = Number(value.toFixed(digits))
  const sign = rounded > 0 ? '+' : rounded < 0 ? '−' : ''
  return `${sign}${Math.abs(rounded).toFixed(digits)} cents`
}

function intervalName(semitones: number): string {
  const abs = Math.abs(Math.round(semitones))
  if (abs > 12) return `${abs} semitones`
  const name = SEMITONE_NAMES[abs]
  if (abs === 0) return name
  return `${name} ${semitones > 0 ? 'up' : 'down'}`
}

export function TimePitchView({ signal }: { signal: SignalSummary }) {
  const { registerDerived } = useSignals()
  const [mode, setMode] = useState<'speed' | 'pitch'>('speed')
  const [rate, setRate] = useState(1.25)
  const [preservePitch, setPreservePitch] = useState(true)
  const [semitones, setSemitones] = useState(2)
  const [speedResult, setSpeedResult] = useState<SpeedData | null>(null)
  const [pitchResult, setPitchResult] = useState<PitchData | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      if (mode === 'speed') {
        const data = await api.speed(signal.id, { rate, preservePitch })
        registerDerived(data.result)
        setSpeedResult(data)
      } else {
        const data = await api.pitch(signal.id, { semitones })
        registerDerived(data.result)
        setPitchResult(data)
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  const result = mode === 'speed' ? speedResult : pitchResult
  const projectedDuration = mode === 'speed' ? signal.duration / rate : signal.duration

  return (
    <ViewBody>
      <ViewHeader
        title="Speed & Pitch"
        description="Change how fast it plays without changing the notes, or change the notes without changing the timing — then measure which actually happened."
        actions={
          <Button variant="primary" loading={running} onClick={() => void run()} icon={<Play size={14} />}>
            {mode === 'speed' ? `Apply ${rate.toFixed(2)}×` : `Shift ${semitones > 0 ? '+' : ''}${semitones} st`}
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.35fr)]">
        <Card title="Transform">
          <div className="space-y-4">
            <SegmentedControl
              label="What to change"
              value={mode}
              options={[
                { value: 'speed', label: 'Speed' },
                { value: 'pitch', label: 'Pitch' },
              ]}
              onChange={setMode}
            />

            {mode === 'speed' ? (
              <>
                <Slider
                  label="Playback rate"
                  value={rate}
                  min={0.5}
                  max={2}
                  step={0.05}
                  onChange={setRate}
                  format={(v) => `${v.toFixed(2)}×`}
                  hint={rate > 1 ? 'Faster and shorter' : rate < 1 ? 'Slower and longer' : 'Unchanged'}
                />
                <Toggle
                  label="Preserve pitch"
                  checked={preservePitch}
                  onChange={setPreservePitch}
                  hint={
                    preservePitch
                      ? 'Phase vocoder — retimes the STFT, notes stay put'
                      : 'Resampling — varispeed, pitch scales with the rate'
                  }
                />
              </>
            ) : (
              <>
                <Slider
                  label="Transpose"
                  value={semitones}
                  min={-12}
                  max={12}
                  step={1}
                  onChange={setSemitones}
                  format={(v) => `${v > 0 ? '+' : ''}${v} st`}
                  hint={intervalName(semitones)}
                />
                <div className="rounded-xl border border-border bg-surface-2 px-3 py-2.5">
                  <div className="text-[11px] font-medium uppercase tracking-[0.07em] text-faint">
                    Frequency ratio
                  </div>
                  <div className="tnum mt-0.5 text-[15px] font-semibold text-text">
                    2<sup className="text-[10px]">{semitones}/12</sup> ={' '}
                    {Math.pow(2, semitones / 12).toFixed(4)}×
                  </div>
                  <p className="mt-1 text-[11px] text-faint">
                    Twelve equal steps must multiply to an octave, so each one is a ratio of
                    2<sup>1/12</sup>. Duration is unaffected.
                  </p>
                </div>
              </>
            )}
          </div>

          <div className="mt-4">
            <MiniPlayer signal={signal} label="Source" />
          </div>
        </Card>

        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <Stat label="Source length" value={fmt.duration(signal.duration)} />
            <Stat
              label={result ? 'Result length' : 'Projected length'}
              value={fmt.duration(result ? result.resultDuration : projectedDuration)}
              tone="primary"
            />
            <Stat
              label="Expected shift"
              value={
                mode === 'speed'
                  ? preservePitch
                    ? '0'
                    : (1200 * Math.log2(rate)).toFixed(0)
                  : (semitones * 100).toFixed(0)
              }
              unit="cents"
              tone="accent"
            />
          </div>

          {result ? (
            <VerificationCard verification={result.verification} />
          ) : (
            <Card title="Verification">
              <div className="flex items-start gap-3 py-2">
                <HelpCircle size={18} className="mt-0.5 shrink-0 text-faint" aria-hidden />
                <p className="text-[13px] text-muted">
                  Run a transform and its effect on pitch will be measured here, not asserted. The
                  measurement slides the result’s log-frequency spectrum across the source’s and
                  reads off the offset — a transposition multiplies every frequency by the same
                  ratio, which is a <em>translation</em> on a log axis.
                </p>
              </div>
            </Card>
          )}

          <Notice tone="info" title={mode === 'speed' ? 'What the phase vocoder costs' : 'How pitch shifting works'}>
            {mode === 'speed' ? (
              <>
                Retiming the STFT smears transients across the frames they were stretched over, and
                can loosen the phase alignment between harmonics — heard as a faint chorusing. Both
                grow with the stretch factor, so 0.8×–1.25× is near-transparent while 0.5× or 2×
                announces itself. Resampling has no such artefacts, because nothing is estimated —
                it just moves the pitch too.
              </>
            ) : (
              <>
                Stretch by the pitch ratio with the phase vocoder (pitch unchanged, duration wrong),
                then resample by the same ratio (both scale). The duration errors cancel and the
                pitch shifts cleanly — which is why this inherits the vocoder’s artefacts.
              </>
            )}
          </Notice>

          {error && <Notice tone="danger">{error}</Notice>}
        </div>
      </div>

      {result && (
        <ResultCard
          signal={result.result}
          title={
            mode === 'speed'
              ? `${(result as SpeedData).rate.toFixed(2)}× — ${(result as SpeedData).method}`
              : `Transposed ${semitones > 0 ? '+' : ''}${(result as PitchData).semitones} semitones`
          }
          tone="accent"
        />
      )}
    </ViewBody>
  )
}

/* -------------------------------------------------------------------------- */

function VerificationCard({ verification: v }: { verification: Verification }) {
  const measured = v.measuredCents
  const error = v.errorCents
  const verified = error != null && Math.abs(error) <= VERIFIED_TOLERANCE_CENTS

  return (
    <Card
      title="Verification"
      subtitle="Measured from the audio, not assumed from the settings"
      actions={
        error == null ? (
          <Badge tone="muted">Not measurable</Badge>
        ) : verified ? (
          <Badge tone="success">
            <CheckCircle2 size={11} aria-hidden /> Verified
          </Badge>
        ) : (
          <Badge tone="accent">Off target</Badge>
        )
      }
    >
      <dl className="divide-y divide-border">
        <Row label="Expected shift" value={signedCents(v.expectedCents, 0)} />
        <Row
          label="Measured shift"
          value={measured == null ? '—' : signedCents(measured)}
          strong
        />
        <Row
          label="Error"
          value={error == null ? '—' : signedCents(error)}
          tone={verified ? 'success' : error == null ? undefined : 'accent'}
        />
        <Row
          label="Match confidence"
          value={v.confidence == null ? '—' : v.confidence.toFixed(3)}
          hint="peak of the normalised spectral cross-correlation"
        />
        {v.f0Stable && v.sourceF0 != null && v.resultF0 != null ? (
          <Row
            label="Fundamental (F0)"
            value={`${v.sourceF0.toFixed(1)} → ${v.resultF0.toFixed(1)} Hz`}
            hint="YIN autocorrelation pitch estimate"
          />
        ) : (
          <Row
            label="Fundamental (F0)"
            value="no stable pitch"
            hint={
              v.f0SpreadCents != null
                ? `F0 spread ${v.f0SpreadCents.toFixed(0)} cents — a chord or dense mix has no single note to measure`
                : 'unpitched material'
            }
          />
        )}
      </dl>

      <p className="mt-3 text-[11px] text-faint">
        {verified
          ? `Within ${VERIFIED_TOLERANCE_CENTS} cents of the target — below what the ear resolves on a sustained tone.`
          : 'The spectral method needs no pitch to be present, so it works on music, speech and noise alike.'}
      </p>
    </Card>
  )
}

function Row({
  label,
  value,
  hint,
  strong,
  tone,
}: {
  label: string
  value: string
  hint?: string
  strong?: boolean
  tone?: 'success' | 'accent'
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-2">
      <dt className="min-w-0">
        <span className="text-[12px] text-muted">{label}</span>
        {hint && <span className="mt-0.5 block text-[11px] text-faint">{hint}</span>}
      </dt>
      <dd
        className={cx(
          'tnum shrink-0 text-right',
          strong ? 'text-[15px] font-semibold' : 'text-[13px]',
          tone === 'success' ? 'text-success' : tone === 'accent' ? 'text-accent' : 'text-text',
        )}
      >
        {value}
      </dd>
    </div>
  )
}
