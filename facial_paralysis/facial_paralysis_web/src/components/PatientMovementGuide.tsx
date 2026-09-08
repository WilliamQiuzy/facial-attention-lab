import { Check, Clock3, Volume2 } from 'lucide-react'

import { FACES_PROTOCOL, type FacesProtocolStep } from '../protocol/facesProtocol'
import { MovementAvatar } from './MovementAvatar'

export type PatientGuidePhase = 'starting' | 'speaking' | 'holding' | 'releasing' | 'finalizing'

interface PatientMovementGuideProps {
  readonly step: FacesProtocolStep
  readonly stepIndex: number
  readonly phase: PatientGuidePhase
  readonly countdown: number | null
  readonly completedStepIndexes: readonly number[]
  readonly reanimatedSmileApplicable: boolean
}

const PHASE_COPY: Record<PatientGuidePhase, string> = {
  starting: 'Get ready',
  speaking: 'Voice prompt playing',
  holding: 'Hold the pose',
  releasing: 'Release the pose',
  finalizing: 'Sequence complete',
}

export function PatientMovementGuide({
  step,
  stepIndex,
  phase,
  countdown,
  completedStepIndexes,
  reanimatedSmileApplicable,
}: PatientMovementGuideProps) {
  const visibleSteps = FACES_PROTOCOL.filter(
    (item) => !item.optional || reanimatedSmileApplicable,
  )
  const timerIsActive = phase === 'holding' && countdown !== null
  const liveStatus = phase === 'speaking'
    ? `Movement ${stepIndex + 1} of ${visibleSteps.length}. ${step.title}. ${step.actionCue}`
    : phase === 'holding'
      ? `Hold ${step.title}. ${countdown ?? step.holdSeconds} seconds remaining.`
      : phase === 'releasing'
        ? step.releaseCue
        : phase === 'finalizing'
          ? 'All movements are complete. Please relax while the video is saved.'
          : `Get ready for movement ${stepIndex + 1}. ${step.title}.`
  const writtenCue = phase === 'releasing'
    ? step.releaseCue
    : phase === 'holding'
      ? 'Hold steady until you hear the release cue.'
      : phase === 'finalizing'
        ? 'All movements are complete. Please relax.'
        : step.actionCue

  return (
    <section
      className={`patient-guidance is-${phase}`}
      role="region"
      aria-label="Patient movement guidance"
      data-phase={phase}
    >
      <p className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">
        {liveStatus}
      </p>
      <header className="patient-guidance-header">
        <span className="patient-recording-chip"><i /> Recording in progress</span>
        <span className="patient-sequence-label">
          <span>Automatic sequence</span>
          <strong>Movement {stepIndex + 1} of {visibleSteps.length}</strong>
        </span>
      </header>

      <div className="patient-guidance-body">
        <MovementAvatar
          action={phase === 'releasing' || phase === 'finalizing' ? 'repose' : step.id}
          title={phase === 'releasing' || phase === 'finalizing' ? 'Relaxed face' : step.title}
          active={phase === 'speaking'}
        />

        <div className="patient-guidance-copy">
          <span className="patient-phase-label">
            {phase === 'speaking' || phase === 'releasing' ? <Volume2 aria-hidden="true" size={19} /> : null}
            {phase === 'holding' || phase === 'starting' ? <Clock3 aria-hidden="true" size={19} /> : null}
            {phase === 'finalizing' ? <Check aria-hidden="true" size={19} /> : null}
            {PHASE_COPY[phase]}
          </span>
          <h2>{step.title}</h2>
          <p>{writtenCue}</p>
          <span className="patient-caption-note">Voice and written guidance stay synchronized.</span>
        </div>

        <div
          className={`patient-timer ${timerIsActive ? 'is-counting' : ''}`}
          aria-label={timerIsActive
            ? `${countdown} seconds remaining`
            : phase === 'finalizing'
              ? 'Recording is being finalized'
              : phase === 'releasing' ? step.releaseCue : 'Hold timer starts after Hold now'}
        >
          {timerIsActive ? (
            <>
              <strong>{countdown}</strong>
              <span>seconds<br />hold steady</span>
            </>
          ) : phase === 'releasing' ? (
            <>
              <Check aria-hidden="true" size={34} />
              <span>Relax now<br />Listen for the next cue</span>
            </>
          ) : phase === 'finalizing' ? (
            <>
              <Check aria-hidden="true" size={34} />
              <span>Please relax<br />Saving video</span>
            </>
          ) : (
            <>
              <Volume2 aria-hidden="true" size={31} />
              <span>Follow the cue<br />Hold after “Hold now”</span>
            </>
          )}
        </div>
      </div>

      <ol className="patient-step-progress" aria-label="Automatic movement progress">
        {visibleSteps.map((item) => {
          const protocolIndex = FACES_PROTOCOL.findIndex((candidate) => candidate.id === item.id)
          return (
            <li
              key={item.id}
              className={[
                protocolIndex === stepIndex ? 'is-active' : '',
                completedStepIndexes.includes(protocolIndex) ? 'is-complete' : '',
              ].filter(Boolean).join(' ')}
              aria-current={protocolIndex === stepIndex ? 'step' : undefined}
              aria-label={`${item.shortLabel}${completedStepIndexes.includes(protocolIndex) ? ', complete' : ''}`}
            >
              <span>{protocolIndex + 1}</span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
