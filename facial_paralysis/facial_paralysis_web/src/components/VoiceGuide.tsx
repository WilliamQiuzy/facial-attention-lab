import { Check, ChevronLeft, ChevronRight, Pause, Volume2 } from 'lucide-react'
import { useEffect, useId, useState } from 'react'

import { FACES_PROTOCOL } from '../protocol/facesProtocol'
import type { GuidedVoiceSequenceState } from '../hooks/useGuidedVoiceSequence'
import { useVoiceInstructions } from '../hooks/useVoiceInstructions'
import { MovementAvatar } from './MovementAvatar'

interface VoiceGuideProps {
  readonly reanimatedSmileApplicable: boolean | null
  readonly onReanimatedSmileApplicableChange: (applicable: boolean) => void
  readonly guidedActive?: boolean
  readonly applicabilityLocked?: boolean
  readonly guidedVoice?: Pick<
    GuidedVoiceSequenceState,
    'phase' | 'activeStepIndex' | 'countdown' | 'completedStepIndexes'
  >
}

export function VoiceGuide({
  reanimatedSmileApplicable,
  onReanimatedSmileApplicableChange,
  guidedActive = false,
  applicabilityLocked = false,
  guidedVoice,
}: VoiceGuideProps) {
  const [stepIndex, setStepIndex] = useState(0)
  const applicabilityName = useId()
  const {
    speaking: manualSpeaking,
    countdown: manualCountdown,
    error,
    play,
    cancel,
  } = useVoiceInstructions()
  const displayedStepIndex = guidedActive && guidedVoice?.activeStepIndex !== null && guidedVoice?.activeStepIndex !== undefined
    ? guidedVoice.activeStepIndex
    : stepIndex
  const step = FACES_PROTOCOL[displayedStepIndex]
  const speaking = guidedActive ? guidedVoice?.phase === 'speaking' : manualSpeaking
  const countdown = guidedActive ? guidedVoice?.countdown ?? null : manualCountdown
  const guidedStepTotal = guidedActive && reanimatedSmileApplicable === false
    ? FACES_PROTOCOL.length - 1
    : FACES_PROTOCOL.length

  useEffect(() => {
    if (guidedActive) cancel()
  }, [cancel, guidedActive])

  const moveTo = (nextIndex: number) => {
    cancel()
    setStepIndex(Math.max(0, Math.min(FACES_PROTOCOL.length - 1, nextIndex)))
  }

  return (
    <section className="voice-guide" aria-labelledby="voice-guide-title">
      <div className="section-heading-row">
        <div>
          <span className="eyebrow">Live voice guide</span>
          <h2 id="voice-guide-title">Keep every movement consistent</h2>
        </div>
        <span className="protocol-pill">Protocol v0.01</span>
      </div>

      <div className="protocol-progress" aria-label="FACES protocol progress">
        {FACES_PROTOCOL.map((item, index) => (
          <button
            className={[
              'protocol-dot',
              index === displayedStepIndex ? 'is-active' : '',
              guidedVoice?.completedStepIndexes.includes(index) ? 'is-complete' : '',
              item.optional && reanimatedSmileApplicable === false ? 'is-skipped' : '',
            ].filter(Boolean).join(' ')}
            key={item.id}
            type="button"
            aria-label={item.optional && reanimatedSmileApplicable === false
              ? `Step ${index + 1} not applicable: ${item.title}`
              : `Go to step ${index + 1}: ${item.title}`}
            aria-current={index === displayedStepIndex ? 'step' : undefined}
            disabled={guidedActive}
            onClick={() => moveTo(index)}
          >
            <span>{index + 1}</span>
          </button>
        ))}
      </div>

      <div className="instruction-stage" aria-live={guidedActive ? 'off' : 'polite'}>
        <div className="instruction-meta">
          <span>Step {displayedStepIndex + 1} of {guidedStepTotal}</span>
          <span>{step.holdSeconds}-second hold</span>
        </div>
        <MovementAvatar action={step.id} title={step.title} active={speaking} />
        <h3>{step.title}</h3>
        <p>{step.instruction}</p>
        {step.optional && reanimatedSmileApplicable === false ? (
          <div className="not-applicable-note">
            <Check aria-hidden="true" size={17} />
            Clinician marked this conditional step as not applicable.
          </div>
        ) : null}
        {step.optional && reanimatedSmileApplicable === null ? (
          <div className="unresolved-note">
            Resolve the clinician choice below before research analysis.
          </div>
        ) : null}
        {countdown !== null ? (
          <div className="hold-countdown" aria-label={`${countdown} seconds remaining`}>
            <strong>{countdown}</strong>
            <span>hold steady</span>
          </div>
        ) : null}
      </div>

      {guidedActive ? (
        <p className="automatic-sequence-note">
          Automatic sequence · No instruction clicks are needed.
        </p>
      ) : (
        <div className="instruction-actions">
          <button
            className="button button-secondary"
            type="button"
            aria-label="Previous instruction"
            disabled={stepIndex === 0}
            onClick={() => moveTo(stepIndex - 1)}
          >
            <ChevronLeft aria-hidden="true" size={18} /> Previous
          </button>
          <button
            className="button button-primary"
            type="button"
            aria-label={speaking ? 'Stop voice preview' : 'Preview voice instruction'}
            onClick={() => (manualSpeaking ? cancel() : play(step))}
          >
            {speaking ? <Pause aria-hidden="true" size={18} /> : <Volume2 aria-hidden="true" size={18} />}
            {speaking ? 'Stop preview' : 'Preview this cue'}
          </button>
          <button
            className="button button-secondary"
            type="button"
            aria-label="Next instruction"
            disabled={stepIndex === FACES_PROTOCOL.length - 1}
            onClick={() => moveTo(stepIndex + 1)}
          >
            Next <ChevronRight aria-hidden="true" size={18} />
          </button>
        </div>
      )}

      {error ? <p className="inline-alert" role="alert">{error}</p> : null}

      <fieldset className="applicability-control" disabled={guidedActive || applicabilityLocked}>
        <legend>
          <strong>Resolve conditional step 8</strong>
          <span>Choose one option for this recording before research analysis.</span>
        </legend>
        <div className="applicability-options">
          <label>
            <input
              type="radio"
              name={applicabilityName}
              checked={reanimatedSmileApplicable === false}
              onChange={() => onReanimatedSmileApplicableChange(false)}
            />
            <span><strong>Step 8 not applicable</strong>No facial reanimation movement is expected.</span>
          </label>
          <label>
            <input
              type="radio"
              name={applicabilityName}
              checked={reanimatedSmileApplicable === true}
              onChange={() => onReanimatedSmileApplicableChange(true)}
            />
            <span><strong>Include step 8</strong>Facial reanimation surgery applies to this recording.</span>
          </label>
        </div>
      </fieldset>
    </section>
  )
}
