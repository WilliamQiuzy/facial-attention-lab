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
  readonly showApplicabilityControl?: boolean
  readonly guidedVoice?: Pick<
    GuidedVoiceSequenceState,
    'phase' | 'activeStepIndex' | 'countdown' | 'completedStepIndexes'
  >
}

interface ReanimationSmileChoiceProps {
  readonly value: boolean | null
  readonly onChange: (applicable: boolean) => void
  readonly disabled?: boolean
  readonly prominent?: boolean
}

export function ReanimationSmileChoice({
  value,
  onChange,
  disabled = false,
  prominent = false,
}: ReanimationSmileChoiceProps) {
  const applicabilityName = useId()
  const applicabilityLegendId = useId()
  const applicabilityDescriptionId = useId()

  return (
    <fieldset
      className={`applicability-control ${prominent ? 'is-prominent' : ''}`}
      disabled={disabled}
      aria-labelledby={applicabilityLegendId}
      aria-describedby={applicabilityDescriptionId}
    >
      <legend>
        <strong id={applicabilityLegendId}>Should this assessment include a reanimation smile?</strong>
        <span id={applicabilityDescriptionId}>Choose Yes only when the patient has undergone facial reanimation surgery and the surgically restored smile is part of today’s examination.</span>
      </legend>
      <div className="applicability-options">
        <label>
          <input
            type="radio"
            name={applicabilityName}
            checked={value === false}
            onChange={() => onChange(false)}
          />
          <span><strong>No — standard assessment</strong>Use the standard 7 movements without a reanimation smile.</span>
        </label>
        <label>
          <input
            type="radio"
            name={applicabilityName}
            checked={value === true}
            onChange={() => onChange(true)}
          />
          <span><strong>Yes — include reanimation smile</strong>Add one final 3-second attempt of the patient’s surgically restored smile.</span>
        </label>
      </div>
    </fieldset>
  )
}

export function VoiceGuide({
  reanimatedSmileApplicable,
  onReanimatedSmileApplicableChange,
  guidedActive = false,
  applicabilityLocked = false,
  showApplicabilityControl = true,
  guidedVoice,
}: VoiceGuideProps) {
  const [stepIndex, setStepIndex] = useState(0)
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

      {showApplicabilityControl ? (
        <ReanimationSmileChoice
          value={reanimatedSmileApplicable}
          onChange={onReanimatedSmileApplicableChange}
          disabled={guidedActive || applicabilityLocked}
        />
      ) : null}

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
              ? `Optional reanimation smile not included: ${item.title}`
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
            Choose whether to include the optional reanimation smile before analysis.
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

    </section>
  )
}
