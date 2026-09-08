import { ChevronLeft, ChevronRight, Pause, Volume2 } from 'lucide-react'
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
    active: manualActive,
    phase: manualPhase,
    countdown: manualCountdown,
    error,
    play,
    cancel,
  } = useVoiceInstructions()
  const guidedStepTotal = reanimatedSmileApplicable === true ? FACES_PROTOCOL.length : FACES_PROTOCOL.length - 1
  const previewStepIndex = Math.min(stepIndex, guidedStepTotal - 1)
  const displayedStepIndex = guidedActive && guidedVoice?.activeStepIndex !== null && guidedVoice?.activeStepIndex !== undefined
    ? guidedVoice.activeStepIndex
    : previewStepIndex
  const step = FACES_PROTOCOL[displayedStepIndex]
  const speaking = guidedActive ? guidedVoice?.phase === 'speaking' : manualSpeaking
  const countdown = guidedActive ? guidedVoice?.countdown ?? null : manualCountdown
  const releasing = (guidedActive ? guidedVoice?.phase : manualPhase) === 'releasing'

  useEffect(() => {
    if (guidedActive) cancel()
  }, [cancel, guidedActive])

  useEffect(() => {
    if (!guidedActive) cancel()
  }, [cancel, guidedActive, reanimatedSmileApplicable])

  const moveTo = (nextIndex: number) => {
    cancel()
    setStepIndex(Math.max(0, Math.min(guidedStepTotal - 1, nextIndex)))
  }

  return (
    <section className="voice-guide" aria-labelledby="voice-guide-title">
      <div className="section-heading-row">
        <div>
          <span className="eyebrow">{guidedActive ? 'Live voice guide' : 'Optional movement practice'}</span>
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
              item.optional && reanimatedSmileApplicable !== true ? 'is-skipped' : '',
            ].filter(Boolean).join(' ')}
            key={item.id}
            type="button"
            aria-label={item.optional && reanimatedSmileApplicable !== true
              ? `Optional reanimation smile not included: ${item.title}`
              : `Go to movement ${index + 1}: ${item.title}`}
            aria-current={index === displayedStepIndex ? 'step' : undefined}
            disabled={guidedActive || (item.optional && reanimatedSmileApplicable !== true)}
            onClick={() => moveTo(index)}
          >
            <span>{index + 1}</span>
          </button>
        ))}
      </div>

      <div className="instruction-stage" aria-live={guidedActive ? 'off' : 'polite'}>
        <div className="instruction-meta">
          <span>Movement {displayedStepIndex + 1} of {guidedStepTotal}</span>
          <span>{step.holdSeconds}-second hold</span>
        </div>
        <MovementAvatar action={releasing ? 'repose' : step.id} title={releasing ? 'Relaxed face' : step.title} active={speaking && !releasing} />
        <h3>{step.title}</h3>
        <p>{releasing ? step.releaseCue : countdown !== null ? 'Hold steady until you hear the release cue.' : step.actionCue}</p>
        {reanimatedSmileApplicable === null ? (
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
            disabled={previewStepIndex === 0}
            onClick={() => moveTo(previewStepIndex - 1)}
          >
            <ChevronLeft aria-hidden="true" size={18} /> Previous
          </button>
          <button
            className="button button-primary"
            type="button"
            aria-label={manualActive ? 'Stop voice preview' : 'Preview voice instruction'}
            onClick={() => (manualActive ? cancel() : play(step))}
          >
            {manualActive ? <Pause aria-hidden="true" size={18} /> : <Volume2 aria-hidden="true" size={18} />}
            {manualActive ? 'Stop preview' : 'Practice this movement'}
          </button>
          <button
            className="button button-secondary"
            type="button"
            aria-label="Next instruction"
            disabled={previewStepIndex === guidedStepTotal - 1}
            onClick={() => moveTo(previewStepIndex + 1)}
          >
            Next <ChevronRight aria-hidden="true" size={18} />
          </button>
        </div>
      )}

      {!guidedActive ? <p className="automatic-sequence-note">Optional practice · No camera or recording. Hear the cue, hold for 3 seconds, then follow the release cue.</p> : null}
      {error ? <p className="inline-alert" role="alert">{error}</p> : null}

    </section>
  )
}

export function VoiceSoundTest({ guidedActive = false }: { readonly guidedActive?: boolean }) {
  const { active, phase, error, testSound, cancel } = useVoiceInstructions()
  useEffect(() => {
    if (guidedActive) cancel()
  }, [cancel, guidedActive])

  return (
    <div className="voice-sound-test">
      <button
        className="button button-secondary"
        type="button"
        disabled={guidedActive}
        onClick={() => active ? cancel() : testSound()}
      >
        <Volume2 aria-hidden="true" size={18} /> {active ? 'Stop sound test' : 'Test sound'}
      </button>
      <p className="field-help">Optional · Does not start the camera or recording. Set a comfortable volume before continuing.</p>
      {phase === 'complete' ? <p role="status">Sound test finished. If you heard the cue clearly, your volume is ready.</p> : null}
      {error ? <p className="inline-alert" role="alert">{error}</p> : null}
    </div>
  )
}
