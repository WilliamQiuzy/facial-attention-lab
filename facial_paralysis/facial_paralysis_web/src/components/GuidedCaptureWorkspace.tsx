import { Camera, Check, CircleStop, Clock3, Volume2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useCameraRecorder } from '../hooks/useCameraRecorder'
import { useGuidedVoiceSequence } from '../hooks/useGuidedVoiceSequence'
import { FACES_PROTOCOL, type FacesActionId } from '../protocol/facesProtocol'
import {
  MediaCapturePanel,
  type CaptureMode,
  type RecordingChangeHandler,
} from './MediaCapture'
import { PatientMovementGuide, type PatientGuidePhase } from './PatientMovementGuide'
import { VoiceGuide } from './VoiceGuide'

type GuidedSessionPhase =
  | 'idle'
  | 'starting'
  | 'guiding'
  | 'finalizing'
  | 'complete'
  | 'cancelled'
  | 'error'

interface GuidedCaptureWorkspaceProps {
  readonly endpointReady?: boolean
  readonly reanimatedSmileApplicable: boolean | null
  readonly onReanimatedSmileApplicableChange: (applicable: boolean) => void
  readonly onRecordingChange: RecordingChangeHandler
}

interface CapturePlanSnapshot {
  readonly captureId: string
  readonly reanimatedSmileApplicable: boolean
  readonly actionIds: readonly FacesActionId[]
}

function snapshotCapturePlan(
  captureId: string,
  reanimatedSmileApplicable: boolean,
): CapturePlanSnapshot {
  return Object.freeze({
    captureId,
    reanimatedSmileApplicable,
    actionIds: Object.freeze(
      FACES_PROTOCOL
        .filter((step) => !step.optional || reanimatedSmileApplicable)
        .map((step) => step.id),
    ),
  })
}

export function GuidedCaptureWorkspace({
  endpointReady = true,
  reanimatedSmileApplicable,
  onReanimatedSmileApplicableChange,
  onRecordingChange,
}: GuidedCaptureWorkspaceProps) {
  const camera = useCameraRecorder()
  const voice = useGuidedVoiceSequence()
  const [mode, setMode] = useState<CaptureMode>('upload')
  const [sessionPhase, setSessionPhase] = useState<GuidedSessionPhase>('idle')
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [cancelledMessage, setCancelledMessage] = useState<string | null>(null)
  const generationRef = useRef(0)
  const runLockedRef = useRef(false)
  const voiceStartedRef = useRef(false)
  const finalizationStartedRef = useRef(false)
  const publishedCaptureRef = useRef<string | null>(null)
  const capturePlanRef = useRef<CapturePlanSnapshot | null>(null)
  const workspaceRef = useRef<HTMLDivElement | null>(null)

  const guidedActive = sessionPhase === 'starting' || sessionPhase === 'guiding' || sessionPhase === 'finalizing'

  const changeMode = useCallback((nextMode: CaptureMode) => {
    if (runLockedRef.current) return
    setMode(nextMode)
    setSessionPhase('idle')
    setSessionError(null)
    setCancelledMessage(null)
  }, [])

  const startGuidedRecording = () => {
    if (
      runLockedRef.current ||
      mode !== 'camera' ||
      camera.status !== 'ready' ||
      reanimatedSmileApplicable === null ||
      !voice.supported
    ) return

    generationRef.current += 1
    const captureId = `guided-${Date.now()}-${generationRef.current}`
    capturePlanRef.current = snapshotCapturePlan(captureId, reanimatedSmileApplicable)
    publishedCaptureRef.current = null
    voiceStartedRef.current = false
    finalizationStartedRef.current = false
    runLockedRef.current = true
    setSessionError(null)
    setCancelledMessage(null)
    setSessionPhase('starting')
    camera.startRecording()
  }

  const stopAndDiscard = useCallback(() => {
    if (!runLockedRef.current) return
    generationRef.current += 1
    runLockedRef.current = false
    voiceStartedRef.current = false
    finalizationStartedRef.current = false
    capturePlanRef.current = null
    voice.cancel()
    camera.discardRecording()
    setSessionError(null)
    setCancelledMessage('Guided recording stopped. The incomplete video was discarded.')
    setSessionPhase('cancelled')
  }, [camera.discardRecording, voice.cancel])

  useEffect(() => {
    if (
      sessionPhase === 'starting' &&
      camera.status === 'recording' &&
      camera.recordingStartedAtMs !== null &&
      !voiceStartedRef.current
    ) {
      const plan = capturePlanRef.current
      if (!plan) return
      voiceStartedRef.current = true
      voice.start(plan.reanimatedSmileApplicable, camera.recordingStartedAtMs)
      setSessionPhase('guiding')
    }
  }, [camera.recordingStartedAtMs, camera.status, sessionPhase, voice.start])

  useEffect(() => {
    if (!guidedActive) return
    const workspace = workspaceRef.current
    if (!workspace) return
    const target = window.innerWidth <= 560
      ? workspace.querySelector<HTMLElement>('.patient-guidance')
      : workspace
    target?.scrollIntoView?.({
      block: window.innerWidth <= 560 ? 'end' : 'start',
      behavior: 'auto',
    })
  }, [guidedActive])

  useEffect(() => {
    if (
      sessionPhase === 'guiding' &&
      voice.phase === 'complete' &&
      !finalizationStartedRef.current
    ) {
      finalizationStartedRef.current = true
      setSessionPhase('finalizing')
      camera.stopRecording()
    }
  }, [camera.stopRecording, sessionPhase, voice.phase])

  useEffect(() => {
    if (!runLockedRef.current) return

    if (
      camera.status === 'error' &&
      (sessionPhase === 'starting' || sessionPhase === 'guiding' || sessionPhase === 'finalizing')
    ) {
      generationRef.current += 1
      runLockedRef.current = false
      capturePlanRef.current = null
      voice.cancel()
      camera.discardRecording()
      setSessionError(camera.error ?? 'Camera recording could not continue. No video was saved.')
      setSessionPhase('error')
      return
    }

    if ((sessionPhase === 'starting' || sessionPhase === 'guiding') && voice.phase === 'error') {
      generationRef.current += 1
      runLockedRef.current = false
      const message = voice.error ?? 'Voice guidance could not continue. The incomplete recording was discarded.'
      capturePlanRef.current = null
      camera.discardRecording()
      setSessionError(message)
      setSessionPhase('error')
    }
  }, [camera.error, camera.status, camera.discardRecording, sessionPhase, voice.cancel, voice.error, voice.phase])

  useEffect(() => {
    if (
      sessionPhase !== 'finalizing' ||
      camera.status !== 'recorded' ||
      !camera.recordingFile
    ) return

    const plan = capturePlanRef.current
    if (!plan || !voice.timeline || publishedCaptureRef.current === plan.captureId) return
    publishedCaptureRef.current = plan.captureId
    runLockedRef.current = false
    onRecordingChange(camera.recordingFile, 'browser-camera', {
      preserveProtocolChoice: true,
      captureId: plan.captureId,
      actionIds: plan.actionIds,
      reanimatedSmileApplicable: plan.reanimatedSmileApplicable,
      timeline: voice.timeline,
    })
    setSessionPhase('complete')
  }, [camera.recordingFile, camera.status, onRecordingChange, sessionPhase, voice.timeline])

  useEffect(() => {
    if (sessionPhase !== 'complete' || camera.status !== 'idle') return
    capturePlanRef.current = null
    publishedCaptureRef.current = null
    setSessionPhase('idle')
  }, [camera.status, sessionPhase])

  useEffect(() => () => {
    if (!runLockedRef.current) return
    generationRef.current += 1
    runLockedRef.current = false
    voice.cancel()
    camera.discardRecording()
  }, [camera.discardRecording, voice.cancel])

  let statusText = 'Upload a LifeLink recording, or choose this device for a voice-guided capture.'
  if (mode === 'camera' && (camera.status === 'idle' || camera.status === 'error')) {
    statusText = 'Enable the front camera, then resolve Step 8 before starting.'
  } else if (mode === 'camera' && camera.status === 'requesting') {
    statusText = 'Waiting for camera permission…'
  } else if (mode === 'camera' && camera.status === 'ready' && reanimatedSmileApplicable === null) {
    statusText = 'Camera ready. Resolve Step 8 below to unlock the complete guided recording.'
  } else if (mode === 'camera' && camera.status === 'ready') {
    statusText = `Ready for ${reanimatedSmileApplicable ? 8 : 7} guided movements. Recording and voice will start together.`
  }
  if (sessionPhase === 'starting') statusText = 'Starting the video recorder before the first voice instruction…'
  if (sessionPhase === 'guiding' && voice.phase === 'speaking') {
    statusText = `Recording · Step ${(voice.activeStepIndex ?? 0) + 1} · Voice instruction playing`
  }
  if (sessionPhase === 'guiding' && voice.phase === 'holding') {
    statusText = `Recording · Step ${(voice.activeStepIndex ?? 0) + 1} · Hold steady for ${voice.countdown ?? 1} seconds`
  }
  if (sessionPhase === 'finalizing') statusText = 'All guided movements are complete. Finalizing the video…'
  if (sessionPhase === 'complete') statusText = 'Guided recording complete. Review the video below before analysis.'
  if (sessionPhase === 'cancelled' && cancelledMessage) statusText = cancelledMessage
  if (!endpointReady && !guidedActive && sessionPhase !== 'complete') {
    statusText = 'Wait for the research endpoint readiness check before starting a patient recording.'
  }

  const canStart = endpointReady && mode === 'camera' && camera.status === 'ready' && reanimatedSmileApplicable !== null && voice.supported
  const patientStepIndex = voice.activeStepIndex ?? 0
  const patientStep = FACES_PROTOCOL[patientStepIndex]
  const patientGuidePhase: PatientGuidePhase = sessionPhase === 'starting'
    ? 'starting'
    : sessionPhase === 'finalizing'
      ? 'finalizing'
      : voice.phase === 'holding'
        ? 'holding'
        : 'speaking'

  return (
    <div ref={workspaceRef} className={`workspace ${guidedActive ? 'is-guided-active' : ''}`}>
      <section className={`guided-session-control ${guidedActive ? 'is-active' : ''}`} aria-labelledby="guided-session-title">
        <div className="guided-control-copy">
          <span className="eyebrow">One guided capture</span>
          <h2 id="guided-session-title">Record and coach in one continuous flow.</h2>
          <p
            role={sessionPhase === 'complete' ? 'status' : undefined}
            aria-live={guidedActive ? 'off' : 'polite'}
          >
            {statusText}
          </p>
        </div>

        <ol className="guided-flow" aria-label="Guided recording sequence">
          <li className={camera.status === 'ready' || guidedActive || sessionPhase === 'complete' ? 'is-ready' : ''}>
            <Camera aria-hidden="true" size={22} /><span><strong>Camera</strong>Starts first</span>
          </li>
          <li className={sessionPhase === 'guiding' || sessionPhase === 'finalizing' || sessionPhase === 'complete' ? 'is-ready' : ''}>
            <Volume2 aria-hidden="true" size={22} /><span><strong>Voice cue</strong>Plays aloud</span>
          </li>
          <li className={voice.phase === 'holding' || sessionPhase === 'finalizing' || sessionPhase === 'complete' ? 'is-ready' : ''}>
            <Clock3 aria-hidden="true" size={22} /><span><strong>3-second hold</strong>Timed automatically</span>
          </li>
          <li className={sessionPhase === 'complete' ? 'is-ready' : ''}>
            <Check aria-hidden="true" size={22} /><span><strong>Auto finish</strong>Video appears below</span>
          </li>
        </ol>

        <div className="guided-control-action">
          {mode !== 'camera' ? (
            <p>Choose <strong>Use this device</strong> below to begin camera setup.</p>
          ) : camera.status === 'idle' || camera.status === 'error' ? (
            <button className="button button-primary" type="button" onClick={camera.enableCamera}>
              <Camera aria-hidden="true" size={18} /> Enable front camera
            </button>
          ) : guidedActive ? (
            <button className="button button-danger" type="button" aria-label="Stop and discard guided recording" onClick={stopAndDiscard}>
              <CircleStop aria-hidden="true" size={18} /> Stop &amp; discard
            </button>
          ) : sessionPhase === 'complete' && camera.status === 'recorded' ? (
            <span className="guided-complete-mark"><Check aria-hidden="true" size={18} /> Recording complete</span>
          ) : (
            <button className="button button-primary" type="button" disabled={!canStart} onClick={startGuidedRecording}>
              <span className="record-dot" /> Start guided recording
            </button>
          )}
        </div>
        {sessionError ? <p className="inline-alert guided-session-error" role="alert">{sessionError}</p> : null}
      </section>

      {guidedActive ? (
        <PatientMovementGuide
          step={patientStep}
          stepIndex={patientStepIndex}
          phase={patientGuidePhase}
          countdown={voice.countdown}
          completedStepIndexes={voice.completedStepIndexes}
          reanimatedSmileApplicable={reanimatedSmileApplicable === true}
        />
      ) : null}

      <MediaCapturePanel
        camera={camera}
        mode={mode}
        onModeChange={changeMode}
        onRecordingChange={onRecordingChange}
        recordingControls="guided"
        guidedActive={guidedActive}
        reportCameraRecording={false}
        showCameraError={!sessionError}
      />
      <div id="protocol" className={guidedActive ? 'is-guided-hidden' : undefined}>
        <VoiceGuide
          reanimatedSmileApplicable={reanimatedSmileApplicable}
          onReanimatedSmileApplicableChange={onReanimatedSmileApplicableChange}
          guidedActive={guidedActive}
          guidedVoice={voice}
          applicabilityLocked={mode === 'camera' && camera.status === 'recorded'}
        />
      </div>
    </div>
  )
}
