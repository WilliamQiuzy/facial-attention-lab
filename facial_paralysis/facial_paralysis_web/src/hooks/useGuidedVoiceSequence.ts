import { useCallback, useEffect, useRef, useState } from 'react'

import type { CaptureActionTimingDraft, CaptureTimelineDraft } from '../model/inference'
import { FACES_PROTOCOL, type FacesProtocolStep } from '../protocol/facesProtocol'

export type GuidedVoicePhase = 'idle' | 'speaking' | 'holding' | 'releasing' | 'complete' | 'error'

export interface GuidedProtocolPlanEntry {
  readonly stepIndex: number
  readonly step: FacesProtocolStep
}

export interface GuidedVoiceSequenceState {
  readonly supported: boolean
  readonly phase: GuidedVoicePhase
  readonly activeStepIndex: number | null
  readonly countdown: number | null
  readonly completedStepIndexes: readonly number[]
  readonly timeline: CaptureTimelineDraft | null
  readonly error: string | null
  readonly start: (reanimatedSmileApplicable: boolean, recordingStartedAtMs?: number) => void
  readonly cancel: () => void
}

const SPEECH_WATCHDOG_MS = 30_000
// Browser speech synthesis is global: release the previous preview before capture starts.
let activeVoiceOwner: { owner: symbol; recording: boolean; cancel: () => void } | null = null

interface VoiceCueRequest {
  readonly plan: readonly GuidedProtocolPlanEntry[]
  readonly recordingStartedAtMs?: number
  readonly soundTest?: boolean
}

export function buildGuidedProtocolPlan(
  reanimatedSmileApplicable: boolean,
): readonly GuidedProtocolPlanEntry[] {
  const entries = FACES_PROTOCOL
    .map((step, stepIndex) => ({ stepIndex, step }))
    .filter(({ step }) => !step.optional || reanimatedSmileApplicable)
    .map((entry) => Object.freeze(entry))
  return Object.freeze(entries)
}

function speechIsSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.speechSynthesis?.speak === 'function' &&
    typeof window.SpeechSynthesisUtterance === 'function'
  )
}

/** Shared executor keeps optional practice and recorded movement timing identical. */
export function useMovementVoiceSequence(recording: boolean) {
  const [phase, setPhase] = useState<GuidedVoicePhase>('idle')
  const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null)
  const [countdown, setCountdown] = useState<number | null>(null)
  const [completedStepIndexes, setCompletedStepIndexes] = useState<readonly number[]>([])
  const [timeline, setTimeline] = useState<CaptureTimelineDraft | null>(null)
  const [error, setError] = useState<string | null>(null)
  const generationRef = useRef(0)
  const holdTimerRef = useRef<number | null>(null)
  const speechWatchdogRef = useRef<number | null>(null)
  const timelineRowsRef = useRef<CaptureActionTimingDraft[]>([])
  const ownsSpeechRef = useRef(false)
  const ownerRef = useRef(Symbol('movement-voice'))

  const releaseOwnership = useCallback(() => {
    ownsSpeechRef.current = false
    if (activeVoiceOwner?.owner === ownerRef.current) activeVoiceOwner = null
  }, [])

  const clearTimers = useCallback(() => {
    if (holdTimerRef.current !== null) {
      window.clearTimeout(holdTimerRef.current)
      holdTimerRef.current = null
    }
    if (speechWatchdogRef.current !== null) {
      window.clearTimeout(speechWatchdogRef.current)
      speechWatchdogRef.current = null
    }
  }, [])

  const cancel = useCallback(() => {
    generationRef.current += 1
    clearTimers()
    if (ownsSpeechRef.current) window.speechSynthesis?.cancel()
    releaseOwnership()
    setPhase('idle')
    setActiveStepIndex(null)
    setCountdown(null)
    setCompletedStepIndexes([])
    timelineRowsRef.current = []
    setTimeline(null)
    setError(null)
  }, [clearTimers, releaseOwnership])

  const start = useCallback(({ plan, recordingStartedAtMs = performance.now(), soundTest = false }: VoiceCueRequest) => {
    if (!recording && activeVoiceOwner?.recording) return
    activeVoiceOwner?.cancel()
    generationRef.current += 1
    const generation = generationRef.current
    clearTimers()
    window.speechSynthesis?.cancel()
    ownsSpeechRef.current = true
    activeVoiceOwner = { owner: ownerRef.current, recording, cancel }
    setActiveStepIndex(null)
    setCountdown(null)
    setCompletedStepIndexes([])
    timelineRowsRef.current = []
    setTimeline(null)
    setError(null)

    if (!speechIsSupported()) {
      setPhase('error')
      releaseOwnership()
      setError(`Voice instructions are unavailable in this browser. ${recording ? 'Guided recording was not started.' : 'You can follow the written cues.'}`)
      return
    }

    const relativeNow = () => Math.max(0, Math.round(performance.now() - recordingStartedAtMs))
    const failureSuffix = recording ? 'The incomplete recording was discarded.' : 'Preview stopped. Try again or follow the written cues.'

    const failRun = (message: string) => {
      if (generationRef.current !== generation) return
      generationRef.current += 1
      clearTimers()
      window.speechSynthesis.cancel()
      releaseOwnership()
      setCountdown(null)
      setPhase('error')
      setError(message)
    }

    const speakCue = (text: string, cuePhase: 'speaking' | 'releasing', onComplete: () => void) => {
      if (generationRef.current !== generation) return
      setPhase(cuePhase)
      const cueName = cuePhase === 'releasing' ? 'release cue' : 'voice instruction'
      let utterance: SpeechSynthesisUtterance
      try {
        utterance = new window.SpeechSynthesisUtterance(text)
        utterance.rate = 0.92
        utterance.pitch = 1
        utterance.volume = 1
      } catch {
        failRun(`The ${cueName} could not be started. ${failureSuffix}`)
        return
      }
      let utteranceSettled = false

      utterance.onstart = () => {
        if (generationRef.current !== generation || utteranceSettled) return
        setPhase(cuePhase)
      }

      utterance.onend = () => {
        if (generationRef.current !== generation || utteranceSettled) return
        utteranceSettled = true
        if (speechWatchdogRef.current !== null) {
          window.clearTimeout(speechWatchdogRef.current)
          speechWatchdogRef.current = null
        }
        onComplete()
      }

      utterance.onerror = () => {
        if (generationRef.current !== generation || utteranceSettled) return
        utteranceSettled = true
        failRun(`The ${cueName} could not be played. ${failureSuffix}`)
      }

      speechWatchdogRef.current = window.setTimeout(() => {
        if (utteranceSettled) return
        utteranceSettled = true
        failRun(`The ${cueName} timed out. ${failureSuffix}`)
      }, SPEECH_WATCHDOG_MS)

      try {
        window.speechSynthesis.speak(utterance)
      } catch {
        failRun(`The ${cueName} could not be started. ${failureSuffix}`)
      }
    }

    const speakEntry = (planIndex: number) => {
      if (generationRef.current !== generation) return
      const entry = plan[planIndex]
      if (!entry) {
        releaseOwnership()
        setPhase('complete')
        return
      }

      setActiveStepIndex(entry.stepIndex)
      setCountdown(null)
      const promptStartMs = relativeNow()

      speakCue(entry.step.actionCue, 'speaking', () => {
        const holdStartMs = relativeNow()
        const holdEndMs = holdStartMs + entry.step.holdSeconds * 1_000
        const deadline = performance.now() + entry.step.holdSeconds * 1_000
        setPhase('holding')
        setCountdown(entry.step.holdSeconds)

        const tick = () => {
          if (generationRef.current !== generation) return
          const remainingMs = deadline - performance.now()
          if (remainingMs > 0) {
            setCountdown(Math.max(1, Math.ceil(remainingMs / 1_000)))
            holdTimerRef.current = window.setTimeout(tick, Math.min(1_000, remainingMs))
            return
          }

          holdTimerRef.current = null
          setCountdown(null)
          speakCue(entry.step.releaseCue, 'releasing', () => {
            const completionMs = Math.max(holdEndMs, relativeNow())
            timelineRowsRef.current = [
              ...timelineRowsRef.current,
              Object.freeze({
                id: entry.step.id,
                promptStartMs,
                holdStartMs,
                holdEndMs,
                completionMs,
              }),
            ]
            setCompletedStepIndexes((current) =>
              current.includes(entry.stepIndex) ? current : [...current, entry.stepIndex],
            )
            if (planIndex + 1 < plan.length) {
              speakEntry(planIndex + 1)
            } else {
              setTimeline(Object.freeze({
                recordingDurationMs: Math.max(1, completionMs),
                actions: Object.freeze([...timelineRowsRef.current]),
              }))
              setPhase('complete')
              releaseOwnership()
            }
          })
        }

        holdTimerRef.current = window.setTimeout(tick, 1_000)
      })
    }

    if (soundTest) {
      speakCue('Sound is working. During recording, follow each cue and hold until you hear relax.', 'speaking', () => {
        releaseOwnership()
        setPhase('complete')
      })
    } else {
      speakEntry(0)
    }
  }, [cancel, clearTimers, recording, releaseOwnership])

  useEffect(() => () => {
    generationRef.current += 1
    clearTimers()
    if (ownsSpeechRef.current) window.speechSynthesis?.cancel()
    releaseOwnership()
  }, [clearTimers, releaseOwnership])

  return {
    supported: speechIsSupported(),
    phase,
    activeStepIndex,
    countdown,
    completedStepIndexes,
    timeline,
    error,
    start,
    cancel,
  }
}

export function useGuidedVoiceSequence(): GuidedVoiceSequenceState {
  const sequence = useMovementVoiceSequence(true)
  const startSequence = sequence.start
  const start = useCallback((reanimatedSmileApplicable: boolean, recordingStartedAtMs?: number) => {
    startSequence({ plan: buildGuidedProtocolPlan(reanimatedSmileApplicable), recordingStartedAtMs })
  }, [startSequence])
  return { ...sequence, start }
}
