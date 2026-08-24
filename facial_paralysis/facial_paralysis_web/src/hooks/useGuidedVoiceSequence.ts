import { useCallback, useEffect, useRef, useState } from 'react'

import type { CaptureActionTimingDraft, CaptureTimelineDraft } from '../model/inference'
import { FACES_PROTOCOL, type FacesProtocolStep } from '../protocol/facesProtocol'

export type GuidedVoicePhase = 'idle' | 'speaking' | 'holding' | 'complete' | 'error'

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

export function useGuidedVoiceSequence(): GuidedVoiceSequenceState {
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
    window.speechSynthesis?.cancel()
    setPhase('idle')
    setActiveStepIndex(null)
    setCountdown(null)
    setCompletedStepIndexes([])
    timelineRowsRef.current = []
    setTimeline(null)
    setError(null)
  }, [clearTimers])

  const start = useCallback((
    reanimatedSmileApplicable: boolean,
    recordingStartedAtMs = performance.now(),
  ) => {
    generationRef.current += 1
    const generation = generationRef.current
    clearTimers()
    window.speechSynthesis?.cancel()
    setActiveStepIndex(null)
    setCountdown(null)
    setCompletedStepIndexes([])
    timelineRowsRef.current = []
    setTimeline(null)
    setError(null)

    if (!speechIsSupported()) {
      setPhase('error')
      setError('Voice instructions are unavailable in this browser. Guided recording was not started.')
      return
    }

    const plan = buildGuidedProtocolPlan(reanimatedSmileApplicable)
    const relativeNow = () => Math.max(0, Math.round(performance.now() - recordingStartedAtMs))

    const failRun = (message: string) => {
      if (generationRef.current !== generation) return
      generationRef.current += 1
      clearTimers()
      window.speechSynthesis.cancel()
      setCountdown(null)
      setPhase('error')
      setError(message)
    }

    const speakEntry = (planIndex: number) => {
      if (generationRef.current !== generation) return
      const entry = plan[planIndex]
      if (!entry) {
        setCountdown(null)
        setPhase('complete')
        return
      }

      setActiveStepIndex(entry.stepIndex)
      setCountdown(null)
      setPhase('speaking')
      const promptStartMs = relativeNow()

      let utterance: SpeechSynthesisUtterance
      try {
        utterance = new window.SpeechSynthesisUtterance(entry.step.instruction)
        utterance.rate = 0.92
        utterance.pitch = 1
        utterance.volume = 1
      } catch {
        failRun('The voice instruction could not be started. The incomplete recording was discarded.')
        return
      }
      let utteranceSettled = false

      utterance.onstart = () => {
        if (generationRef.current !== generation || utteranceSettled) return
        setPhase('speaking')
      }

      utterance.onend = () => {
        if (generationRef.current !== generation || utteranceSettled) return
        utteranceSettled = true
        if (speechWatchdogRef.current !== null) {
          window.clearTimeout(speechWatchdogRef.current)
          speechWatchdogRef.current = null
        }

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
          }
        }

        holdTimerRef.current = window.setTimeout(tick, 1_000)
      }

      utterance.onerror = () => {
        if (utteranceSettled) return
        utteranceSettled = true
        failRun('The voice instruction could not be played. The incomplete recording was discarded.')
      }

      speechWatchdogRef.current = window.setTimeout(() => {
        failRun('The voice instruction timed out. The incomplete recording was discarded.')
      }, SPEECH_WATCHDOG_MS)

      try {
        window.speechSynthesis.speak(utterance)
      } catch {
        failRun('The voice instruction could not be started. The incomplete recording was discarded.')
      }
    }

    speakEntry(0)
  }, [clearTimers])

  useEffect(() => () => {
    generationRef.current += 1
    clearTimers()
    window.speechSynthesis?.cancel()
  }, [clearTimers])

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
