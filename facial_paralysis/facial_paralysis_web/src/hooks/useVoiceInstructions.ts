import { useCallback, useEffect, useRef, useState } from 'react'

import type { FacesProtocolStep } from '../protocol/facesProtocol'

export interface VoiceInstructionState {
  readonly speaking: boolean
  readonly countdown: number | null
  readonly error: string | null
  readonly play: (step: FacesProtocolStep) => void
  readonly cancel: () => void
}

export function useVoiceInstructions(): VoiceInstructionState {
  const [speaking, setSpeaking] = useState(false)
  const [countdown, setCountdown] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<number | null>(null)
  const generationRef = useRef(0)

  const clearCountdown = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setCountdown(null)
  }, [])

  const cancel = useCallback(() => {
    generationRef.current += 1
    window.speechSynthesis?.cancel()
    setSpeaking(false)
    setError(null)
    clearCountdown()
  }, [clearCountdown])

  const beginCountdown = useCallback((seconds: number) => {
    setCountdown(seconds)
    intervalRef.current = window.setInterval(() => {
      setCountdown((current) => {
        if (current === null || current <= 1) {
          if (intervalRef.current !== null) window.clearInterval(intervalRef.current)
          intervalRef.current = null
          return null
        }
        return current - 1
      })
    }, 1_000)
  }, [])

  const play = useCallback(
    (step: FacesProtocolStep) => {
      cancel()
      const generation = generationRef.current
      if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
        setError('Voice instructions are unavailable in this browser. Read the instruction aloud.')
        return
      }

      setError(null)
      const utterance = new window.SpeechSynthesisUtterance(step.instruction)
      utterance.rate = 0.92
      utterance.pitch = 1
      utterance.volume = 1
      utterance.onstart = () => {
        if (generationRef.current !== generation) return
        setSpeaking(true)
      }
      utterance.onend = () => {
        if (generationRef.current !== generation) return
        setSpeaking(false)
        beginCountdown(step.holdSeconds)
      }
      utterance.onerror = () => {
        if (generationRef.current !== generation) return
        setSpeaking(false)
        setError('The voice instruction could not be played. Read the instruction aloud.')
      }
      window.speechSynthesis.speak(utterance)
    },
    [beginCountdown, cancel],
  )

  useEffect(
    () => () => {
      generationRef.current += 1
      window.speechSynthesis?.cancel()
      if (intervalRef.current !== null) window.clearInterval(intervalRef.current)
      intervalRef.current = null
    },
    [],
  )

  return { speaking, countdown, error, play, cancel }
}
