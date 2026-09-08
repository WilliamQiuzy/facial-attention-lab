import { useCallback } from 'react'

import type { FacesProtocolStep } from '../protocol/facesProtocol'
import { useMovementVoiceSequence, type GuidedVoicePhase } from './useGuidedVoiceSequence'

export interface VoiceInstructionState {
  readonly phase: GuidedVoicePhase
  readonly active: boolean
  readonly speaking: boolean
  readonly countdown: number | null
  readonly error: string | null
  readonly play: (step: FacesProtocolStep) => void
  readonly testSound: () => void
  readonly cancel: () => void
}

export function useVoiceInstructions(): VoiceInstructionState {
  const { phase, countdown, error, start, cancel } = useMovementVoiceSequence(false)
  const play = useCallback((step: FacesProtocolStep) => {
    start({ plan: [{ stepIndex: 0, step }] })
  }, [start])
  const testSound = useCallback(() => start({ plan: [], soundTest: true }), [start])
  const speaking = phase === 'speaking' || phase === 'releasing'

  return { phase, active: speaking || phase === 'holding', speaking, countdown, error, play, testSound, cancel }
}
