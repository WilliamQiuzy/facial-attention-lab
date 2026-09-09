import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { buildGuidedProtocolPlan, useGuidedVoiceSequence } from './useGuidedVoiceSequence'

interface MockUtteranceInstance {
  text: string
  rate: number
  pitch: number
  volume: number
  onstart: (() => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
}

describe('guided voice sequence', () => {
  const utterances: MockUtteranceInstance[] = []
  const speak = vi.fn((utterance: MockUtteranceInstance) => {
    utterances.push(utterance)
    utterance.onstart?.()
  })
  const cancelSpeech = vi.fn()

  beforeEach(() => {
    vi.useFakeTimers()
    utterances.length = 0
    speak.mockClear()
    cancelSpeech.mockClear()

    class MockUtterance implements MockUtteranceInstance {
      rate = 1
      pitch = 1
      volume = 1
      onstart: (() => void) | null = null
      onend: (() => void) | null = null
      onerror: (() => void) | null = null

      constructor(readonly text: string) {}
    }

    Object.defineProperty(window, 'SpeechSynthesisUtterance', {
      configurable: true,
      value: MockUtterance,
    })
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: { speak, cancel: cancelSpeech },
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('runs steps 1 through 7 in order and completes after each three-second hold', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(false))

    expect(result.current.phase).toBe('speaking')
    expect(result.current.activeStepIndex).toBe(0)
    expect(utterances[0].text).toContain('Keep your face relaxed')

    for (let stepIndex = 0; stepIndex < 7; stepIndex += 1) {
      act(() => utterances[stepIndex].onend?.())
      expect(result.current.phase).toBe('holding')
      expect(result.current.countdown).toBe(3)

      act(() => vi.advanceTimersByTime(3_000))

      if (stepIndex < 6) {
        expect(result.current.phase).toBe('speaking')
        expect(result.current.activeStepIndex).toBe(stepIndex + 1)
      }
    }

    expect(speak).toHaveBeenCalledTimes(7)
    expect(result.current.phase).toBe('complete')
    expect(result.current.completedStepIndexes).toEqual([0, 1, 2, 3, 4, 5, 6])
  })

  it('builds an immutable eight-step plan only when reanimated smile applies', () => {
    expect(buildGuidedProtocolPlan(false).map((entry) => entry.stepIndex)).toEqual([0, 1, 2, 3, 4, 5, 6])
    expect(buildGuidedProtocolPlan(true).map((entry) => entry.stepIndex)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
  })

  it('speaks the conditional eighth step when reanimated smile applies', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(true))
    for (let stepIndex = 0; stepIndex < 8; stepIndex += 1) {
      act(() => utterances[stepIndex].onend?.())
      act(() => vi.advanceTimersByTime(3_000))
    }

    expect(speak).toHaveBeenCalledTimes(8)
    expect(utterances[7].text).toContain('reanimation surgery')
    expect(result.current.completedStepIndexes).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    expect(result.current.phase).toBe('complete')
  })

  it('does not advance a hold before its monotonic three-second deadline', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(false))
    act(() => utterances[0].onend?.())
    act(() => vi.advanceTimersByTime(2_999))

    expect(result.current.phase).toBe('holding')
    expect(result.current.activeStepIndex).toBe(0)
    expect(speak).toHaveBeenCalledTimes(1)

    act(() => vi.advanceTimersByTime(1))
    expect(result.current.phase).toBe('speaking')
    expect(result.current.activeStepIndex).toBe(1)
    expect(speak).toHaveBeenCalledTimes(2)
  })

  it('settles each utterance once when a browser repeats its end callback', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(false))
    act(() => {
      utterances[0].onend?.()
      utterances[0].onend?.()
    })
    act(() => vi.advanceTimersByTime(3_000))

    expect(result.current.activeStepIndex).toBe(1)
    expect(speak).toHaveBeenCalledTimes(2)
  })

  it('ignores late speech and timer callbacks after cancellation', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(true))
    const staleUtterance = utterances[0]
    act(() => result.current.cancel())
    act(() => staleUtterance.onstart?.())
    act(() => staleUtterance.onend?.())
    act(() => staleUtterance.onerror?.())
    act(() => vi.advanceTimersByTime(30_000))

    expect(cancelSpeech).toHaveBeenCalled()
    expect(speak).toHaveBeenCalledTimes(1)
    expect(result.current.phase).toBe('idle')
    expect(result.current.countdown).toBeNull()
  })

  it('fails explicitly when speech synthesis is unavailable', () => {
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: undefined,
    })
    Object.defineProperty(window, 'SpeechSynthesisUtterance', {
      configurable: true,
      value: undefined,
    })
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(false))

    expect(result.current.phase).toBe('error')
    expect(result.current.error).toMatch(/unavailable/i)
    expect(speak).not.toHaveBeenCalled()
  })

  it('fails explicitly when the browser cannot construct an utterance', () => {
    Object.defineProperty(window, 'SpeechSynthesisUtterance', {
      configurable: true,
      value: class ThrowingUtterance {
        constructor() {
          throw new Error('broken speech engine')
        }
      },
    })
    const { result } = renderHook(() => useGuidedVoiceSequence())

    expect(() => act(() => result.current.start(false))).not.toThrow()
    expect(result.current.phase).toBe('error')
    expect(result.current.error).toMatch(/could not be started/i)
  })
})
