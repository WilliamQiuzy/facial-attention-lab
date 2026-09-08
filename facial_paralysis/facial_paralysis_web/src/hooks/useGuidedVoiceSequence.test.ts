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

  it('runs seven movements with action, exact hold, and audible release before advancing', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())

    act(() => result.current.start(false))

    expect(result.current.phase).toBe('speaking')
    expect(result.current.activeStepIndex).toBe(0)
    expect(utterances[0].text).toContain('Keep your face relaxed')

    for (let stepIndex = 0; stepIndex < 7; stepIndex += 1) {
      expect(utterances[stepIndex * 2].text).toMatch(/Hold now\.$/)
      expect(utterances[stepIndex * 2].text).not.toMatch(/then (open|relax)/)
      act(() => utterances[stepIndex * 2].onend?.())
      expect(result.current.phase).toBe('holding')
      expect(result.current.countdown).toBe(3)

      act(() => vi.advanceTimersByTime(3_000))
      expect(result.current.phase).toBe('releasing')
      expect(result.current.activeStepIndex).toBe(stepIndex)
      expect(utterances[stepIndex * 2 + 1].text).toBe(
        stepIndex === 2 || stepIndex === 3 ? 'Open your eyes and relax.' : 'Relax.',
      )
      act(() => utterances[stepIndex * 2 + 1].onend?.())

      if (stepIndex < 6) {
        expect(result.current.phase).toBe('speaking')
        expect(result.current.activeStepIndex).toBe(stepIndex + 1)
      }
    }

    expect(speak).toHaveBeenCalledTimes(14)
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
      act(() => utterances[stepIndex * 2].onend?.())
      act(() => vi.advanceTimersByTime(3_000))
      act(() => utterances[stepIndex * 2 + 1].onend?.())
    }

    expect(speak).toHaveBeenCalledTimes(16)
    expect(utterances[14].text).toContain('reanimated smile')
    expect(result.current.completedStepIndexes).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    expect(result.current.phase).toBe('complete')
  })

  it('publishes recording-relative prompt and exact hold timestamps', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())
    const recordingOrigin = performance.now()
    act(() => result.current.start(true, recordingOrigin))

    for (let stepIndex = 0; stepIndex < 8; stepIndex += 1) {
      act(() => vi.advanceTimersByTime(500))
      act(() => utterances[stepIndex * 2].onend?.())
      act(() => vi.advanceTimersByTime(3_000))
      act(() => vi.advanceTimersByTime(400))
      act(() => utterances[stepIndex * 2 + 1].onend?.())
    }

    expect(result.current.timeline?.actions).toHaveLength(8)
    expect(result.current.timeline?.actions[0]).toMatchObject({
      id: 'repose',
      promptStartMs: 0,
      holdStartMs: 500,
      holdEndMs: 3_500,
      completionMs: 3_900,
    })
    expect(result.current.timeline?.actions[7].id).toBe('reanimated_smile')
    expect(result.current.timeline?.recordingDurationMs).toBeGreaterThanOrEqual(
      result.current.timeline?.actions[7].completionMs ?? Infinity,
    )
    for (const row of result.current.timeline?.actions ?? []) {
      expect(row.holdEndMs - row.holdStartMs).toBe(3_000)
      expect(row.promptStartMs).toBeLessThanOrEqual(row.holdStartMs)
      expect(row.completionMs).toBeLessThanOrEqual(result.current.timeline!.recordingDurationMs)
    }
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
    expect(result.current.phase).toBe('releasing')
    expect(result.current.activeStepIndex).toBe(0)
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
    act(() => {
      utterances[1].onend?.()
      utterances[1].onend?.()
    })

    expect(result.current.activeStepIndex).toBe(1)
    expect(speak).toHaveBeenCalledTimes(3)
  })

  it('ignores late release callbacks after cancellation and does not publish a timeline', () => {
    const { result } = renderHook(() => useGuidedVoiceSequence())
    act(() => result.current.start(false))
    act(() => utterances[0].onend?.())
    act(() => vi.advanceTimersByTime(3_000))
    const staleRelease = utterances[1]
    expect(staleRelease?.text).toBe('Relax.')
    act(() => result.current.cancel())
    act(() => {
      staleRelease.onend?.()
      staleRelease.onerror?.()
      vi.advanceTimersByTime(30_000)
    })
    expect(result.current.phase).toBe('idle')
    expect(result.current.timeline).toBeNull()
    expect(speak).toHaveBeenCalledTimes(2)
  })

  it.each(['error', 'timeout'] as const)('fails safely when the release cue has a %s', (failure) => {
    const { result } = renderHook(() => useGuidedVoiceSequence())
    act(() => result.current.start(false))
    act(() => utterances[0].onend?.())
    act(() => vi.advanceTimersByTime(3_000))
    expect(utterances[1]?.text).toBe('Relax.')
    act(() => {
      if (failure === 'error') utterances[1].onerror?.()
      else vi.advanceTimersByTime(30_000)
    })
    act(() => utterances[1].onend?.())
    expect(result.current.phase).toBe('error')
    expect(result.current.error).toMatch(/release cue/i)
    expect(result.current.timeline).toBeNull()
    expect(result.current.completedStepIndexes).toEqual([])
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
