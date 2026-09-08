import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { FACES_PROTOCOL } from '../protocol/facesProtocol'
import { useVoiceInstructions } from './useVoiceInstructions'
import { useGuidedVoiceSequence } from './useGuidedVoiceSequence'

class MockUtterance {
  onstart: (() => void) | null = null
  onend: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(readonly text: string) {}
}

describe('optional voice practice', () => {
  const utterances: MockUtterance[] = []
  const cancelSpeech = vi.fn()
  beforeEach(() => {
    vi.useFakeTimers()
    utterances.length = 0
    cancelSpeech.mockClear()
    Object.defineProperty(window, 'SpeechSynthesisUtterance', { configurable: true, value: MockUtterance })
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: { cancel: cancelSpeech, speak: (utterance: MockUtterance) => {
        utterances.push(utterance)
        utterance.onstart?.()
      } },
    })
  })
  afterEach(() => vi.useRealTimers())

  it('practices the same action, exact three-second hold, and release without starting another movement', () => {
    const { result } = renderHook(() => useVoiceInstructions())
    act(() => result.current.play(FACES_PROTOCOL[2]))
    expect(utterances[0].text).toBe(FACES_PROTOCOL[2].actionCue)
    act(() => utterances[0].onend?.())
    expect(result.current.countdown).toBe(3)
    act(() => vi.advanceTimersByTime(2_999))
    expect(utterances).toHaveLength(1)
    act(() => vi.advanceTimersByTime(1))
    expect(utterances[1].text).toBe('Open your eyes and relax.')
    expect(result.current.phase).toBe('releasing')
    act(() => utterances[1].onend?.())
    expect(result.current.phase).toBe('complete')
    expect(result.current.active).toBe(false)
    expect(utterances).toHaveLength(2)
  })

  it('ignores stale preview end callbacks after cancellation', () => {
    const { result } = renderHook(() => useVoiceInstructions())
    act(() => result.current.play(FACES_PROTOCOL[0]))
    act(() => result.current.cancel())
    act(() => {
      utterances[0].onend?.()
      utterances[0].onerror?.()
      vi.advanceTimersByTime(30_000)
    })
    expect(result.current.phase).toBe('idle')
    expect(result.current.error).toBeNull()
    expect(result.current.countdown).toBeNull()
    expect(utterances).toHaveLength(1)
  })

  it('stops practice on release failure without claiming a recording was discarded', () => {
    const { result } = renderHook(() => useVoiceInstructions())
    act(() => result.current.play(FACES_PROTOCOL[0]))
    act(() => utterances[0].onend?.())
    act(() => vi.advanceTimersByTime(3_000))
    expect(utterances[1]?.text).toBe('Relax.')
    act(() => utterances[1].onerror?.())
    act(() => utterances[1].onend?.())
    expect(result.current.phase).toBe('error')
    expect(result.current.error).toMatch(/release cue/i)
    expect(result.current.error).not.toMatch(/recording|discarded/i)
  })

  it('plays the optional sound check without starting a hold', () => {
    const { result } = renderHook(() => useVoiceInstructions())
    act(() => result.current.testSound())
    expect(utterances[0].text).toMatch(/sound is working/i)
    act(() => utterances[0].onend?.())
    act(() => vi.advanceTimersByTime(30_000))
    expect(result.current.countdown).toBeNull()
    expect(result.current.phase).toBe('complete')
    expect(utterances).toHaveLength(1)
  })

  it('does not cancel another speech owner when an unused preview unmounts', () => {
    const { unmount } = renderHook(() => useVoiceInstructions())
    unmount()
    expect(cancelSpeech).not.toHaveBeenCalled()
  })

  it('hands off speech ownership from practice to recording and rejects stale preview callbacks', () => {
    const preview = renderHook(() => useVoiceInstructions())
    const recording = renderHook(() => useGuidedVoiceSequence())
    act(() => preview.result.current.play(FACES_PROTOCOL[2]))
    const stalePreview = utterances[0]
    act(() => recording.result.current.start(false))
    expect(preview.result.current.active).toBe(false)
    const cancellationCount = cancelSpeech.mock.calls.length
    act(() => {
      stalePreview.onend?.()
      stalePreview.onerror?.()
    })
    preview.unmount()
    expect(cancelSpeech).toHaveBeenCalledTimes(cancellationCount)
    expect(recording.result.current.phase).toBe('speaking')
    act(() => utterances[1].onend?.())
    expect(recording.result.current.phase).toBe('holding')
  })

  it('does not let an optional sound test interrupt an active recorded sequence', () => {
    const preview = renderHook(() => useVoiceInstructions())
    const recording = renderHook(() => useGuidedVoiceSequence())
    act(() => recording.result.current.start(false))
    const cancellationCount = cancelSpeech.mock.calls.length
    act(() => preview.result.current.testSound())
    expect(utterances).toHaveLength(1)
    expect(cancelSpeech).toHaveBeenCalledTimes(cancellationCount)
    expect(recording.result.current.phase).toBe('speaking')
  })
})
