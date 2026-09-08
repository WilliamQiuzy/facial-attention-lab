import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { VoiceGuide, VoiceSoundTest } from './VoiceGuide'
import { PatientMovementGuide } from './PatientMovementGuide'
import { FACES_PROTOCOL } from '../protocol/facesProtocol'

describe('VoiceGuide', () => {
  const speak = vi.fn()
  const cancel = vi.fn()

  beforeEach(() => {
    speak.mockClear()
    cancel.mockClear()

    class MockUtterance {
      text: string
      rate = 1
      onstart: (() => void) | null = null
      onend: (() => void) | null = null
      onerror: (() => void) | null = null

      constructor(text: string) {
        this.text = text
      }
    }

    Object.defineProperty(window, 'SpeechSynthesisUtterance', {
      configurable: true,
      value: MockUtterance,
    })
    Object.defineProperty(window, 'speechSynthesis', {
      configurable: true,
      value: { speak, cancel },
    })
  })

  afterEach(() => vi.useRealTimers())

  it('navigates through the selected eight-movement protocol', async () => {
    const user = userEvent.setup()
    render(
      <VoiceGuide
        reanimatedSmileApplicable
        onReanimatedSmileApplicableChange={vi.fn()}
      />,
    )

    expect(screen.getByText('Movement 1 of 8')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Neutral Expression (Repose)' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next instruction' }))
    expect(screen.getByText('Movement 2 of 8')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Eyebrow Raise' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Previous instruction' }))
    expect(screen.getByText('Movement 1 of 8')).toBeInTheDocument()
  })

  it('keeps excluded movements unclickable and limits preview navigation to seven movements', async () => {
    const user = userEvent.setup()
    render(<VoiceGuide reanimatedSmileApplicable={false} onReanimatedSmileApplicableChange={vi.fn()} />)
    expect(screen.getByText('Movement 1 of 7')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /go to movement 7/i }))
    expect(screen.getByText('Movement 7 of 7')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next instruction' })).toBeDisabled()
    const excluded = screen.getByRole('button', { name: /optional reanimation smile not included/i })
    expect(excluded).toBeDisabled()
    await user.click(excluded)
    expect(screen.getByRole('heading', { name: 'Lower Teeth Show' })).toBeInTheDocument()
  })

  it('clamps preview selection and stops playback when reanimation smile is excluded', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<VoiceGuide reanimatedSmileApplicable onReanimatedSmileApplicableChange={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: /go to movement 8/i }))
    await user.click(screen.getByRole('button', { name: 'Preview voice instruction' }))
    rerender(<VoiceGuide reanimatedSmileApplicable={false} onReanimatedSmileApplicableChange={vi.fn()} />)
    expect(screen.getByText('Movement 7 of 7')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Lower Teeth Show' })).toBeInTheDocument()
    expect(cancel).toHaveBeenCalled()
  })

  it('keeps the stop control available throughout practice holding and release', () => {
    vi.useFakeTimers()
    render(<VoiceGuide reanimatedSmileApplicable={false} onReanimatedSmileApplicableChange={vi.fn()} />)
    fireEvent.click(screen.getByRole('button', { name: 'Preview voice instruction' }))
    act(() => speak.mock.calls[0][0].onend?.())
    expect(screen.getByRole('button', { name: 'Stop voice preview' })).toBeEnabled()
    act(() => vi.advanceTimersByTime(3_000))
    expect(screen.getByText('Relax.')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Relaxed face movement demonstration' })).toHaveAttribute('data-action', 'repose')
    expect(screen.getByRole('button', { name: 'Stop voice preview' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: 'Stop voice preview' }))
    expect(screen.getByRole('button', { name: 'Preview voice instruction' })).toBeEnabled()
  })

  it('offers a nonblocking sound test and prevents it while guided capture is active', () => {
    const { rerender } = render(<VoiceSoundTest />)
    expect(screen.getByText(/optional.*does not start.*camera/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Test sound' }))
    expect(speak.mock.calls[0][0].text).toMatch(/sound is working/i)
    rerender(<VoiceSoundTest guidedActive />)
    expect(screen.getByRole('button', { name: 'Test sound' })).toBeDisabled()
    expect(cancel).toHaveBeenCalled()
  })

  it('shows the release cue instead of the action cue in patient guidance', () => {
    render(<PatientMovementGuide step={FACES_PROTOCOL[2]} stepIndex={2} phase="releasing" countdown={null} completedStepIndexes={[0, 1]} reanimatedSmileApplicable={false} />)
    expect(screen.getByText('Movement 3 of 7')).toBeInTheDocument()
    expect(screen.getAllByText('Open your eyes and relax.')).toHaveLength(2)
    expect(screen.getByRole('img', { name: 'Relaxed face movement demonstration' })).toHaveAttribute('data-action', 'repose')
    expect(screen.queryByText(FACES_PROTOCOL[2].instruction)).not.toBeInTheDocument()
  })

  it('offers an optional preview before recording', async () => {
    const user = userEvent.setup()
    render(
      <VoiceGuide
        reanimatedSmileApplicable={null}
        onReanimatedSmileApplicableChange={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Preview voice instruction' }))
    expect(speak).toHaveBeenCalledTimes(1)
    expect(speak.mock.calls[0][0].text).toContain('Keep your face relaxed')
  })

  it('does not report a playback error when the clinician stops a preview', async () => {
    const user = userEvent.setup()
    render(
      <VoiceGuide
        reanimatedSmileApplicable={null}
        onReanimatedSmileApplicableChange={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Preview voice instruction' }))
    const utterance = speak.mock.calls[0][0] as {
      onstart: (() => void) | null
      onerror: (() => void) | null
    }
    act(() => utterance.onstart?.())

    await user.click(screen.getByRole('button', { name: 'Stop voice preview' }))
    act(() => utterance.onerror?.())

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Preview voice instruction' })).toBeEnabled()
  })

  it('requires an explicit clinician applicability choice for reanimated smile', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <VoiceGuide
        reanimatedSmileApplicable={null}
        onReanimatedSmileApplicableChange={onChange}
      />,
    )

    const applicable = screen.getByRole('radio', { name: /yes — include reanimation smile/i })
    const notApplicable = screen.getByRole('radio', { name: /no — standard assessment/i })
    expect(applicable).not.toBeChecked()
    expect(notApplicable).not.toBeChecked()
    await user.click(applicable)
    expect(onChange).toHaveBeenCalledWith(true)
    await user.click(notApplicable)
    expect(onChange).toHaveBeenCalledWith(false)
  })

  it('names the optional movement clinically instead of exposing an unexplained step number', () => {
    render(
      <VoiceGuide
        reanimatedSmileApplicable={null}
        onReanimatedSmileApplicableChange={vi.fn()}
      />,
    )

    const choice = screen.getByRole('group', {
      name: 'Should this assessment include a reanimation smile?',
    })
    expect(choice).toHaveTextContent(
      'Choose Yes only when the patient has undergone facial reanimation surgery',
    )
    expect(screen.getByRole('radio', { name: /no — standard assessment/i })).toBeEnabled()
    expect(screen.getByRole('radio', { name: /yes — include reanimation smile/i })).toBeEnabled()
    expect(choice).not.toHaveTextContent(/step 8/i)
  })

  it('distinguishes a skipped conditional step from an incomplete guided step', () => {
    render(
      <VoiceGuide
        reanimatedSmileApplicable={false}
        onReanimatedSmileApplicableChange={vi.fn()}
        guidedActive
        guidedVoice={{
          phase: 'holding',
          activeStepIndex: 6,
          countdown: 2,
          completedStepIndexes: [0, 1, 2, 3, 4, 5],
        }}
      />,
    )

    expect(screen.getByText('Movement 7 of 7')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /optional reanimation smile not included/i })).toBeDisabled()
    expect(screen.getByText(/automatic sequence/i)).toHaveTextContent(
      'No instruction clicks are needed',
    )
    expect(
      screen.queryByRole('button', { name: /voice instruction/i }),
    ).not.toBeInTheDocument()
  })
})
