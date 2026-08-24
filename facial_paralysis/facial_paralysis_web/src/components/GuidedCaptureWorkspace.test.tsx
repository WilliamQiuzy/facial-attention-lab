import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { GuidedCaptureWorkspace } from './GuidedCaptureWorkspace'

const mocks = vi.hoisted(() => ({
  camera: {
    status: 'ready' as 'idle' | 'requesting' | 'ready' | 'recording' | 'recorded' | 'error',
    error: null as string | null,
    recordingFile: null as File | null,
    recordingStartedAtMs: null as number | null,
    videoRef: { current: null },
    enableCamera: vi.fn(),
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
    discardRecording: vi.fn(),
    resetRecording: vi.fn(),
    closeCamera: vi.fn(),
  },
  voice: {
    supported: true,
    phase: 'idle' as 'idle' | 'speaking' | 'holding' | 'complete' | 'error',
    activeStepIndex: null as number | null,
    countdown: null as number | null,
    completedStepIndexes: [] as number[],
    timeline: null as import('../model/inference').CaptureTimelineDraft | null,
    error: null as string | null,
    start: vi.fn(),
    cancel: vi.fn(),
  },
}))

vi.mock('../hooks/useCameraRecorder', () => ({
  useCameraRecorder: () => mocks.camera,
}))

vi.mock('../hooks/useGuidedVoiceSequence', () => ({
  useGuidedVoiceSequence: () => mocks.voice,
}))

describe('GuidedCaptureWorkspace', () => {
  beforeEach(() => {
    mocks.camera.status = 'ready'
    mocks.camera.error = null
    mocks.camera.recordingFile = null
    mocks.camera.recordingStartedAtMs = null
    mocks.voice.supported = true
    mocks.voice.phase = 'idle'
    mocks.voice.activeStepIndex = null
    mocks.voice.countdown = null
    mocks.voice.completedStepIndexes = []
    mocks.voice.timeline = null
    mocks.voice.error = null
    Object.values(mocks.camera).forEach((value) => {
      if (typeof value === 'function' && 'mockClear' in value) value.mockClear()
    })
    Object.values(mocks.voice).forEach((value) => {
      if (typeof value === 'function' && 'mockClear' in value) value.mockClear()
    })
  })

  it('requires a ready camera and explicit step-8 choice before combined start', async () => {
    const user = userEvent.setup()
    const { rerender } = render(
      <GuidedCaptureWorkspace
        reanimatedSmileApplicable={null}
        onReanimatedSmileApplicableChange={vi.fn()}
        onRecordingChange={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    expect(screen.getByRole('button', { name: 'Start guided recording' })).toBeDisabled()
    expect(screen.getByText(/resolve step 8/i)).toBeInTheDocument()

    rerender(
      <GuidedCaptureWorkspace
        reanimatedSmileApplicable={false}
        onReanimatedSmileApplicableChange={vi.fn()}
        onRecordingChange={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: 'Start guided recording' })).toBeEnabled()
  })

  it('starts recording first, then voice, and stops automatically after the sequence', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    const props = {
      reanimatedSmileApplicable: false as const,
      onReanimatedSmileApplicableChange: vi.fn(),
      onRecordingChange,
    }
    const { rerender } = render(<GuidedCaptureWorkspace {...props} />)
    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    await user.click(screen.getByRole('button', { name: 'Start guided recording' }))

    expect(mocks.camera.startRecording).toHaveBeenCalledTimes(1)
    expect(mocks.voice.start).not.toHaveBeenCalled()

    mocks.camera.recordingStartedAtMs = 12_345
    mocks.camera.status = 'recording'
    rerender(<GuidedCaptureWorkspace {...props} />)
    await waitFor(() => expect(mocks.voice.start).toHaveBeenCalledWith(false, 12_345))

    mocks.voice.timeline = {
      recordingDurationMs: 28_000,
      actions: [
        'repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
        'relaxed_smile', 'lip_pucker', 'lower_teeth_show',
      ].map((id, index) => ({
        id,
        promptStartMs: index * 4_000,
        holdStartMs: index * 4_000 + 500,
        holdEndMs: index * 4_000 + 3_500,
        completionMs: index * 4_000 + 3_750,
      })) as import('../model/inference').CaptureTimelineDraft['actions'],
    }
    mocks.voice.phase = 'complete'
    mocks.voice.activeStepIndex = 6
    rerender(<GuidedCaptureWorkspace {...props} />)
    await waitFor(() => expect(mocks.camera.stopRecording).toHaveBeenCalledTimes(1))

    const recording = new File(['complete'], 'guided-session.webm', { type: 'video/webm' })
    mocks.camera.recordingFile = recording
    mocks.camera.status = 'recorded'
    rerender(<GuidedCaptureWorkspace {...props} />)

    await waitFor(() => {
      expect(onRecordingChange).toHaveBeenCalledWith(
        recording,
        'browser-camera',
        expect.objectContaining({
          preserveProtocolChoice: true,
          reanimatedSmileApplicable: false,
          actionIds: [
            'repose',
            'eyebrow_raise',
            'gentle_eye_closure',
            'tight_eye_squeeze',
            'relaxed_smile',
            'lip_pucker',
            'lower_teeth_show',
          ],
          timeline: mocks.voice.timeline,
        }),
      )
    })
    expect(screen.getByRole('status')).toHaveTextContent(/recording complete/i)
  })

  it('keeps an automatic visual and text cue visible throughout recording', async () => {
    const user = userEvent.setup()
    const props = {
      reanimatedSmileApplicable: false as const,
      onReanimatedSmileApplicableChange: vi.fn(),
      onRecordingChange: vi.fn(),
    }
    const { rerender } = render(<GuidedCaptureWorkspace {...props} />)
    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    await user.click(screen.getByRole('button', { name: 'Start guided recording' }))

    mocks.camera.recordingStartedAtMs = 12_345
    mocks.camera.status = 'recording'
    mocks.voice.phase = 'speaking'
    mocks.voice.activeStepIndex = 1
    rerender(<GuidedCaptureWorkspace {...props} />)

    const patientGuide = await screen.findByRole('region', {
      name: 'Patient movement guidance',
    })
    expect(within(patientGuide).getByText('Step 2 of 7')).toBeInTheDocument()
    expect(
      within(patientGuide).getByRole('heading', { name: 'Eyebrow Raise' }),
    ).toBeInTheDocument()
    expect(
      within(patientGuide.querySelector('.patient-guidance-copy') as HTMLElement)
        .getByText(/raise your eyebrows as high as you can/i),
    ).toBeInTheDocument()
    expect(
      within(patientGuide).getByRole('img', {
        name: 'Eyebrow Raise movement demonstration',
      }),
    ).toHaveClass('is-active')
    expect(within(patientGuide).getByText(/voice prompt playing/i)).toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: /voice instruction/i }),
    ).not.toBeInTheDocument()

    mocks.voice.phase = 'holding'
    mocks.voice.countdown = 3
    rerender(<GuidedCaptureWorkspace {...props} />)

    expect(
      within(patientGuide).getByLabelText('3 seconds remaining'),
    ).toBeInTheDocument()
    expect(within(patientGuide).getByText('Hold the pose')).toBeInTheDocument()
    expect(
      within(patientGuide).getByRole('img', {
        name: 'Eyebrow Raise movement demonstration',
      }),
    ).not.toHaveClass('is-active')
  })

  it('cancels speech and discards an interrupted recording', async () => {
    const user = userEvent.setup()
    const props = {
      reanimatedSmileApplicable: true as const,
      onReanimatedSmileApplicableChange: vi.fn(),
      onRecordingChange: vi.fn(),
    }
    const { rerender } = render(<GuidedCaptureWorkspace {...props} />)
    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    await user.click(screen.getByRole('button', { name: 'Start guided recording' }))
    mocks.camera.recordingStartedAtMs = 12_345
    mocks.camera.status = 'recording'
    mocks.voice.phase = 'speaking'
    mocks.voice.activeStepIndex = 0
    rerender(<GuidedCaptureWorkspace {...props} />)

    await user.click(screen.getByRole('button', { name: 'Stop and discard guided recording' }))

    expect(mocks.voice.cancel).toHaveBeenCalled()
    expect(mocks.camera.discardRecording).toHaveBeenCalled()
    expect(props.onRecordingChange).not.toHaveBeenCalledWith(
      expect.any(File),
      'browser-camera',
      expect.anything(),
    )
  })

  it('discards the recording when voice guidance fails', async () => {
    const user = userEvent.setup()
    const props = {
      reanimatedSmileApplicable: false as const,
      onReanimatedSmileApplicableChange: vi.fn(),
      onRecordingChange: vi.fn(),
    }
    const { rerender } = render(<GuidedCaptureWorkspace {...props} />)
    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    await user.click(screen.getByRole('button', { name: 'Start guided recording' }))
    mocks.camera.recordingStartedAtMs = 12_345
    mocks.camera.status = 'recording'
    rerender(<GuidedCaptureWorkspace {...props} />)
    mocks.voice.phase = 'error'
    mocks.voice.error = 'The voice instruction could not be played.'
    rerender(<GuidedCaptureWorkspace {...props} />)

    await waitFor(() => expect(mocks.camera.discardRecording).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('alert')).toHaveTextContent(/voice instruction/i)
  })

  it('leaves finalizing safely when the recorder cannot produce a file', async () => {
    const user = userEvent.setup()
    const props = {
      reanimatedSmileApplicable: false as const,
      onReanimatedSmileApplicableChange: vi.fn(),
      onRecordingChange: vi.fn(),
    }
    const { rerender } = render(<GuidedCaptureWorkspace {...props} />)
    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    await user.click(screen.getByRole('button', { name: 'Start guided recording' }))
    mocks.camera.recordingStartedAtMs = 12_345
    mocks.camera.status = 'recording'
    rerender(<GuidedCaptureWorkspace {...props} />)
    mocks.voice.phase = 'complete'
    rerender(<GuidedCaptureWorkspace {...props} />)
    await waitFor(() => expect(mocks.camera.stopRecording).toHaveBeenCalledTimes(1))

    mocks.camera.status = 'error'
    mocks.camera.error = 'The browser ended the recording without producing video data.'
    rerender(<GuidedCaptureWorkspace {...props} />)

    await waitFor(() => expect(mocks.camera.discardRecording).toHaveBeenCalledTimes(1))
    expect(screen.getByRole('alert')).toHaveTextContent(/without producing video data/i)
    expect(props.onRecordingChange).not.toHaveBeenCalledWith(
      expect.any(File),
      'browser-camera',
      expect.anything(),
    )
  })
})
