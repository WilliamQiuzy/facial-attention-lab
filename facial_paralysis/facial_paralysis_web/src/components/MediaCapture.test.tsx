import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { MediaCapture } from './MediaCapture'

describe('MediaCapture', () => {
  function withReportedSize(file: File, size: number): File {
    Object.defineProperty(file, 'size', { configurable: true, value: size })
    return file
  }

  async function openUpload(user: ReturnType<typeof userEvent.setup>) {
    await user.click(screen.getByRole('tab', { name: 'Upload from LifeLink' }))
  }

  it('accepts a supported LifeLink Face video and shows session-only metadata', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)

    const file = new File(['synthetic-video'], 'livelink-session.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)

    expect(onRecordingChange).toHaveBeenCalledWith(file, 'livelink-upload')
    expect(screen.getByText('livelink-session.mp4')).toBeInTheDocument()
    expect(screen.getByText(/kept in this browser session only/i)).toBeInTheDocument()
  })

  it('binds an eight-action LifeLink timeline sidecar to the selected video', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)
    const video = new File(['video'], 'livelink-session.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), video)
    const ids = [
      'neutral_repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
      'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
    ]
    const sidecar = new File([JSON.stringify({
      schema_version: 'faces-action-timeline/v1',
      script_version: 'faces-script/24-004956-v1',
      recording_sha256: 'a'.repeat(64),
      timing_source: 'capture_event_log',
      recording_duration_ms: 32_000,
      actions: ids.map((action, index) => ({
        action,
        status: 'completed',
        prompt_start_ms: index * 4_000,
        hold_start_ms: index * 4_000 + 500,
        hold_end_ms: index * 4_000 + 3_500,
        completion_ms: index * 4_000 + 3_750,
      })),
    })], 'livelink-session.timeline.json', { type: 'application/json' })
    await user.upload(screen.getByLabelText('Choose FACES action timeline'), sidecar)

    expect(onRecordingChange).toHaveBeenLastCalledWith(
      video,
      'livelink-upload',
      expect.objectContaining({
        preserveProtocolChoice: true,
        reanimatedSmileApplicable: true,
        timeline: expect.objectContaining({ recordingDurationMs: 32_000 }),
      }),
    )
  })

  it('binds a seven-step timeline without inventing reanimated smile', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)
    const video = new File(['video'], 'seven-step.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), video)
    const ids = [
      'neutral_repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
      'relaxed_smile', 'lip_pucker', 'lower_teeth_show',
    ]
    const sidecar = new File([JSON.stringify({
      schema_version: 'faces-action-timeline/v1',
      script_version: 'faces-script/24-004956-v1',
      recording_sha256: 'a'.repeat(64),
      timing_source: 'capture_event_log',
      recording_duration_ms: 28_000,
      actions: ids.map((action, index) => ({
        action,
        status: 'completed',
        prompt_start_ms: index * 4_000,
        hold_start_ms: index * 4_000 + 500,
        hold_end_ms: index * 4_000 + 3_500,
        completion_ms: index * 4_000 + 3_750,
      })),
    })], 'seven-step.timeline.json', { type: 'application/json' })
    await user.upload(screen.getByLabelText('Choose FACES action timeline'), sidecar)

    expect(onRecordingChange).toHaveBeenLastCalledWith(
      video,
      'livelink-upload',
      expect.objectContaining({
        preserveProtocolChoice: true,
        reanimatedSmileApplicable: false,
        timeline: expect.objectContaining({ recordingDurationMs: 28_000 }),
      }),
    )
  })

  it('revokes the preview URL after replacement and unmount', async () => {
    const user = userEvent.setup()
    const revoke = vi.mocked(URL.revokeObjectURL)
    const { unmount } = render(<MediaCapture onRecordingChange={vi.fn()} />)
    await openUpload(user)
    const first = new File(['first'], 'first.mp4', { type: 'video/mp4' })
    const second = new File(['second'], 'second.mp4', { type: 'video/mp4' })
    const input = screen.getByLabelText('Choose LifeLink Face video')

    await user.upload(input, first)
    await user.upload(input, second)
    expect(revoke).toHaveBeenCalledWith('blob:faces-test')
    const callsBeforeUnmount = revoke.mock.calls.length
    unmount()
    expect(revoke.mock.calls.length).toBeGreaterThan(callsBeforeUnmount)
  })

  it('rejects unsupported non-video files', async () => {
    const user = userEvent.setup({ applyAccept: false })
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)

    const file = new File(['image'], 'portrait.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)

    expect(onRecordingChange).toHaveBeenCalledWith(
      null,
      'livelink-upload',
      { preserveProtocolChoice: true },
    )
    expect(screen.getByRole('alert')).toHaveTextContent(/supported video/i)
  })

  it('clears the previous recording when an invalid replacement is selected', async () => {
    const user = userEvent.setup({ applyAccept: false })
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)

    const valid = new File(['video'], 'first-session.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), valid)
    const invalid = new File(['image'], 'different-patient.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), invalid)

    expect(onRecordingChange).toHaveBeenLastCalledWith(
      null,
      'livelink-upload',
      { preserveProtocolChoice: true },
    )
    expect(screen.queryByText('first-session.mp4')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/supported video/i)
  })

  it.each([
    [0, /empty/i],
    [512 * 1024 * 1024 + 1, /larger than 512 MB/i],
  ] as const)('rejects unusable video size %s before retaining or uploading it', async (size, message) => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)
    const file = withReportedSize(
      new File(['bounded'], 'session.mp4', { type: 'video/mp4' }),
      size,
    )

    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)

    expect(onRecordingChange).toHaveBeenLastCalledWith(
      null,
      'livelink-upload',
      { preserveProtocolChoice: true },
    )
    expect(screen.getByRole('alert')).toHaveTextContent(message)
    expect(screen.queryByText('session.mp4')).not.toBeInTheDocument()
  })

  it('accepts the exact 512 MiB browser boundary', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)
    const file = withReportedSize(
      new File(['bounded'], 'boundary.webm', { type: 'video/webm' }),
      512 * 1024 * 1024,
    )

    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)

    expect(onRecordingChange).toHaveBeenLastCalledWith(file, 'livelink-upload')
    expect(screen.getByText('boundary.webm')).toBeInTheDocument()
  })

  it.each([
    ['video bytes', 'misleading.mp4', 'image/png'],
    ['video bytes', 'misleading.jpg', 'video/mp4'],
  ])('rejects misleading extension and MIME combinations', async (contents, name, type) => {
    const user = userEvent.setup({ applyAccept: false })
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)
    await openUpload(user)

    await user.upload(
      screen.getByLabelText('Choose LifeLink Face video'),
      new File([contents], name, { type }),
    )

    expect(onRecordingChange).toHaveBeenLastCalledWith(
      null,
      'livelink-upload',
      { preserveProtocolChoice: true },
    )
    expect(screen.getByRole('alert')).toHaveTextContent(/supported video/i)
  })

  it('supports arrow-key navigation between the recording-source tabs', async () => {
    const user = userEvent.setup()
    render(<MediaCapture onRecordingChange={vi.fn()} />)

    const uploadTab = screen.getByRole('tab', { name: 'Upload from LifeLink' })
    const cameraTab = screen.getByRole('tab', { name: 'Use this device' })
    cameraTab.focus()
    await user.keyboard('{ArrowRight}')

    expect(uploadTab).toHaveFocus()
    expect(uploadTab).toHaveAttribute('aria-selected', 'true')
    expect(uploadTab).toHaveAttribute('aria-controls')
  })

  it('surfaces camera permission denial without hiding the upload path', async () => {
    const user = userEvent.setup()
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn().mockRejectedValue(new DOMException('denied', 'NotAllowedError')) },
    })
    render(<MediaCapture onRecordingChange={vi.fn()} />)

    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    await user.click(screen.getByRole('button', { name: 'Enable camera' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/permission/i)
    expect(screen.getByRole('tab', { name: 'Upload from LifeLink' })).toBeInTheDocument()
  })
})
