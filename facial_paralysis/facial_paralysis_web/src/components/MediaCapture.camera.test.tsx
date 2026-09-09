import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { MediaCapture } from './MediaCapture'

const cameraRecording = new File(['camera-video'], 'faces-camera.webm', {
  type: 'video/webm',
})
const resetRecording = vi.fn()

vi.mock('../hooks/useCameraRecorder', () => ({
  useCameraRecorder: () => ({
    status: 'recorded' as const,
    error: null,
    recordingFile: cameraRecording,
    videoRef: { current: null },
    enableCamera: vi.fn(),
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
    resetRecording,
    closeCamera: vi.fn(),
  }),
}))

describe('MediaCapture camera session', () => {
  beforeEach(() => {
    resetRecording.mockClear()
  })

  it('clears the parent recording when the clinician chooses to record again', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)

    await user.click(screen.getByRole('tab', { name: 'Use this device' }))
    expect(onRecordingChange).toHaveBeenLastCalledWith(cameraRecording, 'browser-camera')
    expect(screen.getByLabelText('Recorded camera preview')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Record again' }))

    expect(resetRecording).toHaveBeenCalledTimes(1)
    expect(onRecordingChange).toHaveBeenLastCalledWith(null, 'browser-camera')
  })
})
