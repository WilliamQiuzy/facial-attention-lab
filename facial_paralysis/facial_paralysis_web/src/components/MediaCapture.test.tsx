import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { MediaCapture } from './MediaCapture'

describe('MediaCapture', () => {
  it('accepts a supported LifeLink Face video and shows session-only metadata', async () => {
    const user = userEvent.setup()
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)

    const file = new File(['synthetic-video'], 'livelink-session.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)

    expect(onRecordingChange).toHaveBeenCalledWith(file, 'livelink-upload')
    expect(screen.getByText('livelink-session.mp4')).toBeInTheDocument()
    expect(screen.getByText(/kept in this browser session only/i)).toBeInTheDocument()
  })

  it('rejects unsupported non-video files', async () => {
    const user = userEvent.setup({ applyAccept: false })
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)

    const file = new File(['image'], 'portrait.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)

    expect(onRecordingChange).toHaveBeenCalledWith(null, 'livelink-upload')
    expect(screen.getByRole('alert')).toHaveTextContent(/supported video/i)
  })

  it('clears the previous recording when an invalid replacement is selected', async () => {
    const user = userEvent.setup({ applyAccept: false })
    const onRecordingChange = vi.fn()
    render(<MediaCapture onRecordingChange={onRecordingChange} />)

    const valid = new File(['video'], 'first-session.mp4', { type: 'video/mp4' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), valid)
    const invalid = new File(['image'], 'different-patient.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), invalid)

    expect(onRecordingChange).toHaveBeenLastCalledWith(null, 'livelink-upload')
    expect(screen.queryByText('first-session.mp4')).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(/supported video/i)
  })

  it('supports arrow-key navigation between the recording-source tabs', async () => {
    const user = userEvent.setup()
    render(<MediaCapture onRecordingChange={vi.fn()} />)

    const uploadTab = screen.getByRole('tab', { name: 'Upload from LifeLink' })
    const cameraTab = screen.getByRole('tab', { name: 'Use this device' })
    uploadTab.focus()
    await user.keyboard('{ArrowRight}')

    expect(cameraTab).toHaveFocus()
    expect(cameraTab).toHaveAttribute('aria-selected', 'true')
    expect(cameraTab).toHaveAttribute('aria-controls')
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
