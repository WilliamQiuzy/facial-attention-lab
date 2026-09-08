import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { RecordingDownloadButton } from './RecordingDownloadButton'

describe('RecordingDownloadButton', () => {
  it('downloads the in-memory recording with a de-identified filename and releases its URL', async () => {
    const recording = new File(['identifiable-video'], 'patient-name.webm', { type: 'video/webm' })
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:recording-download')
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL')
    let clicked: { href: string; download: string } | null = null
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      clicked = { href: this.href, download: this.download }
    })

    render(<RecordingDownloadButton recording={recording} />)
    expect(screen.getByText(/identifiable source video to this device/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Download recorded video' }))

    expect(createObjectURL).toHaveBeenCalledWith(recording)
    expect(clicked).toEqual({
      href: 'blob:recording-download',
      download: expect.stringMatching(/^faces-research-\d{8}T\d{9}Z-[a-z0-9]+-recording\.webm$/),
    })
    await waitFor(() => expect(revokeObjectURL).toHaveBeenCalledWith('blob:recording-download'))
    expect(screen.getByRole('status')).toHaveTextContent('Download started; check browser downloads.')
    expect(document.querySelector('a[download]')).not.toBeInTheDocument()
  })

  it('uses a fixed container-matched filename rather than a source filename with identifiers', () => {
    const recording = new File(['video'], 'MRN-12345.mov', { type: 'video/quicktime' })
    let download = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      download = this.download
    })

    render(<RecordingDownloadButton recording={recording} compact />)
    fireEvent.click(screen.getByRole('button', { name: 'Download recorded video' }))
    expect(download).toMatch(/-recording\.mov$/)
    expect(document.body.textContent).not.toContain('MRN-12345')
  })

  it('recognizes browser recording MIME types that include codec parameters', () => {
    const recording = new File(['video'], 'camera-blob', { type: 'video/webm;codecs=vp9,opus' })
    let download = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      download = this.download
    })

    render(<RecordingDownloadButton recording={recording} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download recorded video' }))
    expect(download).toMatch(/-recording\.webm$/)
  })

  it('offers retry if the browser cannot request the download and cleans up its URL', async () => {
    const create = vi.spyOn(URL, 'createObjectURL').mockClear().mockReturnValue('blob:failed-download')
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementationOnce(() => { throw new Error('download blocked') }).mockImplementation(() => undefined)
    render(<RecordingDownloadButton recording={new File(['video'], 'capture.webm')} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download recorded video' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Download could not start. Please try again.')
    await waitFor(() => expect(revoke).toHaveBeenCalledWith('blob:failed-download'))
    expect(document.querySelector('a[download]')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Download recorded video' }))
    expect(screen.getByRole('status')).toHaveTextContent('Download started; check browser downloads.')
    expect(create).toHaveBeenCalledTimes(2)
  })
})
