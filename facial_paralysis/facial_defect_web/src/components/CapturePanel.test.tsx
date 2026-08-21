import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CapturePanel } from './CapturePanel'

const originalMediaDevices = navigator.mediaDevices

afterEach(() => {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: originalMediaDevices,
  })
  vi.restoreAllMocks()
})

function installCamera() {
  const stop = vi.fn()
  const stream = {
    getTracks: () => [{ stop }],
  } as unknown as MediaStream
  const getUserMedia = vi.fn(async () => stream)
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  })
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
  return { getUserMedia, stop }
}

describe('CapturePanel', () => {
  it('locks photograph preparation before React can process a second click', async () => {
    let resolveSynthetic!: () => void
    const pending = new Promise<void>((resolve) => {
      resolveSynthetic = resolve
    })
    const onUseSynthetic = vi.fn(() => pending)

    render(
      <CapturePanel
        title="Add frontal photograph"
        onSelectFile={vi.fn()}
        onUseSynthetic={onUseSynthetic}
      />,
    )

    const action = screen.getByRole('button', {
      name: 'Sample photo',
    })
    act(() => {
      action.click()
      action.click()
    })

    expect(onUseSynthetic).toHaveBeenCalledTimes(1)
    expect(
      screen.getByRole('region', {
        name: 'Add frontal photograph',
      }),
    ).toHaveAttribute('aria-busy', 'true')

    await act(async () => {
      resolveSynthetic()
      await pending
    })
    expect(action).toBeEnabled()
  })

  it('offers clear Camera, Upload photo, and Sample photo choices without demo terminology', () => {
    render(
      <CapturePanel
        title="Add frontal photograph"
        onSelectFile={vi.fn()}
        onUseSynthetic={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'Camera' })).toBeVisible()
    expect(screen.getByLabelText('Upload photo')).toBeVisible()
    expect(screen.getByRole('button', { name: 'Sample photo' })).toBeVisible()
    expect(screen.queryByText(/synthetic/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Take photo')).not.toBeInTheDocument()
  })

  it('opens a live front-camera preview and stops it when cancelled', async () => {
    const user = userEvent.setup()
    const camera = installCamera()

    render(
      <CapturePanel
        title="Add frontal photograph"
        onSelectFile={vi.fn()}
        onUseSynthetic={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Camera' }))

    expect(camera.getUserMedia).toHaveBeenCalledWith({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 1280, min: 640 },
        height: { ideal: 960, min: 640 },
      },
    })
    expect(
      await screen.findByRole('region', { name: 'Camera preview' }),
    ).toBeVisible()
    expect(
      screen.getByLabelText('Live camera preview'),
    ).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Cancel camera' }))
    expect(camera.stop).toHaveBeenCalledTimes(1)
    expect(
      screen.queryByRole('region', { name: 'Camera preview' }),
    ).not.toBeInTheDocument()
  })

  it('captures the visible frame as a camera JPEG', async () => {
    const user = userEvent.setup()
    const camera = installCamera()
    const drawImage = vi.fn()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage,
    } as unknown as CanvasRenderingContext2D)
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(
      (callback) => {
        callback(new Blob(['camera jpeg'], { type: 'image/jpeg' }))
      },
    )
    vi.spyOn(HTMLVideoElement.prototype, 'videoWidth', 'get').mockReturnValue(
      640,
    )
    vi.spyOn(HTMLVideoElement.prototype, 'videoHeight', 'get').mockReturnValue(
      480,
    )
    const onSelectFile = vi.fn(
      async (_file: File, _source: 'camera' | 'upload') => undefined,
    )

    render(
      <CapturePanel
        title="Add frontal photograph"
        onSelectFile={onSelectFile}
        onUseSynthetic={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Camera' }))
    await screen.findByLabelText('Live camera preview')
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Capture photo' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Capture photo' }))

    await waitFor(() => expect(onSelectFile).toHaveBeenCalledTimes(1))
    const [file, source] = onSelectFile.mock.calls[0]!
    expect(file).toBeInstanceOf(File)
    expect(file.type).toBe('image/jpeg')
    expect(source).toBe('camera')
    expect(drawImage).toHaveBeenCalledTimes(1)
    expect(camera.stop).toHaveBeenCalledTimes(1)
  })

  it('reserves the known photograph dimensions before the preview loads', () => {
    render(
      <CapturePanel
        title="Current photograph"
        previewUrl="blob:current-photo"
        previewWidth={1024}
        previewHeight={900}
        onSelectFile={vi.fn()}
        onUseSynthetic={vi.fn()}
      />,
    )

    expect(
      screen.getByRole('img', {
        name: 'Current frontal photograph',
      }),
    ).toHaveAttribute('width', '1024')
    expect(
      screen.getByRole('img', {
        name: 'Current frontal photograph',
      }),
    ).toHaveAttribute('height', '900')
  })
})
