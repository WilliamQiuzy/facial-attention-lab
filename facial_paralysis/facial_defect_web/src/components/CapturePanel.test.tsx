import { act, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CapturePanel } from './CapturePanel'

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
      name: 'Use synthetic demo photo',
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
