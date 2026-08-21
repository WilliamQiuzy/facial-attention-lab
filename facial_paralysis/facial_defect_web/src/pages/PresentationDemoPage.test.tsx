import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { PRESENTATION_BOUNDARY } from '../data/presentationDemoAssets'
import { PresentationDemoPage } from './PresentationDemoPage'

describe('PresentationDemoPage', () => {
  it('opens as a photo-first patient comparison without visible demo terminology', () => {
    const { container } = render(<PresentationDemoPage />)

    expect(
      screen.getByRole('heading', { name: 'Patient before and after', level: 1 }),
    ).toBeVisible()
    expect(screen.queryByText(PRESENTATION_BOUNDARY)).not.toBeInTheDocument()
    expect(container.querySelector('[data-provenance]')).toHaveAttribute(
      'data-provenance',
      PRESENTATION_BOUNDARY,
    )
    expect(screen.queryByText(/clinical use blocked/i)).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Both' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Subject A' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Photo' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Side by side' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: 'Show attention layer' })).toBeChecked()
    const comparison = screen.getByRole('region', { name: 'Before and after comparison' })
    expect(comparison).toBeVisible()
    expect(within(comparison).getByText('Pre-operative')).toBeVisible()
    expect(within(comparison).getByText('Post-operative')).toBeVisible()
    expect(screen.getByText('Same patient')).toBeVisible()
    expect(screen.queryByText(/synthetic demo/i)).not.toBeInTheDocument()
  })

  it('switches between two complete sample patients without losing the selected view', async () => {
    const user = userEvent.setup()
    render(<PresentationDemoPage />)

    await user.click(screen.getByRole('radio', { name: 'Photo + outline' }))
    await user.click(screen.getByRole('radio', { name: 'Subject B' }))

    expect(screen.getByRole('radio', { name: 'Subject B' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Photo + outline' })).toBeChecked()
    expect(
      screen.getByRole('img', {
        name: /subject b sample pre-operative facial photograph/i,
      }),
    ).toBeVisible()
    expect(
      screen.getByRole('img', {
        name: /subject b sample post-operative facial photograph/i,
      }),
    ).toBeVisible()
    expect(screen.getByText('Young adult sample patient')).toBeVisible()
  })

  it('switches timepoint, display, and attention without changing routes', async () => {
    const user = userEvent.setup()
    const { container } = render(<PresentationDemoPage />)

    await user.click(screen.getByRole('radio', { name: 'Post-operative' }))
    const comparison = screen.getByRole('region', { name: 'Before and after comparison' })
    expect(within(comparison).queryByText('Pre-operative')).not.toBeInTheDocument()
    expect(within(comparison).getByText('Post-operative')).toBeVisible()

    await user.click(screen.getByRole('radio', { name: 'Outline' }))
    expect(
      screen.getByRole('img', {
        name: /abstract facial outline with illustrative post-operative attention/i,
      }),
    ).toBeVisible()
    expect(screen.queryByRole('img', { name: /photograph/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'Photo + outline' }))
    expect(
      screen.getByRole('img', {
        name: /sample post-operative facial photograph/i,
      }),
    ).toBeVisible()
    expect(container.querySelectorAll('.patient-face-contour__path')).toHaveLength(13)

    await user.click(screen.getByRole('checkbox', { name: 'Show attention layer' }))
    expect(container.querySelectorAll('.presentation-signal-point')).toHaveLength(0)
    expect(screen.queryByText('Low')).not.toBeInTheDocument()
  })

  it('offers an aligned drag comparison in one shared frame', async () => {
    const user = userEvent.setup()
    const { container } = render(<PresentationDemoPage />)

    await user.click(screen.getByRole('radio', { name: 'Drag slider' }))

    const aligned = screen.getByRole('region', {
      name: 'Interactive before and after comparison',
    })
    expect(aligned).toBeVisible()
    expect(
      within(aligned).getByText(
        'Drag the divider to compare the same facial location before and after surgery.',
      ),
    ).toBeVisible()
    expect(within(aligned).getByText('Pre-op')).toBeVisible()
    expect(within(aligned).getByText('Post-op')).toBeVisible()
    const slider = screen.getByRole('slider', { name: 'Comparison position' })
    expect(slider).toHaveValue('50')
    fireEvent.change(slider, { target: { value: '68' } })
    expect(slider).toHaveValue('68')
    expect(
      container.querySelector('.presentation-wipe'),
    ).toHaveStyle({ '--presentation-comparison-position': '68%' })
  })

  it('explains the illustrative change without presentation-cluttering warnings', () => {
    render(<PresentationDemoPage />)

    expect(screen.getByText(/closed, healing surgical incision/i)).toBeVisible()
    expect(screen.getByText(/illustrative attention signal/i)).toBeVisible()
    expect(screen.queryByText(/presentation demo/i)).not.toBeInTheDocument()
  })
})
