import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { PRESENTATION_BOUNDARY } from '../data/presentationDemoAssets'
import { PresentationDemoPage } from './PresentationDemoPage'

describe('PresentationDemoPage', () => {
  it('opens as a photo-first, side-by-side synthetic presentation', () => {
    render(<PresentationDemoPage />)

    expect(
      screen.getByRole('heading', { name: 'Before and after, at a glance', level: 1 }),
    ).toBeVisible()
    expect(screen.getAllByText(PRESENTATION_BOUNDARY).length).toBeGreaterThanOrEqual(2)
    expect(screen.getByRole('radio', { name: 'Both' })).toBeChecked()
    expect(screen.getByRole('radio', { name: 'Photo' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: 'Show attention layer' })).toBeChecked()
    const comparison = screen.getByRole('region', { name: 'Before and after comparison' })
    expect(comparison).toBeVisible()
    expect(within(comparison).getByText('Pre-operative')).toBeVisible()
    expect(within(comparison).getByText('Post-operative')).toBeVisible()
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
        name: /abstract facial outline with simulated post-operative attention/i,
      }),
    ).toBeVisible()
    expect(screen.queryByRole('img', { name: /photograph/i })).not.toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: 'Show attention layer' }))
    expect(container.querySelectorAll('.presentation-signal-point')).toHaveLength(0)
    expect(screen.queryByText('Low')).not.toBeInTheDocument()
  })

  it('explains the simulated change without claiming a predicted outcome', () => {
    render(<PresentationDemoPage />)

    expect(screen.getByText(/small, flat scar-like edit/i)).toBeVisible()
    expect(screen.getByText(/hand-authored attention signal/i)).toBeVisible()
    expect(screen.getByText(/does not estimate an individual patient's result/i)).toBeVisible()
  })
})
