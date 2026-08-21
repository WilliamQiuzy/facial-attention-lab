import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PatientJobProgress } from './PatientJobProgress'

describe('PatientJobProgress', () => {
  it('describes a queued analysis as waiting instead of running or starting', () => {
    render(<PatientJobProgress status="queued" />)

    const progress = screen.getByRole('region', {
      name: 'Analysis progress',
    })
    expect(
      within(progress).getByText('Waiting for analysis to begin…'),
    ).toBeVisible()

    const analysisPhase = within(progress)
      .getByText('Analysis')
      .closest('li')
    expect(analysisPhase).toHaveTextContent('Queued')
    expect(progress).not.toHaveTextContent('Starting')
    expect(progress).not.toHaveTextContent('Analysis running')

    expect(
      screen.getByRole('status', {
        name: 'Analysis status announcement',
      }),
    ).toHaveTextContent('Analysis queued')
  })

  it('exposes a stable visual state for every numbered step', () => {
    const { container } = render(<PatientJobProgress status="running" />)
    const markers = container.querySelectorAll(
      '.patient-job-progress__marker',
    )

    expect(markers).toHaveLength(4)
    expect(
      [...markers].map((marker) => marker.getAttribute('data-step')),
    ).toEqual(['1', '2', '3', '4'])
    expect(
      [...markers].map((marker) => marker.getAttribute('data-state')),
    ).toEqual(['complete', 'complete', 'current', 'pending'])
  })
})
