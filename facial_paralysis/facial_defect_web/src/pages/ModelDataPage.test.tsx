import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'

describe('model and data readiness', () => {
  it('makes source origin, endpoints, versions, inputs, and gates explicit', () => {
    render(
      <MemoryRouter initialEntries={['/model']}>
        <App />
      </MemoryRouter>,
    )

    expect(screen.getByText('mock_simulation')).toBeVisible()
    expect(screen.getByText('observed_gaze')).toBeVisible()
    expect(screen.getByText('model_prediction')).toBeVisible()
    expect(screen.getByText('/api/v1/attention-analyses')).toBeVisible()
    expect(screen.getByText('/api/v1/salience-predictions')).toBeVisible()
    expect(screen.getByText(/disconnected by design/i)).toBeVisible()
    expect(screen.getByText(/model version/i)).toBeVisible()
    expect(screen.getByText(/not connected/i)).toBeVisible()
    expect(screen.getAllByText(/two approved image assets/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/ROI review/i)).toBeVisible()
    expect(screen.getByText(/minimum eligible sample/i)).toBeVisible()
  })
})
