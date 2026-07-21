import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'

describe('methods and safeguards', () => {
  it('defines proposal measures and privacy limitations without upgrading research claims', () => {
    render(
      <MemoryRouter initialEntries={['/methods']}>
        <App />
      </MemoryRouter>,
    )

    expect(screen.getByRole('heading', { name: /methods, provenance & safeguards/i })).toBeVisible()
    expect(screen.getByText(/fixation duration/i)).toBeVisible()
    expect(screen.getByText(/time to first fixation/i)).toBeVisible()
    expect(screen.getByText(/attention is not emotion, judgment, stigma/i)).toBeVisible()
    expect(screen.getByText(/no photos, identifiers, or analysis payloads are persisted/i)).toBeVisible()
    expect(screen.getByText(/IRB and protocol decisions remain institutional gates/i)).toBeVisible()
  })
})
