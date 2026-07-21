import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'

describe('synthetic case worklist', () => {
  it('shows only approved synthetic fixtures and routes the selected case to analysis', () => {
    render(
      <MemoryRouter initialEntries={['/cases']}>
        <App />
      </MemoryRouter>,
    )

    expect(screen.getByRole('heading', { name: /synthetic case worklist/i })).toBeVisible()
    expect(screen.getByText(/D-001/)).toBeVisible()
    expect(screen.getByText(/AI-generated · unpaired/i)).toBeVisible()
    expect(screen.getByRole('link', { name: /open case D-001/i })).toHaveAttribute(
      'href',
      '/analysis?case=demo-001',
    )
    expect(screen.getByLabelText(/worklist summary/i)).toHaveTextContent(/0patient records/i)
  })

  it('provides a useful empty state without widening the data boundary', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/cases']}>
        <App />
      </MemoryRouter>,
    )

    await user.type(screen.getByRole('searchbox', { name: /search synthetic cases/i }), 'missing')
    expect(screen.getByText(/no approved synthetic cases match/i)).toBeVisible()
    expect(screen.getByText(/upload is unavailable in this prototype/i)).toBeVisible()
  })
})
