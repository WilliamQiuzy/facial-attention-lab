import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from './App'

describe('App', () => {
  it('renders a clearly independent research prototype shell', () => {
    render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    )

    expect(screen.getByText(/^research prototype$/i)).toBeVisible()
    expect(screen.getByText(/independent research prototype/i)).toBeVisible()
    expect(screen.getByText(/no affiliation or endorsement is implied/i)).toBeVisible()
    expect(screen.getByRole('link', { name: /skip to main content/i })).toHaveAttribute(
      'href',
      '#main-content',
    )
  })
})
