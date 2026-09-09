import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useNavigate } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { AppHeader } from './AppHeader'

function NavigationControls() {
  const navigate = useNavigate()

  return (
    <>
      <button type="button" onClick={() => navigate('/patients/new')}>
        Open another page
      </button>
      <button type="button" onClick={() => navigate(-1)}>
        Go back
      </button>
    </>
  )
}

describe('AppHeader', () => {
  it('keeps the mobile menu closed after pathname changes, including back navigation', () => {
    render(
      <MemoryRouter initialEntries={['/patients']}>
        <AppHeader />
        <NavigationControls />
      </MemoryRouter>,
    )

    const menuButton = screen.getByRole('button', {
      name: 'Open navigation menu',
    })
    fireEvent.click(menuButton)
    expect(menuButton).toHaveAttribute('aria-expanded', 'true')

    fireEvent.click(screen.getByRole('button', { name: 'Open another page' }))
    expect(menuButton).toHaveAttribute('aria-expanded', 'false')

    fireEvent.click(screen.getByRole('button', { name: 'Go back' }))
    expect(menuButton).toHaveAttribute('aria-expanded', 'false')
  })
})
