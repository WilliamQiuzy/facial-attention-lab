import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'

describe('application frame', () => {
  it('keeps research status, provenance navigation, and current location visible', async () => {
    render(
      <MemoryRouter initialEntries={['/analysis']}>
        <App />
      </MemoryRouter>,
    )

    expect(
      screen.getByRole('status', { name: /research use status/i }),
    ).toHaveTextContent(/not for diagnosis, treatment, or patient-care decisions/i)
    expect(screen.getByRole('navigation', { name: /primary navigation/i })).toBeVisible()
    expect(screen.getByRole('link', { name: /attention demo/i })).toHaveAttribute(
      'aria-current',
      'page',
    )
    expect(screen.getByRole('link', { name: /patient explanation/i })).toBeVisible()
    expect(screen.getByRole('link', { name: /model & data/i })).toBeVisible()
    expect(screen.getByRole('main')).toHaveAttribute('id', 'main-content')
    expect(screen.getByRole('main')).toHaveAttribute('tabindex', '-1')
    await screen.findByText('mock_simulation')
  })

  it('does not use an official Mayo logo or trademark lockup', () => {
    const { container } = render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    )

    expect(container.querySelector('img[alt*="Mayo" i]')).not.toBeInTheDocument()
    expect(screen.getByText(/Mayo-inspired visual system/i)).toBeVisible()
    expect(screen.getByText(/no affiliation or endorsement is implied/i)).toBeVisible()
  })

  it('gives the responsive navigation toggle an explicit changing name', () => {
    const { container } = render(
      <MemoryRouter>
        <App />
      </MemoryRouter>,
    )

    const menuButton = container.querySelector<HTMLButtonElement>('.menu-button')
    expect(menuButton).not.toBeNull()
    expect(menuButton).toHaveAttribute('aria-label', 'Open navigation menu')
    expect(menuButton).toHaveAttribute('aria-expanded', 'false')

    fireEvent.click(menuButton!)

    expect(menuButton).toHaveAttribute('aria-label', 'Close navigation menu')
    expect(menuButton).toHaveAttribute('aria-expanded', 'true')
  })
})
