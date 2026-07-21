import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'

afterEach(() => vi.restoreAllMocks())

describe('patient explanation demo', () => {
  it('uses plain, non-diagnostic language and preserves the unpaired synthetic boundary', () => {
    const { container } = render(
      <MemoryRouter initialEntries={['/patient-report?case=demo-001']}>
        <App />
      </MemoryRouter>,
    )

    expect(screen.getByRole('heading', { name: /a guide to visual attention maps/i })).toBeVisible()
    expect(screen.getByText(/not a result about you or any patient/i)).toBeVisible()
    expect(screen.getByText(/different AI-generated people/i)).toBeVisible()
    expect(screen.getByText(/does not tell us what someone thinks or feels/i)).toBeVisible()
    expect(screen.getByText(/does not show whether a procedure worked/i)).toBeVisible()
    expect(container).not.toHaveTextContent(
      /your outcome|people will judge|more attractive|procedure succeeded|recommended surgery/i,
    )
    expect(screen.getByRole('link', { name: /return to clinician demo/i })).toHaveAttribute(
      'href',
      '/analysis?case=demo-001',
    )
  })

  it('offers a print-friendly handoff action', async () => {
    const printSpy = vi.spyOn(window, 'print').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/patient-report?case=demo-001']}>
        <App />
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('button', { name: /print patient explanation/i }))
    expect(printSpy).toHaveBeenCalledOnce()
  })
})
