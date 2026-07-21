import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { App } from '../App'

describe('clinician attention demo', () => {
  it('labels every map as simulated and every image as synthetic and unpaired', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const { container } = render(
      <MemoryRouter initialEntries={['/analysis']}>
        <App />
      </MemoryRouter>,
    )

    expect(await screen.findByText('Synthetic image A — unpaired')).toBeVisible()
    expect(screen.getByText('Synthetic image B — unpaired')).toBeVisible()
    expect(screen.getAllByText('SIMULATED — NOT HUMAN GAZE')).toHaveLength(2)
    expect(screen.getByText('mock_simulation')).toBeVisible()
    expect(screen.getByText(/these are different generated identities/i)).toBeVisible()
    expect(container).not.toHaveTextContent(/preoperative|postoperative|treatment improvement/i)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('supports accessible map, opacity, and ROI controls', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/analysis']}>
        <App />
      </MemoryRouter>,
    )

    await screen.findByText('Synthetic image A — unpaired')
    const opacity = screen.getByRole('slider', { name: /heatmap opacity/i })
    expect(opacity).toHaveValue('68')
    fireEvent.change(opacity, { target: { value: '42' } })
    expect(opacity).toHaveValue('42')

    const roiButton = screen.getByRole('button', { name: /show scar region/i })
    expect(roiButton).toHaveAttribute('aria-pressed', 'false')
    await user.click(roiButton)
    expect(roiButton).toHaveAttribute('aria-pressed', 'true')

    await user.click(screen.getByRole('button', { name: /^original images$/i }))
    await waitFor(() =>
      expect(screen.queryAllByText('SIMULATED — NOT HUMAN GAZE')).toHaveLength(0),
    )
    await user.click(screen.getByRole('button', { name: /^attention maps$/i }))
    expect(screen.getAllByText('SIMULATED — NOT HUMAN GAZE')).toHaveLength(2)
  })

  it('hands the selected synthetic case to the patient explanation', async () => {
    render(
      <MemoryRouter initialEntries={['/analysis']}>
        <App />
      </MemoryRouter>,
    )

    const link = await screen.findByRole('link', { name: /prepare patient explanation/i })
    expect(link).toHaveAttribute('href', '/patient-report?case=demo-001')
    expect(screen.getByText(/not evidence of surgical change/i)).toBeVisible()
  })
})
