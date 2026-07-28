import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { createInitialWorkspaceState } from '../workbench/reducer'

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function renderDashboard(initialState = createInitialWorkspaceState()) {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <App initialState={initialState} />
    </MemoryRouter>,
  )
}

describe('clinician entry route', () => {
  it('opens the patient list without exposing a system dashboard', () => {
    renderDashboard()

    expect(
      screen.getByRole('heading', { name: 'Patients', level: 1 }),
    ).toBeVisible()
    expect(
      screen.getByRole('status', {
        name: 'Patient search results',
      }),
    ).toHaveTextContent('3 of 3 patients')
    const firstPatient = DEMO_PATIENT_RECORDS[0]!
    const firstRow = screen.getByRole('listitem', {
      name: firstPatient.displayName,
    })
    expect(
      within(firstRow).getByRole('link', { name: 'Open' }),
    ).toHaveAttribute('href', `/patients/${firstPatient.id}`)
    expect(
      screen.getByRole('link', { name: 'New patient' }),
    ).toHaveAttribute('href', '/patients/new')
    expect(
      screen.queryByRole('region', { name: 'Workspace summary' }),
    ).not.toBeInTheDocument()
    expect(screen.queryByTestId('metric-session-jobs')).not.toBeInTheDocument()
    expect(
      within(screen.getByRole('navigation', { name: 'Primary navigation' }))
        .getAllByRole('link')
        .map((link) => link.textContent),
    ).toEqual(['Patients', 'Reviews', 'Help'])
  })

  it('does not fetch or read/write browser storage in the default mock session', () => {
    const fetchSpy = vi.fn()
    const storageReadSpy = vi.spyOn(Storage.prototype, 'getItem')
    const storageWriteSpy = vi.spyOn(Storage.prototype, 'setItem')
    vi.stubGlobal('fetch', fetchSpy)

    renderDashboard()

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageReadSpy).not.toHaveBeenCalled()
    expect(storageWriteSpy).not.toHaveBeenCalled()
    expect(
      screen.getByRole('status', { name: 'Workspace environment' }),
    ).toHaveTextContent(
      'Research prototype · synthetic/test records only · session data resets on refresh · clinical use blocked',
    )
  })
})
