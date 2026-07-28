import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { PatientWorkflowProvider } from '../patientWorkflow/PatientWorkflowProvider'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from '../patientWorkflow/reducer'
import type { PatientWorkflowState } from '../patientWorkflow/types'
import { createPatientVisitId } from '../patientWorkflow/validation'
import { PatientsPage } from './PatientsPage'
import { PatientDetailPage } from './PatientDetailPage'

function LocationStateProbe() {
  const location = useLocation()
  return (
    <output data-testid="location-state">
      {JSON.stringify(location.state)}
    </output>
  )
}

function patientListState(): PatientWorkflowState {
  let state = createInitialPatientWorkflowState(
    DEMO_PATIENT_RECORDS,
    '2026-07-27',
  )
  state = patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: createPatientVisitId('visit-demo-list-older'),
      patientId: DEMO_PATIENT_RECORDS[0]!.id,
      timepoint: 'preoperative',
      visitDate: '2026-01-10',
      createdAt: '2026-01-10T09:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
  return patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: createPatientVisitId('visit-demo-list-latest'),
      patientId: DEMO_PATIENT_RECORDS[0]!.id,
      timepoint: 'follow_up',
      visitDate: '2026-06-18',
      createdAt: '2026-06-18T09:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
}

function renderPage(initialState = patientListState()) {
  return render(
    <MemoryRouter initialEntries={['/patients']}>
      <PatientWorkflowProvider initialState={initialState}>
        <LocationStateProbe />
        <Routes>
          <Route path="/patients" element={<PatientsPage />} />
          <Route
            path="/patients/:patientId"
            element={<PatientDetailPage />}
          />
        </Routes>
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

describe('PatientsPage', () => {
  it('shows text-first patient rows with the latest visit and one clear new-patient action', () => {
    const view = renderPage()

    expect(
      screen.getByRole('heading', { name: 'Patients', level: 1 }),
    ).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'New patient' }),
    ).toHaveAttribute('href', '/patients/new')
    expect(view.container.querySelector('img')).toBeNull()
    expect(view.container.querySelector('main')).toBeNull()

    const firstRow = screen.getByRole('listitem', {
      name: /Synthetic Demo — Facial Paralysis/i,
    })
    expect(
      within(firstRow).getByText('DEMO-1001'),
    ).toBeVisible()
    expect(
      within(firstRow).getByText('Follow-up · Jun 18, 2026'),
    ).toBeVisible()
    expect(within(firstRow).getByText('Photo needed')).toBeVisible()
    expect(
      within(firstRow).getByRole('link', { name: 'Open' }),
    ).toHaveAttribute(
      'href',
      `/patients/${DEMO_PATIENT_RECORDS[0]!.id}`,
    )
    expect(screen.getAllByText('Synthetic demo')).toHaveLength(3)
  })

  it('searches by display name or normalized record number and gives a useful empty state', async () => {
    const user = userEvent.setup()
    renderPage()

    const search = screen.getByRole('searchbox', {
      name: 'Search patients',
    })
    await user.type(search, 'reconstruction')
    expect(screen.getAllByRole('listitem')).toHaveLength(1)
    expect(
      screen.getByText('Synthetic Demo — Facial Reconstruction'),
    ).toBeVisible()

    await user.clear(search)
    await user.type(search, 'demo 1003')
    expect(screen.getAllByRole('listitem')).toHaveLength(1)
    expect(screen.getByText('DEMO-1003')).toBeVisible()

    await user.clear(search)
    await user.type(search, 'not in this session')
    expect(screen.queryAllByRole('listitem')).toHaveLength(0)
    expect(
      screen.getByRole('status', { name: 'Patient search results' }),
    ).toHaveTextContent('0 of 3 patients')
    expect(screen.getByText('No patients match this search')).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'Clear search' }))
    expect(search).toHaveValue('')
    expect(screen.getAllByRole('listitem')).toHaveLength(3)

    await user.type(search, '@@@')
    expect(screen.queryAllByRole('listitem')).toHaveLength(0)
  })

  it('restores the in-memory search after opening a patient and using the patient back link', async () => {
    const user = userEvent.setup()
    renderPage()

    const search = screen.getByRole('searchbox', {
      name: 'Search patients',
    })
    await user.type(search, 'reconstruction')
    expect(screen.getAllByRole('listitem')).toHaveLength(1)

    await user.click(screen.getByRole('link', { name: 'Open' }))
    expect(screen.getByTestId('location-state')).toHaveTextContent('null')
    expect(
      screen.getByRole('heading', {
        name: 'Synthetic Demo — Facial Reconstruction',
        level: 1,
      }),
    ).toBeVisible()

    await user.click(
      screen.getByRole('link', { name: 'Back to patients' }),
    )

    expect(screen.getByTestId('location-state')).toHaveTextContent('null')
    expect(
      screen.getByRole('searchbox', { name: 'Search patients' }),
    ).toHaveValue('reconstruction')
    expect(screen.getAllByRole('listitem')).toHaveLength(1)
    expect(
      screen.getByText('Synthetic Demo — Facial Reconstruction'),
    ).toBeVisible()
  })
})
