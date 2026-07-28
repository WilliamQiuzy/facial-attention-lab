import { render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import { PatientWorkflowProvider } from '../patientWorkflow/PatientWorkflowProvider'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from '../patientWorkflow/reducer'
import type { PatientWorkflowState } from '../patientWorkflow/types'
import { createPatientVisitId } from '../patientWorkflow/validation'
import { PatientDetailPage } from './PatientDetailPage'

function detailState(): PatientWorkflowState {
  let state = createInitialPatientWorkflowState(
    [DEMO_PATIENT_RECORDS[0]!],
    '2026-07-27',
  )
  state = patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: createPatientVisitId('visit-detail-later'),
      patientId: DEMO_PATIENT_RECORDS[0]!.id,
      timepoint: 'follow_up',
      visitDate: '2026-06-20',
      createdAt: '2026-06-20T09:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
  return patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: createPatientVisitId('visit-detail-earlier'),
      patientId: DEMO_PATIENT_RECORDS[0]!.id,
      timepoint: 'preoperative',
      visitDate: '2026-01-05',
      createdAt: '2026-01-05T09:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
}

function renderPage(
  path: string,
  initialState: PatientWorkflowState = detailState(),
) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <PatientWorkflowProvider initialState={initialState}>
        <Routes>
          <Route
            path="/patients/:patientId"
            element={<PatientDetailPage />}
          />
        </Routes>
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

describe('PatientDetailPage', () => {
  it('keeps identity visible and lists visits in chronological order with one next action each', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const view = renderPage(`/patients/${patient.id}`)
    expect(view.container.querySelector('main')).toBeNull()

    expect(
      screen.getByRole('heading', { name: patient.displayName, level: 1 }),
    ).toBeVisible()
    const identity = screen.getByRole('region', {
      name: 'Patient identity',
    })
    expect(within(identity).getByText(patient.recordNumber)).toBeVisible()
    expect(within(identity).getByText('Mar 14, 1962')).toBeVisible()
    expect(within(identity).getByText(patient.carePathway)).toBeVisible()
    expect(within(identity).getByText('Synthetic demo')).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'Add photo visit' }),
    ).toHaveAttribute('href', `/patients/${patient.id}/visits/new`)

    const timeline = screen.getByRole('region', {
      name: 'Visit timeline',
    })
    const visits = within(timeline).getAllByRole('listitem')
    expect(visits).toHaveLength(2)
    expect(within(visits[0]!).getByText('Preoperative')).toBeVisible()
    expect(within(visits[0]!).getByText('Jan 5, 2026')).toBeVisible()
    expect(within(visits[1]!).getByText('Follow-up')).toBeVisible()
    expect(within(visits[1]!).getByText('Jun 20, 2026')).toBeVisible()
    for (const visit of visits) {
      expect(within(visit).getByText('Photo needed')).toBeVisible()
      expect(
        within(visit).getByRole('link', { name: 'Add photo' }),
      ).toBeVisible()
    }
  })

  it('fails closed for an unknown patient', () => {
    renderPage('/patients/patient-does-not-exist')

    expect(
      screen.getByRole('heading', {
        name: 'Patient record unavailable',
        level: 1,
      }),
    ).toBeVisible()
    expect(
      screen.getByText(
        'This patient record is not available in the current session.',
      ),
    ).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'Back to patients' }),
    ).toHaveAttribute('href', '/patients')
    expect(
      screen.queryByRole('link', { name: 'Add photo visit' }),
    ).not.toBeInTheDocument()
  })
})
