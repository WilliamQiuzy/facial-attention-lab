import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import {
  PatientWorkflowProvider,
  usePatientWorkflow,
  type PatientWorkflowRuntime,
} from '../patientWorkflow/PatientWorkflowProvider'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from '../patientWorkflow/reducer'
import type { PatientWorkflowState } from '../patientWorkflow/types'
import { createPatientVisitId } from '../patientWorkflow/validation'
import { NewPatientVisitPage } from './NewPatientVisitPage'

const runtime: PatientWorkflowRuntime = {
  nextIdToken: (kind) => `${kind}_new_visit`,
  now: () => '2026-07-27T15:00:00.000Z',
  today: () => '2026-07-27',
}

function Destination() {
  const location = useLocation()
  return <output aria-label="Current location">{location.pathname}</output>
}

function VisitCount() {
  const { state } = usePatientWorkflow()
  return (
    <output aria-label="Visit count">{state.visitOrder.length}</output>
  )
}

function renderPage(
  initialState: PatientWorkflowState,
  patientId: string = DEMO_PATIENT_RECORDS[0]!.id,
) {
  return render(
    <MemoryRouter
      initialEntries={[`/patients/${patientId}/visits/new`]}
    >
      <PatientWorkflowProvider
        initialState={initialState}
        runtime={runtime}
      >
        <Routes>
          <Route
            path="/patients/:patientId/visits/new"
            element={<NewPatientVisitPage />}
          />
          <Route
            path="/patients/:patientId/visits/:visitId"
            element={<Destination />}
          />
        </Routes>
        <VisitCount />
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

function stateWithOneVisit() {
  const patient = DEMO_PATIENT_RECORDS[0]!
  return patientWorkflowReducer(
    createInitialPatientWorkflowState([patient], '2026-07-27'),
    {
      type: 'visit/create',
      visit: {
        id: createPatientVisitId('visit-existing-record'),
        patientId: patient.id,
        timepoint: 'preoperative',
        visitDate: '2026-01-10',
        createdAt: '2026-01-10T09:00:00.000Z',
      },
      trustedToday: '2026-07-27',
    },
  )
}

describe('NewPatientVisitPage', () => {
  it('keeps date autofill off and offers a clear way back to the patient record', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    renderPage(
      createInitialPatientWorkflowState([patient], '2026-07-27'),
    )

    expect(screen.getByLabelText('Visit date')).toHaveAttribute(
      'autocomplete',
      'off',
    )
    expect(screen.getByRole('link', { name: 'Cancel' })).toHaveAttribute(
      'href',
      `/patients/${patient.id}`,
    )
  })

  it('marks its header for the compact tablet layout', () => {
    const patient = DEMO_PATIENT_RECORDS[0]!
    const view = renderPage(
      createInitialPatientWorkflowState([patient], '2026-07-27'),
    )

    expect(
      view.container.querySelector(
        'header.patient-page-header.patient-visit-create-header',
      ),
    ).toBeInTheDocument()
  })

  it('keeps secondary identity details closed while preserving access to them', async () => {
    const user = userEvent.setup()
    const patient = DEMO_PATIENT_RECORDS[0]!
    renderPage(
      createInitialPatientWorkflowState([patient], '2026-07-27'),
    )

    const identity = screen.getByRole('region', {
      name: 'Patient identity',
    })
    expect(identity).toHaveTextContent(patient.displayName)
    expect(identity).toHaveTextContent(patient.recordNumber)
    const details = within(identity)
      .getByText('More patient details')
      .closest('details')
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute('open')
    expect(within(identity).getByText('Date of birth')).not.toBeVisible()

    await user.click(within(identity).getByText('More patient details'))

    expect(details).toHaveAttribute('open')
    expect(within(identity).getByText('Date of birth')).toBeVisible()
    expect(within(identity).getByText('Care pathway')).toBeVisible()
  })

  it('shows patient identity and blocks a missing timepoint and future visit date', async () => {
    const user = userEvent.setup()
    const patient = DEMO_PATIENT_RECORDS[0]!
    const view = renderPage(
      createInitialPatientWorkflowState([patient], '2026-07-27'),
    )
    expect(view.container.querySelector('main')).toBeNull()

    expect(
      screen.getByRole('heading', { name: 'Add photo visit', level: 1 }),
    ).toBeVisible()
    expect(
      screen.getByRole('region', { name: 'Patient identity' }),
    ).toHaveTextContent(patient.displayName)
    const timepoint = screen.getByRole('combobox', {
      name: 'Timepoint',
    })
    await user.clear(screen.getByLabelText('Visit date'))
    await user.type(screen.getByLabelText('Visit date'), '2099-01-01')
    await user.click(
      screen.getByRole('button', { name: 'Continue to photo' }),
    )

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Check the highlighted fields.',
    )
    expect(screen.getByText('Timepoint is required.')).toBeVisible()
    expect(
      screen.getByText('Visit date cannot be in the future.'),
    ).toBeVisible()
    expect(timepoint).toHaveFocus()
    expect(screen.getByLabelText('Visit count')).toHaveTextContent('0')
  })

  it('clears only the corrected field error without waiting for another submit', async () => {
    const user = userEvent.setup()
    const patient = DEMO_PATIENT_RECORDS[0]!
    renderPage(
      createInitialPatientWorkflowState([patient], '2026-07-27'),
    )

    const timepoint = screen.getByRole('combobox', {
      name: 'Timepoint',
    })
    const visitDate = screen.getByLabelText('Visit date')
    await user.clear(visitDate)
    await user.type(visitDate, '2099-01-01')
    await user.click(
      screen.getByRole('button', { name: 'Continue to photo' }),
    )

    expect(timepoint).toHaveAttribute('aria-invalid', 'true')
    expect(visitDate).toHaveAttribute('aria-invalid', 'true')

    await user.selectOptions(timepoint, 'preoperative')

    expect(
      screen.queryByText('Timepoint is required.'),
    ).not.toBeInTheDocument()
    expect(timepoint).toHaveAttribute('aria-invalid', 'false')
    expect(
      screen.getByText('Visit date cannot be in the future.'),
    ).toBeVisible()
    expect(visitDate).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Check the highlighted fields.',
    )
  })

  it('creates exactly one initial visit and opens its photo screen', async () => {
    const user = userEvent.setup()
    const patient = DEMO_PATIENT_RECORDS[0]!
    renderPage(
      createInitialPatientWorkflowState([patient], '2026-07-27'),
    )

    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Timepoint' }),
      'postoperative',
    )
    await user.clear(screen.getByLabelText('Visit date'))
    await user.type(screen.getByLabelText('Visit date'), '2026-07-26')
    await user.click(
      screen.getByRole('button', { name: 'Continue to photo' }),
    )

    expect(
      screen.getByRole('status', { name: 'Current location' }),
    ).toHaveTextContent(
      `/patients/${patient.id}/visits/visit-visit_new_visit`,
    )
    expect(screen.getByLabelText('Visit count')).toHaveTextContent('1')
  })

  it('adds exactly one later visit to an existing timeline', async () => {
    const user = userEvent.setup()
    const patient = DEMO_PATIENT_RECORDS[0]!
    renderPage(stateWithOneVisit())

    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Timepoint' }),
      'follow_up',
    )
    await user.clear(screen.getByLabelText('Visit date'))
    await user.type(screen.getByLabelText('Visit date'), '2026-07-27')
    await user.click(
      screen.getByRole('button', { name: 'Continue to photo' }),
    )

    expect(screen.getByLabelText('Visit count')).toHaveTextContent('2')
    expect(
      screen.getByRole('status', { name: 'Current location' }),
    ).toHaveTextContent(
      `/patients/${patient.id}/visits/visit-visit_new_visit`,
    )
  })

  it('fails closed for an unknown patient', () => {
    renderPage(createInitialPatientWorkflowState(), 'patient-unknown')

    expect(
      screen.getByRole('heading', {
        name: 'Patient record unavailable',
        level: 1,
      }),
    ).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'Back to patients' }),
    ).toHaveAttribute('href', '/patients')
    expect(
      screen.queryByRole('button', { name: 'Continue to photo' }),
    ).not.toBeInTheDocument()
  })
})
