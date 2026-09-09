import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import {
  PatientWorkflowProvider,
  usePatientWorkflow,
  type PatientWorkflowRuntime,
} from '../patientWorkflow/PatientWorkflowProvider'
import { createInitialPatientWorkflowState } from '../patientWorkflow/reducer'
import { NewPatientPage } from './NewPatientPage'

const runtime: PatientWorkflowRuntime = {
  nextIdToken: (kind) => `${kind}_new_patient`,
  now: () => '2026-07-27T14:00:00.000Z',
  today: () => '2026-07-27',
}

function Destination() {
  const location = useLocation()
  return <output aria-label="Current location">{location.pathname}</output>
}

function WorkflowCounts() {
  const { state } = usePatientWorkflow()
  return (
    <output aria-label="Workflow counts">
      {state.patientOrder.length} patients, {state.visitOrder.length} visits
    </output>
  )
}

function renderPage(
  initialState = createInitialPatientWorkflowState(),
) {
  return render(
    <MemoryRouter initialEntries={['/patients/new']}>
      <PatientWorkflowProvider
        initialState={initialState}
        runtime={runtime}
      >
        <Routes>
          <Route path="/patients/new" element={<NewPatientPage />} />
          <Route
            path="/patients/:patientId/visits/:visitId"
            element={<Destination />}
          />
        </Routes>
        <WorkflowCounts />
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

async function fillValidPatientForm(
  user: ReturnType<typeof userEvent.setup>,
) {
  await user.type(
    screen.getByRole('textbox', { name: 'Display name' }),
    'Synthetic Test Record',
  )
  await user.type(
    screen.getByRole('textbox', { name: 'Record or study ID' }),
    ' study 204 ',
  )
  await user.type(
    screen.getByLabelText('Date of birth'),
    '1984-04-12',
  )
  await user.selectOptions(
    screen.getByRole('combobox', { name: 'Care pathway' }),
    'facial_reconstruction',
  )
  await user.selectOptions(
    screen.getByRole('combobox', { name: 'First visit timepoint' }),
    'preoperative',
  )
  await user.clear(screen.getByLabelText('First visit date'))
  await user.type(
    screen.getByLabelText('First visit date'),
    '2026-07-20',
  )
}

describe('NewPatientPage', () => {
  it('warns before the first field, keeps date autofill off, and offers a clear way back', () => {
    renderPage()

    const dataBoundary = screen.getByText(
      'Only synthetic or test information may be entered. Do not enter real patient information.',
    )
    const firstField = screen.getByRole('textbox', { name: 'Display name' })
    expect(dataBoundary).toBeVisible()
    expect(
      dataBoundary.compareDocumentPosition(firstField) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(screen.getByLabelText('Date of birth')).toHaveAttribute(
      'autocomplete',
      'off',
    )
    expect(screen.getByLabelText('First visit date')).toHaveAttribute(
      'autocomplete',
      'off',
    )
    expect(screen.getByRole('link', { name: 'Cancel' })).toHaveAttribute(
      'href',
      '/patients',
    )
  })

  it('creates one test record with its initial visit and opens the photo workflow', async () => {
    const user = userEvent.setup()
    const view = renderPage()
    expect(view.container.querySelector('main')).toBeNull()
    await fillValidPatientForm(user)

    expect(
      screen.getByText(
        'Only synthetic or test information may be entered. Do not enter real patient information.',
      ),
    ).toBeVisible()
    await user.click(
      screen.getByRole('checkbox', {
        name: /I confirm this record contains synthetic\/test information only/i,
      }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Save and add photo' }),
    )

    expect(
      screen.getByRole('status', { name: 'Current location' }),
    ).toHaveTextContent(
      '/patients/patient-patient_new_patient/visits/visit-visit_new_patient',
    )
    expect(screen.getByLabelText('Workflow counts')).toHaveTextContent(
      '1 patients, 1 visits',
    )
  })

  it('shows inline duplicate, future-date, timepoint, and attestation errors and focuses the first invalid field', async () => {
    const user = userEvent.setup()
    renderPage(
      createInitialPatientWorkflowState(
        [DEMO_PATIENT_RECORDS[0]!],
        '2026-07-27',
      ),
    )

    await user.type(
      screen.getByRole('textbox', { name: 'Display name' }),
      'Synthetic Duplicate',
    )
    const recordNumber = screen.getByRole('textbox', {
      name: 'Record or study ID',
    })
    await user.type(recordNumber, 'demo 1001')
    await user.type(screen.getByLabelText('Date of birth'), '2099-01-01')
    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Care pathway' }),
      'facial_paralysis',
    )
    await user.click(
      screen.getByRole('button', { name: 'Save and add photo' }),
    )

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Check the highlighted fields.',
    )
    expect(
      screen.getByText(
        'Record number is already in use in this session.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText('Date of birth cannot be in the future.'),
    ).toBeVisible()
    expect(screen.getByText('Timepoint is required.')).toBeVisible()
    expect(
      screen.getByText(
        'Confirm that only synthetic/test information is being entered.',
      ),
    ).toBeVisible()
    expect(recordNumber).toHaveFocus()
    expect(screen.getByLabelText('Workflow counts')).toHaveTextContent(
      '1 patients, 0 visits',
    )
  })

  it('clears only the corrected field error without waiting for another submit', async () => {
    const user = userEvent.setup()
    renderPage()

    const displayName = screen.getByRole('textbox', {
      name: 'Display name',
    })
    const recordNumber = screen.getByRole('textbox', {
      name: 'Record or study ID',
    })
    await user.click(
      screen.getByRole('button', { name: 'Save and add photo' }),
    )

    expect(displayName).toHaveAttribute('aria-invalid', 'true')
    expect(recordNumber).toHaveAttribute('aria-invalid', 'true')

    await user.type(displayName, 'Synthetic Test Record')

    expect(
      screen.queryByText('Display name is required.'),
    ).not.toBeInTheDocument()
    expect(displayName).toHaveAttribute('aria-invalid', 'false')
    expect(
      screen.getByText('Record number is required.'),
    ).toBeVisible()
    expect(recordNumber).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Check the highlighted fields.',
    )
  })
})
