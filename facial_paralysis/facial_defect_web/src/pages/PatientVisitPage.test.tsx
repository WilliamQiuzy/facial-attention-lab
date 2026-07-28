import {
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { approvedAssets } from '../data/approvedAssetManifest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import {
  PatientWorkflowProvider,
  type PatientWorkflowProviderProps,
  type PatientWorkflowRuntime,
  type PatientWorkflowRuntimeIdKind,
} from '../patientWorkflow/PatientWorkflowProvider'
import { SessionMediaVault } from '../patientWorkflow/SessionMediaVault'
import {
  CaptureFileValidationError,
  type CaptureFileValidationResult,
} from '../patientWorkflow/captureFile'
import {
  createInitialPatientWorkflowState,
  patientWorkflowReducer,
} from '../patientWorkflow/reducer'
import type {
  PatientRunBinding,
  PatientSimulationOutput,
  PatientWorkflowState,
} from '../patientWorkflow/types'
import {
  createCaptureAssetId,
  createPatientVisitId,
  createSessionMediaHandle,
} from '../patientWorkflow/validation'
import { PatientVisitPage } from './PatientVisitPage'

const patient = DEMO_PATIENT_RECORDS[0]!
const visitId = createPatientVisitId('visit-task-5-primary')
const DEFAULT_SHA = 'a'.repeat(64)
const qualityLabels = [
  'Full face is visible and centered',
  'Focus, lighting, and occlusion are acceptable',
  'Patient left/right orientation is confirmed and the image is not mirrored',
  'Photography and research authorization is documented for this demo workflow',
] as const

type TaggedMedia = Blob & {
  readonly testSha256?: string
}

function visitState(): PatientWorkflowState {
  const initial = createInitialPatientWorkflowState(
    [patient],
    '2026-07-27',
  )
  return patientWorkflowReducer(initial, {
    type: 'visit/create',
    visit: {
      id: visitId,
      patientId: patient.id,
      timepoint: 'preoperative',
      visitDate: '2026-07-27',
      createdAt: '2026-07-27T13:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
}

function createRuntime(): PatientWorkflowRuntime {
  const counters = new Map<PatientWorkflowRuntimeIdKind, number>()
  return {
    nextIdToken(kind) {
      const next = (counters.get(kind) ?? 0) + 1
      counters.set(kind, next)
      return `${kind}_${String(next).padStart(4, '0')}`
    },
    now: () => '2026-07-27T14:00:00.000Z',
    today: () => '2026-07-27',
  }
}

function prepareCapture(
  media: Blob,
): Promise<CaptureFileValidationResult> {
  const tagged = media as TaggedMedia
  const vaultMedia = media.slice(0, media.size, media.type) as TaggedMedia
  if (tagged.testSha256) {
    Object.defineProperty(vaultMedia, 'testSha256', {
      value: tagged.testSha256,
    })
  }
  return Promise.resolve({
    ok: true,
    value: {
      metadata: {
        sha256: tagged.testSha256 ?? DEFAULT_SHA,
        mimeType: media.type as 'image/jpeg' | 'image/png' | 'image/webp',
        sizeBytes: media.size,
        width: 1_024,
        height: 900,
      },
      vaultMedia,
    },
  })
}

function loadSyntheticMedia() {
  const media = new Blob(['approved synthetic'], {
    type: 'image/png',
  }) as TaggedMedia
  Object.defineProperty(media, 'testSha256', {
    value: approvedAssets[0]!.sha256,
  })
  return Promise.resolve(media)
}

function validOutput(
  _binding: PatientRunBinding,
): PatientSimulationOutput {
  return {
    origin: 'workflow_simulation',
    points: [
      { x: 0.3, y: 0.25, intensity: 0.6, radius: 0.09 },
      { x: 0.7, y: 0.45, intensity: 0.9, radius: 0.12 },
      { x: 0.48, y: 0.72, intensity: 0.7, radius: 0.1 },
    ],
  }
}

function createVault() {
  let sequence = 0
  return new SessionMediaVault({
    createObjectURL: () => `blob:patient-visit-${++sequence}`,
    revokeObjectURL: vi.fn(),
  })
}

type RenderVisitOptions = Omit<
  PatientWorkflowProviderProps,
  'children' | 'initialState' | 'runtime' | 'mediaVault'
> & {
  readonly initialState?: PatientWorkflowState
}

function renderVisit({
  initialState = visitState(),
  ...options
}: RenderVisitOptions = {}) {
  return render(
    <MemoryRouter
      initialEntries={[
        `/patients/${patient.id}/visits/${visitId}`,
      ]}
    >
      <PatientWorkflowProvider
        initialState={initialState}
        runtime={createRuntime()}
        mediaVault={createVault()}
        prepareCapture={prepareCapture}
        loadSyntheticMedia={loadSyntheticMedia}
        simulationRunner={validOutput}
        {...options}
      >
        <Routes>
          <Route
            path="/patients/:patientId/visits/:visitId"
            element={<PatientVisitPage />}
          />
          <Route
            path="/patients/:patientId"
            element={<p>Patient record</p>}
          />
        </Routes>
      </PatientWorkflowProvider>
    </MemoryRouter>,
  )
}

function stateWithSyntheticCaptureOnAnotherVisit(): PatientWorkflowState {
  const otherVisitId = createPatientVisitId(
    'visit-task-5-other-synthetic',
  )
  let state = patientWorkflowReducer(visitState(), {
    type: 'visit/create',
    visit: {
      id: otherVisitId,
      patientId: patient.id,
      timepoint: 'postoperative',
      visitDate: '2026-07-27',
      createdAt: '2026-07-27T13:30:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
  state = patientWorkflowReducer(state, {
    type: 'capture/add',
    capture: {
      id: createCaptureAssetId('capture-task-5-other-synthetic'),
      patientId: patient.id,
      visitId: otherVisitId,
      version: 1,
      status: 'current',
      source: 'synthetic_demo',
      mediaHandle: createSessionMediaHandle(
        'other_synthetic_media_001',
      ),
      sha256: approvedAssets[0]!.sha256,
      mimeType: 'image/png',
      sizeBytes: 1_024,
      width: 1_024,
      height: 1_024,
      captureProtocol: 'frontal_relaxed_non_mirrored_v1',
      qualityChecks: {
        faceVisibleAndCentered: false,
        focusLightingAndOcclusionAcceptable: false,
        orientationConfirmed: false,
        authorizationDocumented: false,
      },
      capturedAt: '2026-07-27T13:31:00.000Z',
      syntheticSourceAssetId: approvedAssets[0]!.id,
    },
  })
  return state
}

function testFile(name = 'test-photo.png') {
  return new File(['test photo'], name, { type: 'image/png' })
}

async function attachUpload(user: ReturnType<typeof userEvent.setup>) {
  await user.upload(
    screen.getByLabelText('Upload photo'),
    testFile(),
  )
  expect(
    await screen.findByRole('heading', {
      name: 'Photo quality confirmation',
    }),
  ).toBeVisible()
}

async function confirmQuality(user: ReturnType<typeof userEvent.setup>) {
  for (const label of qualityLabels) {
    await user.click(screen.getByRole('checkbox', { name: label }))
  }
}

describe('PatientVisitPage', () => {
  it('shows patient identity and camera, upload, and standalone synthetic capture choices', async () => {
    const user = userEvent.setup()
    const loadSynthetic = vi.fn(loadSyntheticMedia)
    renderVisit({ loadSyntheticMedia: loadSynthetic })

    const identity = screen.getByRole('region', {
      name: 'Patient identity',
    })
    expect(
      within(identity).getByRole('heading', {
        name: patient.displayName,
        level: 1,
      }),
    ).toBeVisible()
    expect(within(identity).getByText(patient.recordNumber)).toBeVisible()

    expect(screen.getByLabelText('Take photo')).toHaveAttribute(
      'accept',
      'image/*',
    )
    expect(screen.getByLabelText('Take photo')).toHaveAttribute(
      'capture',
      'user',
    )
    expect(screen.getByLabelText('Upload photo')).toHaveAttribute(
      'accept',
      'image/jpeg,image/png,image/webp',
    )

    await user.click(
      screen.getByRole('button', {
        name: 'Use synthetic demo photo',
      }),
    )

    expect(loadSynthetic).toHaveBeenCalledWith(approvedAssets[0])
    expect(
      await screen.findByRole('heading', {
        name: 'Photo quality confirmation',
      }),
    ).toBeVisible()
    expect(
      screen.getByRole('img', { name: 'Current frontal photograph' }),
    ).toHaveAttribute('src', 'blob:patient-visit-1')
    expect(
      screen.queryByRole('button', {
        name: 'Use synthetic demo photo',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(
        'This visit already uses the standalone catalog demo. Take or upload a different test image to replace it.',
      ),
    ).toBeVisible()
  })

  it('does not offer catalog demo media as a second longitudinal visit', () => {
    renderVisit({
      initialState: stateWithSyntheticCaptureOnAnotherVisit(),
    })

    expect(
      screen.queryByRole('button', {
        name: 'Use synthetic demo photo',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(
        'A standalone catalog demo is already used by another visit in this record. Upload a separate test image; catalog demos cannot establish longitudinal identity.',
      ),
    ).toBeVisible()
  })

  it('shows a readable file error without exposing the selected filename', async () => {
    const user = userEvent.setup()
    renderVisit({
      prepareCapture: async () => ({
        ok: false,
        error: new CaptureFileValidationError(
          'UNSUPPORTED_TYPE',
          'Choose a JPEG, PNG, or WebP image. Other file types are not supported.',
        ),
      }),
    })

    await user.upload(
      screen.getByLabelText('Upload photo'),
      testFile('private-study-name.png'),
    )

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(
      'Choose a JPEG, PNG, or WebP image.',
    )
    expect(alert).not.toHaveTextContent('private-study-name.png')
    expect(
      screen.getByRole('heading', { name: 'Add frontal photograph' }),
    ).toBeVisible()
  })

  it('blocks analysis until all four capture-consistency checks pass and announces four processing phases', async () => {
    const user = userEvent.setup()
    renderVisit({ queueDelayMs: 30, analysisDelayMs: 60 })
    await attachUpload(user)

    const run = screen.getByRole('button', {
      name: 'Run simulated analysis',
    })
    expect(run).toBeDisabled()
    expect(screen.getAllByRole('checkbox')).toHaveLength(4)

    await confirmQuality(user)
    expect(
      screen.getByRole('button', {
        name: 'Run simulated analysis',
      }),
    ).toBeEnabled()
    await user.click(
      screen.getByRole('button', {
        name: 'Run simulated analysis',
      }),
    )

    const progress = screen.getByRole('region', {
      name: 'Analysis progress',
    })
    expect(within(progress).getAllByRole('listitem')).toHaveLength(4)
    expect(within(progress).getByText('Photo received')).toBeVisible()
    expect(within(progress).getByText('Quality confirmed')).toBeVisible()
    expect(within(progress).getByText('Analysis running')).toBeVisible()
    expect(within(progress).getByText('Result prepared')).toBeVisible()
    expect(
      within(progress).getByRole('status'),
    ).toHaveTextContent('Analysis queued')
    const queuedAnalysisPhase = within(progress)
      .getByText('Analysis running')
      .closest('li')
    expect(queuedAnalysisPhase).toHaveTextContent('Waiting')
    expect(queuedAnalysisPhase).not.toHaveAttribute(
      'aria-current',
      'step',
    )

    await waitFor(() => {
      expect(within(progress).getByRole('status')).toHaveTextContent(
        'Analysis running',
      )
    })
    expect(queuedAnalysisPhase).toHaveTextContent('In progress')
    expect(queuedAnalysisPhase).toHaveAttribute('aria-current', 'step')
    expect(
      await screen.findByRole('heading', { name: 'Review result' }),
    ).toBeVisible()
  })

  it('presents the image-first result in original, overlay, density, and post-inference facial-area order', async () => {
    const user = userEvent.setup()
    renderVisit()
    await attachUpload(user)
    await confirmQuality(user)
    await user.click(
      screen.getByRole('button', {
        name: 'Run simulated analysis',
      }),
    )
    await screen.findByRole('heading', { name: 'Review result' })

    const original = screen.getByRole('heading', {
      name: 'Original photograph',
    })
    const overlay = screen.getByRole('heading', {
      name: 'Simulated overlay',
    })
    const density = screen.getByRole('heading', {
      name: 'Attention density',
    })
    const aoi = screen.getByRole('heading', {
      name: 'Attention by facial area',
    })

    expect(
      original.compareDocumentPosition(overlay) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      overlay.compareDocumentPosition(density) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      density.compareDocumentPosition(aoi) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      screen.getByRole('img', {
        name: 'Simulated attention density without the source photograph',
      }),
    ).toHaveStyle({ aspectRatio: '1024 / 900' })
    expect(
      screen.getByText(
        'Brighter overlap indicates more simulated attention density. Same deterministic simulation.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Patient right is viewer left; patient left is viewer right. Orientation confirmed for this frontal, non-mirrored photograph.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'AOI is a post-inference summary only. It does not crop the photograph or alter the simulation.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Percentages use a fixed illustrative face template, not detected landmarks or patient-specific anatomical registration.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Four template facial bands plus outside-template density total 100%. Patient-right and patient-left shares form a separate 100% partition.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Simulated estimate of where observers may attend. Not eye-tracking, diagnosis, severity, treatment guidance, or evidence of surgical outcome.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Demo engine only: this fixed-template field is seeded by the capture hash. It does not detect this person’s scar and was not produced by the checked-in facial-paralysis model.',
      ),
    ).toBeVisible()

    const technical = screen
      .getByText('Technical details')
      .closest('details')
    expect(technical).not.toHaveAttribute('open')
    expect(document.body).not.toHaveTextContent(
      /run-run_|result-result_|[a-f0-9]{64}/i,
    )
  })

  it('requires a repeat reason, completes the review, and opens a clean replacement capture', async () => {
    const user = userEvent.setup()
    renderVisit()
    await attachUpload(user)
    await confirmQuality(user)
    await user.click(
      screen.getByRole('button', {
        name: 'Run simulated analysis',
      }),
    )
    await screen.findByRole('heading', { name: 'Review result' })

    await user.click(
      screen.getByRole('button', { name: 'Complete review' }),
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Choose Reviewed or Repeat photo.',
    )
    expect(
      screen.getByRole('radio', { name: 'Reviewed' }),
    ).toHaveFocus()

    await user.click(
      screen.getByRole('radio', { name: 'Repeat photo' }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Complete review' }),
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Enter a reason for the repeat photo.',
    )
    expect(
      screen.getByLabelText('Reason for repeat photo'),
    ).toHaveFocus()

    await user.type(
      screen.getByLabelText('Reason for repeat photo'),
      'Lighting was uneven.',
    )
    await user.click(
      screen.getByRole('button', { name: 'Complete review' }),
    )

    expect(
      screen.getByRole('heading', { name: 'Repeat photo' }),
    ).toBeVisible()
    expect(
      screen.queryByRole('img', { name: 'Current frontal photograph' }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', {
        name: 'Retry exact photo analysis',
      }),
    ).not.toBeInTheDocument()

    await user.upload(
      screen.getByLabelText('Upload photo'),
      testFile('replacement.png'),
    )
    expect(
      await screen.findByRole('heading', {
        name: 'Photo quality confirmation',
      }),
    ).toBeVisible()
    expect(screen.getAllByRole('checkbox')).toHaveLength(4)
    for (const checkbox of screen.getAllByRole('checkbox')) {
      expect(checkbox).not.toBeChecked()
    }
  })

  it('offers an exact-photo retry for a failed current run and removes retry after capture replacement', async () => {
    const user = userEvent.setup()
    renderVisit({
      simulationRunner: vi.fn(async () => {
        throw new Error('simulated runner failure')
      }),
    })
    await attachUpload(user)
    await confirmQuality(user)
    await user.click(
      screen.getByRole('button', {
        name: 'Run simulated analysis',
      }),
    )

    expect(
      await screen.findByRole('button', {
        name: 'Retry exact photo analysis',
      }),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Retry uses this same photograph and confirmed quality record.',
      ),
    ).toBeVisible()

    await user.upload(
      screen.getByLabelText('Upload photo'),
      testFile('new-current-capture.png'),
    )
    expect(
      await screen.findByRole('heading', {
        name: 'Photo quality confirmation',
      }),
    ).toBeVisible()
    expect(
      screen.queryByRole('button', {
        name: 'Retry exact photo analysis',
      }),
    ).not.toBeInTheDocument()
  })
})
