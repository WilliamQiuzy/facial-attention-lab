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
  patientSamplePhotoPairs,
  type PatientSamplePhotoAsset,
} from '../data/patientSamplePhotoPair'
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
  PatientFaceRegistration,
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
const activePatientSamplePhotoAssets =
  patientSamplePhotoPairs['patient-demo-001']
const visitId = createPatientVisitId('visit-task-5-primary')
const DEFAULT_SHA = 'a'.repeat(64)
const qualityLabels = [
  'Full face is visible and centered',
  'Focus, lighting, and occlusion are acceptable',
  'Patient left/right orientation is confirmed and the image is not mirrored',
  'Photography authorization is documented for this visit',
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

function loadSyntheticMedia(
  asset: Pick<PatientSamplePhotoAsset, 'sha256'> =
    activePatientSamplePhotoAssets.preoperative,
) {
  const media = new Blob(['approved synthetic'], {
    type: 'image/png',
  }) as TaggedMedia
  Object.defineProperty(media, 'testSha256', {
    value: asset.sha256,
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

function validFaceRegistration(): PatientFaceRegistration {
  const closedTriangle = [
    { x: 0.25, y: 0.25 },
    { x: 0.5, y: 0.82 },
    { x: 0.75, y: 0.25 },
  ] as const

  return {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: DEFAULT_SHA,
    sourceWidth: 1_024,
    sourceHeight: 900,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    detectorId: 'mediapipe_face_landmarker',
    detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
    faceCount: 1,
    paths: [
      { feature: 'face_oval', closed: true, points: closedTriangle },
      { feature: 'left_eye', closed: true, points: closedTriangle },
      { feature: 'right_eye', closed: true, points: closedTriangle },
      {
        feature: 'left_eyebrow',
        closed: false,
        points: closedTriangle,
      },
      {
        feature: 'right_eyebrow',
        closed: false,
        points: closedTriangle,
      },
      { feature: 'lips', closed: true, points: closedTriangle },
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
        faceRegistrationRunner={async (input) => ({
          ...validFaceRegistration(),
          captureSha256: input.captureSha256,
          sourceWidth: input.sourceWidth,
          sourceHeight: input.sourceHeight,
          captureProtocol: input.captureProtocol,
        })}
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

function stateWithPairedPreoperativeSampleAndPostoperativeVisit(): PatientWorkflowState {
  const preoperativeVisitId = createPatientVisitId(
    'visit-task-5-paired-preoperative',
  )
  let state = createInitialPatientWorkflowState(
    [patient],
    '2026-07-27',
  )
  state = patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: preoperativeVisitId,
      patientId: patient.id,
      timepoint: 'preoperative',
      visitDate: '2026-07-20',
      createdAt: '2026-07-20T13:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
  state = patientWorkflowReducer(state, {
    type: 'visit/create',
    visit: {
      id: visitId,
      patientId: patient.id,
      timepoint: 'postoperative',
      visitDate: '2026-07-27',
      createdAt: '2026-07-27T13:00:00.000Z',
    },
    trustedToday: '2026-07-27',
  })
  return patientWorkflowReducer(state, {
    type: 'capture/add',
    capture: {
      id: createCaptureAssetId('capture-task-5-paired-preoperative'),
      patientId: patient.id,
      visitId: preoperativeVisitId,
      version: 1,
      status: 'current',
      source: 'synthetic_demo',
      mediaHandle: createSessionMediaHandle(
        'paired_preoperative_media_001',
      ),
      sha256: activePatientSamplePhotoAssets.preoperative.sha256,
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
      capturedAt: '2026-07-20T13:01:00.000Z',
      syntheticSourceAssetId:
        activePatientSamplePhotoAssets.preoperative.id,
    },
  })
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
  it('shows patient identity and Camera, Upload photo, and Sample photo choices', async () => {
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

    expect(screen.getByRole('button', { name: 'Camera' })).toBeVisible()
    expect(screen.getByLabelText('Upload photo')).toHaveAttribute(
      'accept',
      'image/jpeg,image/png,image/webp',
    )

    await user.click(
      screen.getByRole('button', {
        name: 'Sample photo',
      }),
    )

    expect(loadSynthetic).toHaveBeenCalledWith(
      activePatientSamplePhotoAssets.preoperative,
    )
    const qualityHeading = await screen.findByRole('heading', {
      name: 'Photo quality confirmation',
    })
    expect(qualityHeading).toBeVisible()
    expect(qualityHeading).toHaveFocus()
    expect(
      screen.getByRole('img', { name: 'Current frontal photograph' }),
    ).toHaveAttribute('src', 'blob:patient-visit-1')
    expect(
      screen.getByText(
        activePatientSamplePhotoAssets.preoperative.disclosure,
      ),
    ).toBeVisible()
    expect(
      screen.queryByRole('button', {
        name: 'Sample photo',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(
        'This visit already uses a sample photo. Use Camera or Upload photo to replace it.',
      ),
    ).toBeVisible()
  })

  it('does not offer catalog demo media as a second longitudinal visit', () => {
    renderVisit({
      initialState: stateWithSyntheticCaptureOnAnotherVisit(),
    })

    expect(
      screen.queryByRole('button', {
        name: 'Sample photo',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByText(
        'This record already uses a different sample photo. Use Camera or Upload photo to keep the same patient across visits.',
      ),
    ).toBeVisible()
  })

  it('offers the matching postoperative sample after the paired preoperative sample is used', async () => {
    const user = userEvent.setup()
    const loadSynthetic = vi.fn(loadSyntheticMedia)
    renderVisit({
      initialState: stateWithPairedPreoperativeSampleAndPostoperativeVisit(),
      loadSyntheticMedia: loadSynthetic,
    })

    await user.click(
      screen.getByRole('button', {
        name: 'Sample photo',
      }),
    )

    expect(loadSynthetic).toHaveBeenCalledWith(
      activePatientSamplePhotoAssets.postoperative,
    )
    expect(
      await screen.findByRole('heading', {
        name: 'Photo quality confirmation',
      }),
    ).toBeVisible()
  })

  it('shows a calm, explicit busy state while a photograph is being prepared', async () => {
    const user = userEvent.setup()
    let resolveSynthetic!: (media: Blob) => void
    const pendingSynthetic = new Promise<Blob>((resolve) => {
      resolveSynthetic = resolve
    })
    renderVisit({
      loadSyntheticMedia: () => pendingSynthetic,
    })

    await user.click(
      screen.getByRole('button', {
        name: 'Sample photo',
      }),
    )

    const captureRegion = screen.getByRole('region', {
      name: 'Add frontal photograph',
    })
    expect(captureRegion).toHaveAttribute('aria-busy', 'true')
    expect(
      within(captureRegion).getByRole('status', {
        name: 'Photograph preparation status',
      }),
    ).toHaveTextContent('Preparing photograph…')
    expect(
      within(captureRegion).getByText(
        'Checking the image before it is added to this visit.',
      ),
    ).toBeVisible()
    expect(
      within(captureRegion).getByRole('button', { name: 'Camera' }),
    ).toBeDisabled()
    expect(within(captureRegion).getByLabelText('Upload photo')).toBeDisabled()
    expect(
      within(captureRegion).getByRole('button', {
        name: 'Sample photo',
      }),
    ).toBeDisabled()

    resolveSynthetic(await loadSyntheticMedia())

    expect(
      await screen.findByRole('heading', {
        name: 'Photo quality confirmation',
      }),
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
    renderVisit({ queueDelayMs: 250, analysisDelayMs: 60 })
    await attachUpload(user)

    const run = screen.getByRole('button', {
      name: 'Run analysis',
    })
    expect(run).toBeDisabled()
    expect(screen.getAllByRole('checkbox')).toHaveLength(4)

    await confirmQuality(user)
    expect(
      screen.getByRole('button', {
        name: 'Run analysis',
      }),
    ).toBeEnabled()
    await user.click(
      screen.getByRole('button', {
        name: 'Run analysis',
      }),
    )

    const progress = screen.getByRole('region', {
      name: 'Analysis progress',
    })
    expect(progress).toHaveAttribute('aria-busy', 'true')
    const progressHeading = within(progress).getByRole('heading', {
      name: 'Preparing result',
    })
    expect(progressHeading).toHaveAttribute('tabindex', '-1')
    expect(progressHeading).toHaveFocus()
    expect(
      within(progress).getByRole('img', {
        name: 'Photograph being analyzed',
      }),
    ).toBeVisible()
    expect(
      within(progress).getByRole('progressbar', {
        name: 'Analysis completion',
      }),
    ).toHaveAttribute('value', '2')
    expect(
      within(progress).getByText(
        'Keep this page open. No action is needed.',
      ),
    ).toBeVisible()
    const phases = within(progress).getByRole('list')
    expect(within(phases).getAllByRole('listitem')).toHaveLength(4)
    expect(within(phases).getByText('Photo received')).toBeVisible()
    expect(within(phases).getByText('Quality confirmed')).toBeVisible()
    expect(within(phases).getByText('Analysis')).toBeVisible()
    expect(within(phases).getByText('Result prepared')).toBeVisible()
    expect(within(progress).queryByRole('status')).not.toBeInTheDocument()
    const analysisAnnouncement = screen.getByRole('status', {
      name: 'Analysis status announcement',
    })
    expect(progress).not.toContainElement(analysisAnnouncement)
    expect(analysisAnnouncement).toHaveTextContent('Analysis queued')
    const queuedAnalysisPhase = within(phases)
      .getByText('Analysis')
      .closest('li')
    expect(queuedAnalysisPhase).toHaveTextContent('Queued')
    expect(progress).toHaveTextContent(
      'Waiting for analysis to begin…',
    )
    expect(progress).not.toHaveTextContent('Starting')
    expect(progress).not.toHaveTextContent('Analysis running')
    expect(queuedAnalysisPhase).toHaveAttribute('aria-current', 'step')

    await waitFor(() => {
      expect(analysisAnnouncement).toHaveTextContent(
        'Analysis running',
      )
    })
    expect(queuedAnalysisPhase).toHaveTextContent('In progress')
    expect(queuedAnalysisPhase).toHaveAttribute('aria-current', 'step')
    const resultHeading = await screen.findByRole('heading', {
      name: 'Review result',
    })
    expect(resultHeading).toBeVisible()
    expect(resultHeading).toHaveFocus()
  })

  it('presents the image-first result with a contour matched to the uploaded photograph', async () => {
    const user = userEvent.setup()
    const { container } = renderVisit()
    await attachUpload(user)
    await confirmQuality(user)
    await user.click(
      screen.getByRole('button', {
        name: 'Run analysis',
      }),
    )
    await screen.findByRole('heading', { name: 'Review result' })

    const original = screen.getByRole('heading', {
      name: 'Original photograph',
    })
    const overlay = screen.getByRole('heading', {
      name: 'Attention overlay',
    })
    const density = screen.getByRole('heading', {
      name: 'Attention density + matched face contour',
    })
    const aoi = screen.getByRole('heading', {
      name: 'Attention by facial area',
    })
    const reviewDecision = screen.getByText('Review decision')

    expect(
      original.compareDocumentPosition(overlay) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      overlay.compareDocumentPosition(density) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      density.compareDocumentPosition(reviewDecision) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    expect(
      reviewDecision.compareDocumentPosition(aoi) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    const densityGraphic = screen.getByRole('img', {
      name: "Illustrative attention density aligned to this photograph's estimated face contour",
    })
    expect(densityGraphic).toHaveStyle({ aspectRatio: '1024 / 900' })
    expect(densityGraphic).toHaveAccessibleDescription(
      'Automatically estimated from this photograph for spatial reference. It is not a defect boundary, clinical segmentation, or attention prediction.',
    )
    expect(
      container.querySelector('.face-reference-outline'),
    ).not.toBeInTheDocument()
    expect(
      container.querySelector('.patient-face-contour'),
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        'Automatically estimated from this photograph for spatial reference. It is not a defect boundary, clinical segmentation, or attention prediction.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Patient right is viewer left; patient left is viewer right. Orientation confirmed for this frontal, non-mirrored photograph.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Face-relative areas summarize this attention density. They do not change the analysis.',
      ),
    ).toBeVisible()
    expect(
      screen.queryByText(
        'AOI is a post-inference summary only. It does not crop the photograph or alter the simulation.',
      ),
    ).not.toBeInTheDocument()
    const percentageDetails = screen
      .getByText('How percentages are calculated')
      .closest('details')
    expect(percentageDetails).not.toBeNull()
    expect(percentageDetails).not.toHaveAttribute('open')
    expect(
      screen.getByText(
        'Percentages use face-relative areas positioned within the face contour estimated from this photograph. The contour is a spatial reference, not clinical anatomical segmentation.',
      ),
    ).not.toBeVisible()
    expect(
      screen.getByText(
        'Four face-relative bands plus density outside those bands total 100%. Patient-right and patient-left shares form a separate 100% partition.',
      ),
    ).not.toBeVisible()
    expect(
      screen.getByText(
        'Illustrative estimate of where observers may attend. Not measured eye-tracking, diagnosis, treatment guidance, or evidence of surgical outcome.',
      ),
    ).toBeVisible()
    expect(
      screen.getByText(
        'Illustrative engine only: this field is seeded by the capture hash and positioned using on-device face landmarks. It does not detect this person’s scar and was not produced by the checked-in facial-paralysis model.',
      ),
    ).not.toBeVisible()
    const engineBoundary = screen.getByText(
      'Illustrative engine only: this field is seeded by the capture hash and positioned using on-device face landmarks. It does not detect this person’s scar and was not produced by the checked-in facial-paralysis model.',
    )
    expect(
      density.compareDocumentPosition(engineBoundary) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

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
        name: 'Run analysis',
      }),
    )
    await screen.findByRole('heading', { name: 'Review result' })

    await user.click(
      screen.getByRole('button', { name: 'Save decision' }),
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Choose Accept photo for comparison or Request a new photo.',
    )
    expect(
      screen.getByRole('radio', { name: 'Accept photo for comparison' }),
    ).toHaveFocus()

    await user.click(
      screen.getByRole('radio', { name: 'Request a new photo' }),
    )
    await user.click(
      screen.getByRole('button', { name: 'Save decision' }),
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Enter a reason for requesting a new photo.',
    )
    expect(
      screen.getByLabelText('Reason for requesting a new photo'),
    ).toHaveFocus()
    expect(
      screen.getByLabelText('Reason for requesting a new photo'),
    ).toHaveAttribute('name', 'reviewNote')
    expect(
      screen.getByLabelText('Reason for requesting a new photo'),
    ).toHaveAttribute('autocomplete', 'off')

    await user.type(
      screen.getByLabelText('Reason for requesting a new photo'),
      'Lighting was uneven.',
    )
    await user.click(
      screen.getByRole('button', { name: 'Save decision' }),
    )

    expect(
      screen.getByRole('heading', { name: 'Add replacement photo' }),
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

  it('reopens a completed visit as a read-only single-visit image record', async () => {
    const user = userEvent.setup()
    renderVisit()
    await attachUpload(user)
    await confirmQuality(user)
    await user.click(
      screen.getByRole('button', {
        name: 'Run analysis',
      }),
    )
    await screen.findByRole('heading', { name: 'Review result' })

    await user.click(
      screen.getByRole('radio', {
        name: 'Accept photo for comparison',
      }),
    )
    await user.type(
      screen.getByLabelText('Review note (optional)'),
      'Photo accepted for the longitudinal comparison.',
    )
    await user.click(
      screen.getByRole('button', { name: 'Save decision' }),
    )

    expect(
      screen.getByRole('heading', {
        name: 'Preoperative visit record',
      }),
    ).toHaveFocus()
    expect(screen.getByRole('status')).toHaveTextContent(
      'Decision saved',
    )
    const savedReview = screen.getByRole('region', {
      name: 'Saved review',
    })
    expect(
      within(savedReview).getByText('Accepted for comparison'),
    ).toBeVisible()
    expect(
      within(savedReview).getByText(
        'Photo accepted for the longitudinal comparison.',
      ),
    ).toBeVisible()
    const visitImages = screen.getByRole('region', {
      name: 'Visit photo and result',
    })
    expect(
      within(visitImages).getByRole('img', {
        name: 'Original frontal photograph',
      }),
    ).toBeVisible()
    expect(
      within(visitImages).getByRole('img', {
        name: 'Frontal photograph with illustrative attention overlay',
      }),
    ).toBeVisible()
    const additionalDetails = screen
      .getByText('Additional details')
      .closest('details')
    expect(additionalDetails).not.toBeNull()
    expect(additionalDetails).not.toHaveAttribute('open')
    expect(
      screen.getByRole('img', {
        name: "Illustrative attention density aligned to this photograph's estimated face contour",
      }),
    ).not.toBeVisible()
    expect(
      screen.queryByRole('link', { name: 'Back to reviews' }),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: 'Return to patient record' }),
    ).toHaveAttribute('href', `/patients/${patient.id}`)
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
        name: 'Run analysis',
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

  it('gives a simple retake instruction when the uploaded face cannot be registered', async () => {
    const user = userEvent.setup()
    renderVisit({
      faceRegistrationRunner: async () => {
        throw new Error('NO_FACE')
      },
    })
    await attachUpload(user)
    await confirmQuality(user)
    await user.click(
      screen.getByRole('button', {
        name: 'Run analysis',
      }),
    )

    const alignmentHeading = await screen.findByRole('heading', {
      name: 'Face alignment needs attention',
    })
    expect(alignmentHeading).toBeVisible()
    expect(alignmentHeading).toHaveFocus()
    expect(
      screen.getByText(
        'We could not match one clear face to this photograph. Retake or upload a centered frontal image with only one face visible.',
      ),
    ).toBeVisible()
    expect(
      screen.getByRole('heading', {
        name: 'Replace photograph',
      }),
    ).toBeVisible()
    expect(document.body).not.toHaveTextContent(
      /mediapipe|landmark|wasm|NO_FACE/i,
    )
  })
})
