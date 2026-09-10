import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { DEMO_PATIENT_RECORDS } from '../data/demoPatientRecords'
import type {
  CaptureAsset,
  PatientComparisonResultEntry,
  PatientFaceRegistration,
  PatientResult,
  PatientRunBinding,
  PatientVisit,
} from '../patientWorkflow/types'
import {
  createCaptureAssetId,
  createPatientResultId,
  createPatientRunId,
  createPatientVisitId,
  createSessionMediaHandle,
} from '../patientWorkflow/validation'
import { PatientLongitudinalComparison } from './PatientLongitudinalComparison'

function comparisonEntry(
  timepoint: 'preoperative' | 'postoperative',
  suffix: string,
  date: string,
): PatientComparisonResultEntry & { readonly previewUrl: string } {
  const patient = DEMO_PATIENT_RECORDS[0]!
  const visit: PatientVisit = {
    id: createPatientVisitId(`visit-longitudinal-${suffix}`),
    patientId: patient.id,
    timepoint,
    visitDate: date,
    createdAt: `${date}T10:00:00.000Z`,
  }
  const capture: CaptureAsset = {
    id: createCaptureAssetId(`capture-longitudinal-${suffix}`),
    patientId: patient.id,
    visitId: visit.id,
    version: 1,
    status: 'current',
    source: 'upload',
    mediaHandle: createSessionMediaHandle(`media_longitudinal_${suffix}`),
    sha256: suffix.padEnd(64, 'a').slice(0, 64),
    mimeType: 'image/png',
    sizeBytes: 1_024,
    width: 1_024,
    height: 1_024,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    qualityChecks: {
      faceVisibleAndCentered: true,
      focusLightingAndOcclusionAcceptable: true,
      orientationConfirmed: true,
      authorizationDocumented: true,
    },
    capturedAt: `${date}T10:01:00.000Z`,
    qualityConfirmedAt: `${date}T10:02:00.000Z`,
  }
  const binding: PatientRunBinding = {
    patientId: patient.id,
    visitId: visit.id,
    captureId: capture.id,
    captureVersion: capture.version,
    captureSha256: capture.sha256,
    mediaHandle: capture.mediaHandle,
    authorizationRevision: 1,
    captureProtocol: capture.captureProtocol,
  }
  const facePoints = [
    { x: 0.3, y: 0.3 },
    { x: 0.5, y: 0.75 },
    { x: 0.7, y: 0.3 },
  ] as const
  const registration: PatientFaceRegistration = {
    schemaVersion: 'patient-face-registration/1',
    source: 'on_device_face_landmarks',
    coordinateSpace: 'decoded_image_normalized_v1',
    captureSha256: capture.sha256,
    sourceWidth: capture.width,
    sourceHeight: capture.height,
    captureProtocol: capture.captureProtocol,
    detectorId: 'mediapipe_face_landmarker',
    detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
    faceCount: 1,
    paths: [
      { feature: 'face_oval', closed: true, points: facePoints },
      { feature: 'left_eye', closed: true, points: facePoints },
      { feature: 'right_eye', closed: true, points: facePoints },
      { feature: 'left_eyebrow', closed: false, points: facePoints },
      { feature: 'right_eyebrow', closed: false, points: facePoints },
      { feature: 'lips', closed: true, points: facePoints },
    ],
  }
  const result: PatientResult = {
    id: createPatientResultId(`result-longitudinal-${suffix}`),
    runId: createPatientRunId(`run-longitudinal-${suffix}`),
    binding,
    freshness: 'current',
    createdAt: `${date}T10:04:00.000Z`,
    faceRegistration: registration,
    output: {
      origin: 'workflow_simulation',
      points: [
        { x: 0.42, y: 0.5, intensity: 0.9, radius: 0.12 },
      ],
    },
  }
  return {
    visit,
    capture,
    result,
    previewUrl: `blob:${suffix}`,
  }
}

describe('PatientLongitudinalComparison', () => {
  it('keeps the two current visits together and switches between photo and outline views', async () => {
    const user = userEvent.setup()
    const preoperative = comparisonEntry(
      'preoperative',
      'pre',
      '2026-07-20',
    )
    const postoperative = comparisonEntry(
      'postoperative',
      'post',
      '2026-07-25',
    )
    const view = render(
      <PatientLongitudinalComparison
        pair={{ preoperative, postoperative }}
      />,
    )

    const comparison = screen.getByRole('region', {
      name: 'Patient before and after comparison',
    })
    expect(
      within(comparison).getByRole('heading', {
        name: 'Before and after',
        level: 2,
      }),
    ).toBeVisible()
    expect(
      within(comparison).getByText('Latest pre- and postoperative visits'),
    ).toBeVisible()
    expect(within(comparison).getByText('Jul 20, 2026')).toBeVisible()
    expect(within(comparison).getByText('Jul 25, 2026')).toBeVisible()

    const prePhoto = within(comparison).getByRole('img', {
      name: 'Preoperative patient photograph with illustrative attention',
    })
    const postPhoto = within(comparison).getByRole('img', {
      name: 'Postoperative patient photograph with illustrative attention',
    })
    expect(prePhoto).toHaveAttribute('src', 'blob:pre')
    expect(postPhoto).toHaveAttribute('src', 'blob:post')
    const photoMode = within(comparison).getByRole('radio', {
      name: 'Photo',
    })
    const outlineMode = within(comparison).getByRole('radio', {
      name: 'Outline',
    })
    expect(photoMode).toBeChecked()
    expect(
      within(comparison).getByRole('checkbox', {
        name: 'Show attention layer',
      }),
    ).toBeChecked()
    expect(
      view.container.querySelectorAll(
        '.patient-longitudinal-comparison__attention-layer',
      ),
    ).toHaveLength(2)

    photoMode.focus()
    await user.keyboard('{ArrowRight}')
    expect(outlineMode).toBeChecked()
    expect(
      within(comparison).queryByRole('img', {
        name: 'Preoperative patient photograph with illustrative attention',
      }),
    ).not.toBeInTheDocument()
    expect(
      within(comparison).getByRole('img', {
        name: 'Preoperative face outline with illustrative attention',
      }),
    ).toBeVisible()
    expect(
      within(comparison).getByRole('img', {
        name: 'Postoperative face outline with illustrative attention',
      }),
    ).toBeVisible()

    await user.click(
      within(comparison).getByRole('checkbox', {
        name: 'Show attention layer',
      }),
    )
    expect(
      view.container.querySelectorAll(
        '.patient-longitudinal-comparison__attention-layer',
      ),
    ).toHaveLength(0)
    expect(
      within(comparison).getByRole('img', {
        name: 'Preoperative face outline',
      }),
    ).toBeVisible()
    expect(
      within(comparison).queryByRole('img', {
        name: 'Preoperative face outline with illustrative attention',
      }),
    ).not.toBeInTheDocument()

    await user.click(
      within(comparison).getByRole('radio', { name: 'Photo' }),
    )
    expect(
      within(comparison).getByRole('img', {
        name: 'Preoperative patient photograph',
      }),
    ).toBeVisible()
  })

  it('keeps the simulation and clinical-use boundary visible', () => {
    render(
      <PatientLongitudinalComparison
        pair={{
          preoperative: comparisonEntry(
            'preoperative',
            'pre',
            '2026-07-20',
          ),
          postoperative: comparisonEntry(
            'postoperative',
            'post',
            '2026-07-25',
          ),
        }}
      />,
    )

    expect(
      screen.getByText(
        'Illustrative workflow output—not measured gaze, a clinical measurement, or evidence of treatment effect.',
      ),
    ).toBeVisible()
  })
})
