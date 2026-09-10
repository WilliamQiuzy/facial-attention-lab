import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { PatientFaceRegistration } from '../patientWorkflow/types'
import { PatientAoiSummary } from './PatientAoiSummary'

const faceRegistration: PatientFaceRegistration = {
  schemaVersion: 'patient-face-registration/1',
  source: 'on_device_face_landmarks',
  coordinateSpace: 'decoded_image_normalized_v1',
  captureSha256: 'a'.repeat(64),
  sourceWidth: 1024,
  sourceHeight: 1024,
  captureProtocol: 'frontal_relaxed_non_mirrored_v1',
  detectorId: 'mediapipe_face_landmarker',
  detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
  faceCount: 1,
  paths: [
    {
      feature: 'face_oval',
      closed: true,
      points: [
        { x: 0.2, y: 0.15 },
        { x: 0.8, y: 0.15 },
        { x: 0.8, y: 0.9 },
        { x: 0.2, y: 0.9 },
      ],
    },
  ],
}

describe('PatientAoiSummary', () => {
  it('does not offer percentage methodology when no summary can be calculated', () => {
    render(
      <PatientAoiSummary
        points={[]}
        faceRegistration={faceRegistration}
      />,
    )

    expect(
      screen.getByText(
        'A facial-area summary is unavailable for this result.',
      ),
    ).toBeVisible()
    expect(
      screen.queryByText('How percentages are calculated'),
    ).not.toBeInTheDocument()
  })
})
