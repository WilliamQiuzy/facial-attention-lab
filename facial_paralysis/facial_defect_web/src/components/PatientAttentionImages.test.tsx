import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PatientAttentionImages } from './PatientAttentionImages'

const attentionPoints = [
  { x: 0.45, y: 0.52, intensity: 0.8, radius: 0.08 },
] as const

const wideFaceRegistration = {
  schemaVersion: 'patient-face-registration/1',
  source: 'on_device_face_landmarks',
  coordinateSpace: 'decoded_image_normalized_v1',
  captureSha256: 'a'.repeat(64),
  sourceWidth: 1_024,
  sourceHeight: 900,
  captureProtocol: 'frontal_relaxed_non_mirrored_v1',
  detectorId: 'mediapipe_face_landmarker',
  detectorVersion: 'tasks-vision-1.0.0-model-float16-1',
  faceCount: 1,
  paths: [
    {
      feature: 'face_oval',
      closed: true,
      points: [
        { x: 0.18, y: 0.1 },
        { x: 0.1, y: 0.48 },
        { x: 0.5, y: 0.94 },
        { x: 0.9, y: 0.48 },
        { x: 0.82, y: 0.1 },
      ],
    },
    {
      feature: 'left_eye',
      closed: true,
      points: [
        { x: 0.27, y: 0.4 },
        { x: 0.41, y: 0.4 },
        { x: 0.34, y: 0.44 },
      ],
    },
  ],
} as const

const narrowFaceRegistration = {
  ...wideFaceRegistration,
  captureSha256: 'b'.repeat(64),
  paths: [
    {
      feature: 'face_oval',
      closed: true,
      points: [
        { x: 0.32, y: 0.1 },
        { x: 0.24, y: 0.48 },
        { x: 0.5, y: 0.94 },
        { x: 0.76, y: 0.48 },
        { x: 0.68, y: 0.1 },
      ],
    },
  ],
} as const

describe('PatientAttentionImages', () => {
  it('draws the contour supplied by the exact uploaded photograph instead of a fixed face template', () => {
    const { container, rerender } = render(
      <PatientAttentionImages
        previewUrl="blob:photo-a"
        width={1_024}
        height={900}
        points={attentionPoints}
        faceRegistration={wideFaceRegistration}
      />,
    )

    const contour = container.querySelector('.patient-face-contour')
    expect(contour).toHaveAttribute(
      'data-geometry-source',
      'on_device_face_landmarks',
    )
    expect(
      contour?.querySelector('[data-feature="face_oval"]'),
    ).toHaveAttribute(
      'points',
      '18,10 10,48 50,94 90,48 82,10',
    )
    expect(
      container.querySelector(
        '[data-reference="fixed-illustrative-template"]',
      ),
    ).not.toBeInTheDocument()

    rerender(
      <PatientAttentionImages
        previewUrl="blob:photo-b"
        width={1_024}
        height={900}
        points={attentionPoints}
        faceRegistration={narrowFaceRegistration}
      />,
    )

    expect(
      container.querySelector('[data-feature="face_oval"]'),
    ).toHaveAttribute(
      'points',
      '32,10 24,48 50,94 76,48 68,10',
    )
  })

  it('describes photo-derived contour provenance without calling it anatomy or model output', () => {
    const { container } = render(
      <PatientAttentionImages
        previewUrl="blob:photo"
        width={1_024}
        height={900}
        points={attentionPoints}
        faceRegistration={wideFaceRegistration}
      />,
    )

    expect(
      screen.getByRole('heading', {
        name: 'Attention density + matched face contour',
      }),
    ).toBeVisible()
    expect(
      screen.getByRole('img', {
        name: "Illustrative attention density aligned to this photograph's estimated face contour",
      }),
    ).toHaveAccessibleDescription(
      'Automatically estimated from this photograph for spatial reference. It is not a defect boundary, clinical segmentation, or attention prediction.',
    )
    expect(
      container.querySelector('.patient-face-contour'),
    ).toHaveAttribute('aria-hidden', 'true')
    expect(
      container.querySelector('.patient-face-contour'),
    ).toHaveAttribute('focusable', 'false')
  })
})
