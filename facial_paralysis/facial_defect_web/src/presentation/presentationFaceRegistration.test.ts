import { describe, expect, it } from 'vitest'
import { presentationDemoAssets } from '../data/presentationDemoAssets'
import { registrationByTimepoint } from './presentationFaceRegistration'

const expectedFeatures = [
  'face_oval',
  'left_eye',
  'right_eye',
  'left_eyebrow',
  'right_eyebrow',
  'lips',
] as const

describe('presentation face registration snapshots', () => {
  it.each(['preoperative', 'postoperative'] as const)(
    'binds the %s contour to its exact image and MediaPipe provenance',
    (timepoint) => {
      const registration = registrationByTimepoint[timepoint]
      const asset = presentationDemoAssets[timepoint]

      expect(registration.captureSha256).toBe(asset.sha256)
      expect(registration.sourceWidth).toBe(asset.width)
      expect(registration.sourceHeight).toBe(asset.height)
      expect(registration.source).toBe('on_device_face_landmarks')
      expect(registration.detectorId).toBe(
        'mediapipe_face_landmarker',
      )
      expect(registration.faceCount).toBe(1)
      expect(registration.paths).toHaveLength(13)
      expect(
        new Set(registration.paths.map((path) => path.feature)),
      ).toEqual(new Set(expectedFeatures))

      for (const path of registration.paths) {
        expect(path.points.length).toBeGreaterThan(1)
        for (const point of path.points) {
          expect(Number.isFinite(point.x)).toBe(true)
          expect(Number.isFinite(point.y)).toBe(true)
          expect(point.x).toBeGreaterThanOrEqual(0)
          expect(point.x).toBeLessThanOrEqual(1)
          expect(point.y).toBeGreaterThanOrEqual(0)
          expect(point.y).toBeLessThanOrEqual(1)
        }
      }
    },
  )

  it('stores separate registrations for the two exact image byte streams', () => {
    expect(registrationByTimepoint.preoperative.captureSha256).not.toBe(
      registrationByTimepoint.postoperative.captureSha256,
    )
    expect(registrationByTimepoint.preoperative).not.toBe(
      registrationByTimepoint.postoperative,
    )
  })
})
