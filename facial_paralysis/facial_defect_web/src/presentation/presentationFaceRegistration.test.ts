import { describe, expect, it } from 'vitest'
import {
  presentationDemoAssets,
  presentationSubjectIds,
} from '../data/presentationDemoAssets'
import { registrationBySubject } from './presentationFaceRegistration'

const expectedFeatures = [
  'face_oval',
  'left_eye',
  'right_eye',
  'left_eyebrow',
  'right_eyebrow',
  'lips',
] as const

describe('presentation face registration snapshots', () => {
  it.each(
    presentationSubjectIds.flatMap((subjectId) =>
      (['preoperative', 'postoperative'] as const).map(
        (timepoint) => [subjectId, timepoint] as const,
      ),
    ),
  )(
    'binds %s %s contour to its exact image and MediaPipe provenance',
    (subjectId, timepoint) => {
      const registration = registrationBySubject[subjectId][timepoint]
      const asset = presentationDemoAssets[subjectId][timepoint]

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

  it('stores four separate registrations for four exact image byte streams', () => {
    const registrations = presentationSubjectIds.flatMap((subjectId) => [
      registrationBySubject[subjectId].preoperative,
      registrationBySubject[subjectId].postoperative,
    ])

    expect(new Set(registrations.map((item) => item.captureSha256))).toHaveLength(4)
    expect(new Set(registrations)).toHaveLength(4)
  })
})
