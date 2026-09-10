import { afterEach, describe, expect, it, vi } from 'vitest'
import { patientSamplePhotoPairs } from '../data/patientSamplePhotoPair'
import { createDemoPatientResult } from './demoPatientInference'
import type { PatientRunBinding } from './types'
import {
  createCaptureAssetId,
  createPatientId,
  createPatientVisitId,
  createSessionMediaHandle,
} from './validation'

function makeBinding(
  overrides: Partial<PatientRunBinding> = {},
): PatientRunBinding {
  return {
    patientId: createPatientId('patient-demo-001'),
    visitId: createPatientVisitId('visit-demo-001'),
    captureId: createCaptureAssetId('capture-demo-001'),
    captureVersion: 1,
    captureSha256: 'a'.repeat(64),
    mediaHandle: createSessionMediaHandle('demo_capture_001'),
    authorizationRevision: 1,
    captureProtocol: 'frontal_relaxed_non_mirrored_v1',
    ...overrides,
  }
}

const leftShiftedFaceRegistration = {
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
        { x: 0.26, y: 0.16 },
        { x: 0.08, y: 0.48 },
        { x: 0.28, y: 0.84 },
        { x: 0.48, y: 0.48 },
      ],
    },
  ],
} as const

const rightShiftedFaceRegistration = {
  ...leftShiftedFaceRegistration,
  paths: [
    {
      feature: 'face_oval',
      closed: true,
      points: [
        { x: 0.72, y: 0.16 },
        { x: 0.52, y: 0.48 },
        { x: 0.72, y: 0.84 },
        { x: 0.92, y: 0.48 },
      ],
    },
  ],
} as const

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('createDemoPatientResult', () => {
  it('returns a deterministic, deeply frozen workflow simulation', () => {
    const binding = makeBinding()

    const first = createDemoPatientResult(binding)
    const second = createDemoPatientResult(binding)

    expect(first).toEqual(second)
    expect(first).not.toBe(second)
    expect(first.origin).toBe('workflow_simulation')
    expect(first.points.length).toBeGreaterThan(0)
    expect(Object.isFrozen(first)).toBe(true)
    expect(Object.isFrozen(first.points)).toBe(true)
    expect(first.points.every((point) => Object.isFrozen(point))).toBe(
      true,
    )
  })

  it('returns only finite normalized spatial point fields', () => {
    const output = createDemoPatientResult(makeBinding())

    expect(Object.keys(output).sort()).toEqual(['origin', 'points'])
    for (const point of output.points) {
      expect(Object.keys(point).sort()).toEqual([
        'intensity',
        'radius',
        'x',
        'y',
      ])
      for (const value of [
        point.x,
        point.y,
        point.intensity,
        point.radius,
      ]) {
        expect(Number.isFinite(value)).toBe(true)
        expect(value).toBeGreaterThanOrEqual(0)
        expect(value).toBeLessThanOrEqual(1)
      }
    }
  })

  it('keeps the display simulation inside a conservative frontal-face template', () => {
    const output = createDemoPatientResult(makeBinding())

    expect(output.points).toHaveLength(24)
    expect(
      output.points.every(
        (point) =>
          point.x >= 0.26 &&
          point.x <= 0.74 &&
          point.y >= 0.18 &&
          point.y <= 0.84 &&
          point.radius >= 0.035 &&
          point.radius <= 0.085,
      ),
    ).toBe(true)

    const facialBands = [
      output.points.filter((point) => point.y <= 0.34),
      output.points.filter(
        (point) => point.y > 0.34 && point.y <= 0.48,
      ),
      output.points.filter(
        (point) => point.y > 0.48 && point.y <= 0.66,
      ),
      output.points.filter(
        (point) => point.y > 0.66 && point.y <= 0.84,
      ),
    ]
    expect(facialBands.every((band) => band.length > 0)).toBe(true)
  })

  it('keeps one clearly dominant illustrative cheek cluster instead of covering the whole face uniformly', () => {
    const output = createDemoPatientResult(makeBinding())
    const illustrativeCheek = output.points.filter(
      (point) =>
        point.x >= 0.58 &&
        point.y >= 0.5 &&
        point.y <= 0.64,
    )
    const otherTemplatePoints = output.points.filter(
      (point) => !illustrativeCheek.includes(point),
    )

    expect(illustrativeCheek.length).toBeGreaterThanOrEqual(3)
    expect(
      Math.min(...illustrativeCheek.map((point) => point.intensity)),
    ).toBeGreaterThan(
      Math.max(
        ...otherTemplatePoints.map((point) => point.intensity),
      ),
    )
  })

  it('binds every sample heat field to that photograph\'s declared visual target', () => {
    for (const pair of Object.values(patientSamplePhotoPairs)) {
      for (const asset of [pair.preoperative, pair.postoperative]) {
        const output = createDemoPatientResult(
          makeBinding({ captureSha256: asset.sha256 }),
        )
        const focusPoint = output.points.find(
          (point) =>
            point.x === asset.attentionProfile.focus.x &&
            point.y === asset.attentionProfile.focus.y,
        )

        expect(focusPoint).toMatchObject({
          intensity: asset.attentionProfile.focusIntensity,
          radius: asset.attentionProfile.focusRadius,
        })
      }
    }
  })

  it('keeps paired focus locations aligned and makes postoperative focus clearly less prominent', () => {
    for (const pair of Object.values(patientSamplePhotoPairs)) {
      const preoperative = createDemoPatientResult(
        makeBinding({ captureSha256: pair.preoperative.sha256 }),
      )
      const postoperative = createDemoPatientResult(
        makeBinding({ captureSha256: pair.postoperative.sha256 }),
      )
      const focus = pair.preoperative.attentionProfile.focus
      const preFocus = preoperative.points.find(
        (point) => point.x === focus.x && point.y === focus.y,
      )
      const postFocus = postoperative.points.find(
        (point) => point.x === focus.x && point.y === focus.y,
      )

      expect(preFocus).toBeDefined()
      expect(postFocus).toBeDefined()
      expect(
        (preFocus?.intensity ?? 0) - (postFocus?.intensity ?? 0),
      ).toBeGreaterThanOrEqual(0.5)
      expect(postFocus?.radius).toBeLessThan(preFocus?.radius ?? 0)
    }
  })

  it('changes the spatial field when the exact capture SHA changes', () => {
    const first = createDemoPatientResult(makeBinding())
    const second = createDemoPatientResult(
      makeBinding({ captureSha256: 'b'.repeat(64) }),
    )

    expect(second.points).not.toEqual(first.points)
  })

  it('registers every illustrative attention point to the detected face in the uploaded photograph', () => {
    const left = createDemoPatientResult(
      makeBinding(),
      leftShiftedFaceRegistration,
    )
    const right = createDemoPatientResult(
      makeBinding(),
      rightShiftedFaceRegistration,
    )

    expect(
      left.points.every(
        (point) =>
          point.x >= 0.08 &&
          point.x <= 0.48 &&
          point.y >= 0.16 &&
          point.y <= 0.84,
      ),
    ).toBe(true)
    expect(
      right.points.every(
        (point) =>
          point.x >= 0.52 &&
          point.x <= 0.92 &&
          point.y >= 0.16 &&
          point.y <= 0.84,
      ),
    ).toBe(true)
    expect(right.points).not.toEqual(left.points)
  })

  it('derives points only from capture SHA when other binding fields change', () => {
    const first = createDemoPatientResult(makeBinding())
    const second = createDemoPatientResult(
      makeBinding({
        patientId: createPatientId('patient-demo-999'),
        visitId: createPatientVisitId('visit-demo-999'),
        captureId: createCaptureAssetId('capture-demo-999'),
        captureVersion: 9,
        mediaHandle: createSessionMediaHandle('demo_capture_999'),
        authorizationRevision: 7,
      }),
    )

    expect(second.points).toEqual(first.points)
  })

  it('normalizes capture SHA casing before deriving points', () => {
    const lowerCase = createDemoPatientResult(
      makeBinding({ captureSha256: 'ab'.repeat(32) }),
    )
    const upperCase = createDemoPatientResult(
      makeBinding({ captureSha256: 'AB'.repeat(32) }),
    )

    expect(upperCase.points).toEqual(lowerCase.points)
  })

  it('does not expose binding identifiers, PHI, or clinical metrics', () => {
    const binding = makeBinding()
    const serialized = JSON.stringify(createDemoPatientResult(binding))

    expect(serialized).not.toContain(binding.patientId)
    expect(serialized).not.toContain(binding.visitId)
    expect(serialized).not.toContain(binding.captureId)
    expect(serialized).not.toContain(binding.mediaHandle)
    expect(serialized).not.toContain(binding.captureSha256)
    expect(serialized).not.toMatch(
      /patientName|recordNumber|dateOfBirth|severity|diagnosis|treatment|clinicalMetric|score/i,
    )
  })

  it('does not use randomness, browser storage, IndexedDB, or network', () => {
    const random = vi.spyOn(Math, 'random')
    const storageGet = vi.spyOn(Storage.prototype, 'getItem')
    const storageSet = vi.spyOn(Storage.prototype, 'setItem')
    const storageRemove = vi.spyOn(Storage.prototype, 'removeItem')
    const storageClear = vi.spyOn(Storage.prototype, 'clear')
    const indexedDbOpen = vi.fn()
    const fetchRequest = vi.fn()
    vi.stubGlobal('indexedDB', { open: indexedDbOpen })
    vi.stubGlobal('fetch', fetchRequest)

    createDemoPatientResult(makeBinding())

    expect(random).not.toHaveBeenCalled()
    expect(storageGet).not.toHaveBeenCalled()
    expect(storageSet).not.toHaveBeenCalled()
    expect(storageRemove).not.toHaveBeenCalled()
    expect(storageClear).not.toHaveBeenCalled()
    expect(indexedDbOpen).not.toHaveBeenCalled()
    expect(fetchRequest).not.toHaveBeenCalled()
  })
})
