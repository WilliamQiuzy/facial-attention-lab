import {
  FaceLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import wasmLoaderUrl from '@mediapipe/tasks-vision/vision_wasm_internal.js?url'
import wasmBinaryUrl from '@mediapipe/tasks-vision/vision_wasm_internal.wasm?url'
import faceLandmarkerModelUrl from '../assets/mediapipe/face_landmarker.task?url'
import type {
  PatientFaceFeature,
  PatientFacePath,
  PatientFaceRegistration,
} from './types'
import type {
  PatientFaceRegistrationInput,
  PatientFaceRegistrationRunner,
} from './PatientWorkflowProvider'

type LandmarkConnection = Readonly<{
  readonly start: number
  readonly end: number
}>

type FeatureTopology = Readonly<{
  readonly feature: PatientFaceFeature
  readonly connections: readonly LandmarkConnection[]
}>

const DETECTOR_VERSION =
  'tasks-vision-1.0.0-model-float16-1' as const

let faceLandmarkerPromise: Promise<FaceLandmarker> | undefined

function getFaceLandmarker(): Promise<FaceLandmarker> {
  faceLandmarkerPromise ??= FaceLandmarker.createFromOptions(
    {
      wasmLoaderPath: wasmLoaderUrl,
      wasmBinaryPath: wasmBinaryUrl,
    },
    {
      baseOptions: {
        modelAssetPath: faceLandmarkerModelUrl,
        delegate: 'CPU',
      },
      runningMode: 'IMAGE',
      numFaces: 2,
      minFaceDetectionConfidence: 0.65,
      minFacePresenceConfidence: 0.65,
      minTrackingConfidence: 0.65,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    },
  )
  return faceLandmarkerPromise
}

function featureTopology(): readonly FeatureTopology[] {
  return [
    {
      feature: 'face_oval',
      connections: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
    },
    {
      feature: 'left_eye',
      connections: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    },
    {
      feature: 'right_eye',
      connections: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    },
    {
      feature: 'left_eyebrow',
      connections: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
    },
    {
      feature: 'right_eyebrow',
      connections: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
    },
    {
      feature: 'lips',
      connections: FaceLandmarker.FACE_LANDMARKS_LIPS,
    },
  ]
}

function orderedConnectionChains(
  connections: readonly LandmarkConnection[],
): readonly (readonly number[])[] {
  const chains: number[][] = []
  let current: number[] = []

  for (const connection of connections) {
    const last = current[current.length - 1]
    if (current.length === 0 || last !== connection.start) {
      if (current.length > 1) chains.push(current)
      current = [connection.start, connection.end]
    } else {
      current.push(connection.end)
    }
  }
  if (current.length > 1) chains.push(current)
  return chains
}

function normalizedPoint(
  landmark: NormalizedLandmark | undefined,
) {
  if (
    !landmark ||
    !Number.isFinite(landmark.x) ||
    landmark.x < 0 ||
    landmark.x > 1 ||
    !Number.isFinite(landmark.y) ||
    landmark.y < 0 ||
    landmark.y > 1
  ) {
    throw new Error('INVALID_GEOMETRY')
  }
  return Object.freeze({ x: landmark.x, y: landmark.y })
}

function pathsFromLandmarks(
  landmarks: readonly NormalizedLandmark[],
): readonly PatientFacePath[] {
  const paths: PatientFacePath[] = []

  for (const topology of featureTopology()) {
    for (const chain of orderedConnectionChains(
      topology.connections,
    )) {
      const closed = chain[0] === chain[chain.length - 1]
      const indices = closed ? chain.slice(0, -1) : chain
      if (indices.length < 2) {
        throw new Error('INVALID_GEOMETRY')
      }
      paths.push(
        Object.freeze({
          feature: topology.feature,
          closed,
          points: Object.freeze(
            indices.map((index) =>
              normalizedPoint(landmarks[index]),
            ),
          ),
        }),
      )
    }
  }
  return Object.freeze(paths)
}

export const detectPatientFaceRegistration: PatientFaceRegistrationRunner =
  async (
    input: PatientFaceRegistrationInput,
  ): Promise<PatientFaceRegistration> => {
    if (typeof createImageBitmap !== 'function') {
      throw new Error('DETECTOR_UNAVAILABLE')
    }

    const bitmap = await createImageBitmap(input.media)
    try {
      if (
        bitmap.width !== input.sourceWidth ||
        bitmap.height !== input.sourceHeight
      ) {
        throw new Error('SOURCE_DIMENSION_MISMATCH')
      }
      const landmarker = await getFaceLandmarker()
      const detected = landmarker.detect(bitmap)
      if (detected.faceLandmarks.length === 0) {
        throw new Error('NO_FACE')
      }
      if (detected.faceLandmarks.length !== 1) {
        throw new Error('MULTIPLE_FACES')
      }

      return Object.freeze({
        schemaVersion: 'patient-face-registration/1',
        source: 'on_device_face_landmarks',
        coordinateSpace: 'decoded_image_normalized_v1',
        captureSha256: input.captureSha256,
        sourceWidth: input.sourceWidth,
        sourceHeight: input.sourceHeight,
        captureProtocol: input.captureProtocol,
        detectorId: 'mediapipe_face_landmarker',
        detectorVersion: DETECTOR_VERSION,
        faceCount: 1,
        paths: pathsFromLandmarks(detected.faceLandmarks[0]!),
      })
    } finally {
      bitmap.close()
    }
  }
