import type { PatientAttentionPoint } from '../patientWorkflow/types'
import type {
  PresentationSubjectId,
  PresentationTimepoint,
} from '../data/presentationDemoAssets'

type PresentationAttentionSummary = Readonly<{
  provenance: 'hand_authored_simulation'
  cheekSignal: 'higher' | 'lower'
  copy: string
}>

const subjectAGeometry = [
  { x: 0.37, y: 0.39, radius: 0.04 },
  { x: 0.63, y: 0.39, radius: 0.04 },
  { x: 0.36, y: 0.46, radius: 0.055 },
  { x: 0.43, y: 0.46, radius: 0.04 },
  { x: 0.57, y: 0.46, radius: 0.04 },
  { x: 0.64, y: 0.46, radius: 0.055 },
  { x: 0.5, y: 0.54, radius: 0.038 },
  { x: 0.5, y: 0.61, radius: 0.045 },
  { x: 0.34, y: 0.6, radius: 0.05 },
  { x: 0.36, y: 0.66, radius: 0.042 },
  { x: 0.62, y: 0.59, radius: 0.058 },
  { x: 0.66, y: 0.62, radius: 0.062 },
  { x: 0.69, y: 0.65, radius: 0.054 },
  { x: 0.63, y: 0.67, radius: 0.046 },
  { x: 0.42, y: 0.71, radius: 0.052 },
  { x: 0.5, y: 0.73, radius: 0.06 },
  { x: 0.58, y: 0.71, radius: 0.052 },
  { x: 0.5, y: 0.8, radius: 0.04 },
] as const

const subjectBGeometry = subjectAGeometry.map((point, index) =>
  Object.freeze(
    index >= 10 && index <= 13
      ? { ...point, y: point.y - 0.045 }
      : { ...point },
  ),
)

const referenceIntensity = [
  0.32, 0.32, 0.7, 0.48, 0.48, 0.7, 0.3, 0.25, 0.24,
  0.2,
] as const
const preoperativeCheekIntensity = [0.9, 0.96, 0.86, 0.76] as const
const postoperativeCheekIntensity = [0.32, 0.34, 0.25, 0.2] as const
const mouthAndChinIntensity = [0.44, 0.42, 0.44, 0.18] as const

function field(
  geometry: readonly Readonly<{ x: number; y: number; radius: number }>[],
  cheekIntensity: readonly number[],
): readonly PatientAttentionPoint[] {
  const intensity = [
    ...referenceIntensity,
    ...cheekIntensity,
    ...mouthAndChinIntensity,
  ]

  return Object.freeze(
    geometry.map((point, index) =>
      Object.freeze({
        ...point,
        intensity: intensity[index]!,
      }),
    ),
  )
}

function attentionPair(
  geometry: readonly Readonly<{ x: number; y: number; radius: number }>[],
): Readonly<Record<PresentationTimepoint, readonly PatientAttentionPoint[]>> {
  return Object.freeze({
    preoperative: field(geometry, preoperativeCheekIntensity),
    postoperative: field(geometry, postoperativeCheekIntensity),
  })
}

export const presentationAttentionBySubject: Readonly<
  Record<
    PresentationSubjectId,
    Readonly<Record<PresentationTimepoint, readonly PatientAttentionPoint[]>>
  >
> = Object.freeze({
  'subject-a': attentionPair(subjectAGeometry),
  'subject-b': attentionPair(subjectBGeometry),
})

export const presentationAttentionSummary: Readonly<
  Record<PresentationTimepoint, PresentationAttentionSummary>
> = Object.freeze({
  preoperative: Object.freeze({
    provenance: 'hand_authored_simulation',
    cheekSignal: 'higher',
    copy:
      'The hand-authored demonstration places a stronger attention signal around the visible lesion.',
  }),
  postoperative: Object.freeze({
    provenance: 'hand_authored_simulation',
    cheekSignal: 'lower',
    copy:
      'The illustrative postoperative-like edit retains a small cheek signal, but it is deliberately lower than the lesion example.',
  }),
})
