import facialParalysisPostoperativeUrl from '../assets/patient-samples/facial-paralysis_postoperative_east-asian-woman.png?url'
import facialParalysisPreoperativeUrl from '../assets/patient-samples/facial-paralysis_preoperative_east-asian-woman.png?url'
import followUpPostoperativeUrl from '../assets/patient-samples/follow-up_postoperative-faded-cheek-scar_black-man.png?url'
import followUpPreoperativeUrl from '../assets/patient-samples/follow-up_preoperative-flat-cheek-scar_black-man.png?url'
import traumaPostoperativeUrl from '../assets/patient-samples/trauma_postop_revised-cheek-laceration-scar_middle-aged-middle-eastern-male_7811b9.png?url'
import traumaPreoperativeUrl from '../assets/patient-samples/trauma_preoperative-cheek-laceration_middle-eastern-man.png?url'

export type PatientSamplePhotoTimepoint =
  | 'preoperative'
  | 'postoperative'

export type PatientSamplePhotoAssetId =
  | 'SAMPLE-FACIAL-PARALYSIS-PREOPERATIVE'
  | 'SAMPLE-FACIAL-PARALYSIS-POSTOPERATIVE'
  | 'SAMPLE-TRAUMA-CHEEK-PREOPERATIVE'
  | 'SAMPLE-TRAUMA-CHEEK-POSTOPERATIVE'
  | 'SAMPLE-FOLLOW-UP-SCAR-PREOPERATIVE'
  | 'SAMPLE-FOLLOW-UP-SCAR-POSTOPERATIVE'

export type PatientSamplePhotoPairId =
  | 'SAMPLE-FACIAL-PARALYSIS-PAIR-01'
  | 'SAMPLE-TRAUMA-CHEEK-PAIR-01'
  | 'SAMPLE-FOLLOW-UP-SCAR-PAIR-01'

export type PatientSampleSubjectProfileId =
  | 'SYNTHETIC-FACIAL-PARALYSIS-WOMAN-01'
  | 'SYNTHETIC-TRAUMA-MAN-01'
  | 'SYNTHETIC-FOLLOW-UP-MAN-01'

export type PatientSampleAttentionProfile = Readonly<{
  focus: Readonly<{ x: number; y: number }>
  viewerSide: 'viewer_left' | 'viewer_right'
  focusIntensity: number
  focusRadius: number
  visualTarget: 'facial_asymmetry' | 'trauma_scar' | 'follow_up_scar'
}>

export type PatientSamplePhotoAsset = Readonly<{
  id: PatientSamplePhotoAssetId
  pairId: PatientSamplePhotoPairId
  subjectProfileId: PatientSampleSubjectProfileId
  timepoint: PatientSamplePhotoTimepoint
  label: string
  sourcePath: string
  sha256: string
  url: string
  sourceClass: 'synthetic_ai_generated'
  relationship: 'same_subject_paired_demo'
  allowedUse: 'simulated_patient_workflow'
  disclosure: string
  attentionProfile: PatientSampleAttentionProfile
}>

type PatientSamplePhotoPair = Readonly<{
  pairId: PatientSamplePhotoPairId
  subjectProfileId: PatientSampleSubjectProfileId
  preoperative: PatientSamplePhotoAsset
  postoperative: PatientSamplePhotoAsset
}>

function attentionProfile(
  focus: Readonly<{ x: number; y: number }>,
  viewerSide: PatientSampleAttentionProfile['viewerSide'],
  focusIntensity: number,
  focusRadius: number,
  visualTarget: PatientSampleAttentionProfile['visualTarget'],
): PatientSampleAttentionProfile {
  return Object.freeze({
    focus,
    viewerSide,
    focusIntensity,
    focusRadius,
    visualTarget,
  })
}

function pairedAsset(
  asset: PatientSamplePhotoAsset,
): PatientSamplePhotoAsset {
  return Object.freeze(asset)
}

const facialParalysisFocus = Object.freeze({ x: 0.64, y: 0.66 })
const traumaScarFocus = Object.freeze({ x: 0.34, y: 0.53 })
const followUpScarFocus = Object.freeze({ x: 0.65, y: 0.55 })

const facialParalysisPair = Object.freeze({
  pairId: 'SAMPLE-FACIAL-PARALYSIS-PAIR-01',
  subjectProfileId: 'SYNTHETIC-FACIAL-PARALYSIS-WOMAN-01',
  preoperative: pairedAsset({
    id: 'SAMPLE-FACIAL-PARALYSIS-PREOPERATIVE',
    pairId: 'SAMPLE-FACIAL-PARALYSIS-PAIR-01',
    subjectProfileId: 'SYNTHETIC-FACIAL-PARALYSIS-WOMAN-01',
    timepoint: 'preoperative',
    label: 'Sample facial asymmetry — preoperative',
    sourcePath:
      'facial_defect_web/src/assets/patient-samples/facial-paralysis_preoperative_east-asian-woman.png',
    sha256:
      '2708751042e0d615201e6932bc9d1e6a83dbd96dbf6620e0eda76cfa06b921e0',
    url: facialParalysisPreoperativeUrl,
    sourceClass: 'synthetic_ai_generated',
    relationship: 'same_subject_paired_demo',
    allowedUse: 'simulated_patient_workflow',
    disclosure:
      'AI-generated synthetic facial-asymmetry example for interface demonstration only; not a patient image or measured outcome.',
    attentionProfile: attentionProfile(
      facialParalysisFocus,
      'viewer_right',
      0.96,
      0.105,
      'facial_asymmetry',
    ),
  }),
  postoperative: pairedAsset({
    id: 'SAMPLE-FACIAL-PARALYSIS-POSTOPERATIVE',
    pairId: 'SAMPLE-FACIAL-PARALYSIS-PAIR-01',
    subjectProfileId: 'SYNTHETIC-FACIAL-PARALYSIS-WOMAN-01',
    timepoint: 'postoperative',
    label: 'Sample facial asymmetry — postoperative',
    sourcePath:
      'facial_defect_web/src/assets/patient-samples/facial-paralysis_postoperative_east-asian-woman.png',
    sha256:
      '72406ce387619444f8fc2b7e08997e3ca81be877c0c53d00a2c4faced4be32c9',
    url: facialParalysisPostoperativeUrl,
    sourceClass: 'synthetic_ai_generated',
    relationship: 'same_subject_paired_demo',
    allowedUse: 'simulated_patient_workflow',
    disclosure:
      'AI-generated synthetic facial-asymmetry example for interface demonstration only; not a patient image or measured outcome.',
    attentionProfile: attentionProfile(
      facialParalysisFocus,
      'viewer_right',
      0.28,
      0.05,
      'facial_asymmetry',
    ),
  }),
} satisfies PatientSamplePhotoPair)

const traumaPair = Object.freeze({
  pairId: 'SAMPLE-TRAUMA-CHEEK-PAIR-01',
  subjectProfileId: 'SYNTHETIC-TRAUMA-MAN-01',
  preoperative: pairedAsset({
    id: 'SAMPLE-TRAUMA-CHEEK-PREOPERATIVE',
    pairId: 'SAMPLE-TRAUMA-CHEEK-PAIR-01',
    subjectProfileId: 'SYNTHETIC-TRAUMA-MAN-01',
    timepoint: 'preoperative',
    label: 'Sample trauma cheek scar — preoperative',
    sourcePath:
      'facial_defect_web/src/assets/patient-samples/trauma_preoperative-cheek-laceration_middle-eastern-man.png',
    sha256:
      'e977d5406f2065ff88cd22e1dfb45e406714c96f00e2b34a77b10a494ad9cd6d',
    url: traumaPreoperativeUrl,
    sourceClass: 'synthetic_ai_generated',
    relationship: 'same_subject_paired_demo',
    allowedUse: 'simulated_patient_workflow',
    disclosure:
      'AI-generated synthetic trauma-scar example for interface demonstration only; not a patient image or measured outcome.',
    attentionProfile: attentionProfile(
      traumaScarFocus,
      'viewer_left',
      0.98,
      0.115,
      'trauma_scar',
    ),
  }),
  postoperative: pairedAsset({
    id: 'SAMPLE-TRAUMA-CHEEK-POSTOPERATIVE',
    pairId: 'SAMPLE-TRAUMA-CHEEK-PAIR-01',
    subjectProfileId: 'SYNTHETIC-TRAUMA-MAN-01',
    timepoint: 'postoperative',
    label: 'Sample trauma cheek scar — postoperative',
    sourcePath:
      'facial_defect_web/src/assets/patient-samples/trauma_postop_revised-cheek-laceration-scar_middle-aged-middle-eastern-male_7811b9.png',
    sha256:
      '7811b922a58920d57a51b2f2376fbfcea7d3ccbfffd2aff73b16de2010de9ede',
    url: traumaPostoperativeUrl,
    sourceClass: 'synthetic_ai_generated',
    relationship: 'same_subject_paired_demo',
    allowedUse: 'simulated_patient_workflow',
    disclosure:
      'AI-generated synthetic trauma-scar example for interface demonstration only; not a patient image or measured outcome.',
    attentionProfile: attentionProfile(
      traumaScarFocus,
      'viewer_left',
      0.24,
      0.05,
      'trauma_scar',
    ),
  }),
} satisfies PatientSamplePhotoPair)

const followUpPair = Object.freeze({
  pairId: 'SAMPLE-FOLLOW-UP-SCAR-PAIR-01',
  subjectProfileId: 'SYNTHETIC-FOLLOW-UP-MAN-01',
  preoperative: pairedAsset({
    id: 'SAMPLE-FOLLOW-UP-SCAR-PREOPERATIVE',
    pairId: 'SAMPLE-FOLLOW-UP-SCAR-PAIR-01',
    subjectProfileId: 'SYNTHETIC-FOLLOW-UP-MAN-01',
    timepoint: 'preoperative',
    label: 'Sample flat cheek scar — earlier visit',
    sourcePath:
      'facial_defect_web/src/assets/patient-samples/follow-up_preoperative-flat-cheek-scar_black-man.png',
    sha256:
      '617168e249a048ece8e472aeeffa64890e62a0f2504138c3d0c372f8602f5a0f',
    url: followUpPreoperativeUrl,
    sourceClass: 'synthetic_ai_generated',
    relationship: 'same_subject_paired_demo',
    allowedUse: 'simulated_patient_workflow',
    disclosure:
      'AI-generated synthetic flat-scar follow-up example for interface demonstration only; not a patient image or measured outcome.',
    attentionProfile: attentionProfile(
      followUpScarFocus,
      'viewer_right',
      0.97,
      0.105,
      'follow_up_scar',
    ),
  }),
  postoperative: pairedAsset({
    id: 'SAMPLE-FOLLOW-UP-SCAR-POSTOPERATIVE',
    pairId: 'SAMPLE-FOLLOW-UP-SCAR-PAIR-01',
    subjectProfileId: 'SYNTHETIC-FOLLOW-UP-MAN-01',
    timepoint: 'postoperative',
    label: 'Sample flat cheek scar — later follow-up',
    sourcePath:
      'facial_defect_web/src/assets/patient-samples/follow-up_postoperative-faded-cheek-scar_black-man.png',
    sha256:
      'caa4f7669453c5d84252151e49d551b1a7abe9da7004d411b7fde283f1e2a421',
    url: followUpPostoperativeUrl,
    sourceClass: 'synthetic_ai_generated',
    relationship: 'same_subject_paired_demo',
    allowedUse: 'simulated_patient_workflow',
    disclosure:
      'AI-generated synthetic flat-scar follow-up example for interface demonstration only; not a patient image or measured outcome.',
    attentionProfile: attentionProfile(
      followUpScarFocus,
      'viewer_right',
      0.2,
      0.045,
      'follow_up_scar',
    ),
  }),
} satisfies PatientSamplePhotoPair)

export const patientSamplePhotoPairs = Object.freeze({
  'patient-demo-001': facialParalysisPair,
  'patient-demo-002': traumaPair,
  'patient-demo-003': followUpPair,
} as const)

/** Legacy trauma-pair export retained for existing workflow fixtures. */
export const patientSamplePhotoAssets = traumaPair

export const patientSamplePhotoAssetList = Object.freeze(
  Object.values(patientSamplePhotoPairs).flatMap((pair) => [
    pair.preoperative,
    pair.postoperative,
  ]),
)

export function findPatientSamplePhotoAsset(
  assetId: string,
): PatientSamplePhotoAsset | undefined {
  return patientSamplePhotoAssetList.find(
    (asset) => asset.id === assetId,
  )
}

export function findPatientSamplePhotoAssetBySha256(
  sha256: string,
): PatientSamplePhotoAsset | undefined {
  const normalized = sha256.trim().toLowerCase()
  return patientSamplePhotoAssetList.find(
    (asset) => asset.sha256 === normalized,
  )
}

export function patientSamplePhotoAssetForPatientTimepoint(
  patientId: string,
  timepoint: string,
): PatientSamplePhotoAsset | undefined {
  const pair =
    patientSamplePhotoPairs[
      patientId as keyof typeof patientSamplePhotoPairs
    ]
  if (!pair) return undefined
  if (timepoint === 'preoperative') return pair.preoperative
  if (timepoint === 'postoperative') return pair.postoperative
  return undefined
}

/** @deprecated Resolve by patient id with patientSamplePhotoAssetForPatientTimepoint. */
export function patientSamplePhotoAssetForTimepoint(
  timepoint: string,
): PatientSamplePhotoAsset | undefined {
  if (timepoint === 'preoperative') return traumaPair.preoperative
  if (timepoint === 'postoperative') return traumaPair.postoperative
  return undefined
}
