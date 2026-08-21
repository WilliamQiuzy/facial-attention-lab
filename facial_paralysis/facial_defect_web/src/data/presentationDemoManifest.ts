export type PresentationTimepoint = 'preoperative' | 'postoperative'

export type PresentationSubjectId = 'subject-a' | 'subject-b'

export type PresentationDemoAssetMetadata = Readonly<{
  id: string
  subjectId: PresentationSubjectId
  derivedSyntheticIdentityId: string
  timepoint: PresentationTimepoint
  label: string
  sourcePath: string
  sha256: string
  width: 1024
  height: 1024
  sourceClass:
    | 'synthetic_ai_generated'
    | 'synthetic_ai_generated_edit'
  relationship: 'paired_synthetic_presentation'
  allowedUse: 'simulated_presentation_demo'
  disclosure: string
  sourceAssetId?: 'SYN-MOHS-SCC-CHEEK' | 'SYN-NEVUS-CHEEK'
  derivedFromAssetId?: 'SYN-MOHS-SCC-CHEEK' | 'SYN-NEVUS-CHEEK'
}>

export type PresentationSubjectOption = Readonly<{
  id: PresentationSubjectId
  label: string
  description: string
}>

export const PRESENTATION_BOUNDARY =
  'SYNTHETIC DEMO — ILLUSTRATIVE ATTENTION, NOT MEASURED GAZE'

export const presentationSubjectIds = [
  'subject-a',
  'subject-b',
] as const satisfies readonly PresentationSubjectId[]

export const presentationSubjectOptions: readonly PresentationSubjectOption[] =
  Object.freeze([
    Object.freeze({
      id: 'subject-a',
      label: 'Subject A',
      description: 'Middle-aged sample patient',
    }),
    Object.freeze({
      id: 'subject-b',
      label: 'Subject B',
      description: 'Young adult sample patient',
    }),
  ])

const shared = {
  width: 1024,
  height: 1024,
  relationship: 'paired_synthetic_presentation',
  allowedUse: 'simulated_presentation_demo',
  disclosure: PRESENTATION_BOUNDARY,
} as const

export const presentationDemoManifest: Readonly<
  Record<
    PresentationSubjectId,
    Readonly<Record<PresentationTimepoint, PresentationDemoAssetMetadata>>
  >
> = Object.freeze({
  'subject-a': Object.freeze({
    preoperative: Object.freeze({
      ...shared,
      id: 'SYN-PRESENTATION-MOHS-CHEEK-PRE',
      subjectId: 'subject-a',
      derivedSyntheticIdentityId: 'SYNTHETIC-IDENTITY-MOHS-CHEEK-001',
      timepoint: 'preoperative',
      label: 'Pre-operative-like synthetic lesion',
      sourcePath:
        'facial_defect_synthesis/output/synthetic/mohs/mohs_preop_scc-cheek_middle-aged-black-female_a905e7.png',
      sha256:
        '1c43951ed068dc7b88a28e1a0e68d724f1f8b649067b81323ceb30fe2cc5eb30',
      sourceClass: 'synthetic_ai_generated',
      sourceAssetId: 'SYN-MOHS-SCC-CHEEK',
    }),
    postoperative: Object.freeze({
      ...shared,
      id: 'SYN-PRESENTATION-MOHS-CHEEK-POST',
      subjectId: 'subject-a',
      derivedSyntheticIdentityId: 'SYNTHETIC-IDENTITY-MOHS-CHEEK-001',
      timepoint: 'postoperative',
      label: 'Post-operative-like healing sutured incision edit',
      sourcePath:
        'facial_defect_synthesis/output/synthetic/presentation/mohs_postop_healing-cheek-incision_middle-aged-black-female.png',
      sha256:
        '4e62df4478f6852d788adf58a77056e208a72f5936b0960af8d6ed17af6d95e5',
      sourceClass: 'synthetic_ai_generated_edit',
      derivedFromAssetId: 'SYN-MOHS-SCC-CHEEK',
    }),
  }),
  'subject-b': Object.freeze({
    preoperative: Object.freeze({
      ...shared,
      id: 'SYN-PRESENTATION-NEVUS-CHEEK-PRE',
      subjectId: 'subject-b',
      derivedSyntheticIdentityId: 'SYNTHETIC-IDENTITY-NEVUS-CHEEK-002',
      timepoint: 'preoperative',
      label: 'Pre-operative-like synthetic cheek lesion',
      sourcePath:
        'facial_defect_synthesis/output/synthetic/nevus/nevus_preop_cheek-patch_young-adult-white-female_3fa4d7.png',
      sha256:
        '14909f75c0ba2ae4ab5e4ac1cf976f261d6d61eff09ae5804d865e2e6229d374',
      sourceClass: 'synthetic_ai_generated',
      sourceAssetId: 'SYN-NEVUS-CHEEK',
    }),
    postoperative: Object.freeze({
      ...shared,
      id: 'SYN-PRESENTATION-NEVUS-CHEEK-POST',
      subjectId: 'subject-b',
      derivedSyntheticIdentityId: 'SYNTHETIC-IDENTITY-NEVUS-CHEEK-002',
      timepoint: 'postoperative',
      label: 'Post-operative-like healing sutured incision edit',
      sourcePath:
        'facial_defect_synthesis/output/synthetic/presentation/nevus_postop_healing-cheek-incision_young-adult-white-female.png',
      sha256:
        'ecbb751e8c13b37465f899fd912a4dc4f713088388ca7a3ae94e925e55743b8b',
      sourceClass: 'synthetic_ai_generated_edit',
      derivedFromAssetId: 'SYN-NEVUS-CHEEK',
    }),
  }),
})
