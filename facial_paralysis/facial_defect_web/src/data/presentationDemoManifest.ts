export type PresentationTimepoint =
  | 'preoperative'
  | 'postoperative'

export type PresentationDemoAssetMetadata = Readonly<{
  id: string
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
  sourceAssetId?: 'SYN-MOHS-SCC-CHEEK'
  derivedFromAssetId?: 'SYN-MOHS-SCC-CHEEK'
}>

export const PRESENTATION_BOUNDARY =
  'HAND-AUTHORED SIMULATION — NOT HUMAN GAZE — NOT A PREDICTED SURGICAL OUTCOME — CLINICAL USE BLOCKED'

const shared = {
  derivedSyntheticIdentityId: 'SYNTHETIC-IDENTITY-MOHS-CHEEK-001',
  width: 1024,
  height: 1024,
  relationship: 'paired_synthetic_presentation',
  allowedUse: 'simulated_presentation_demo',
  disclosure: PRESENTATION_BOUNDARY,
} as const

export const presentationDemoManifest: Readonly<
  Record<PresentationTimepoint, PresentationDemoAssetMetadata>
> = Object.freeze({
  preoperative: Object.freeze({
    ...shared,
    id: 'SYN-PRESENTATION-MOHS-CHEEK-PRE',
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
    timepoint: 'postoperative',
    label: 'Post-operative-like synthetic scar edit',
    sourcePath:
      'facial_defect_synthesis/output/synthetic/presentation/mohs_postop_small-cheek-scar_middle-aged-black-female.png',
    sha256:
      '72d0ee02fa6313b9ddb3c6b4ccf3c1f8c277c98b51a3b4453152948f21e7a58b',
    sourceClass: 'synthetic_ai_generated_edit',
    derivedFromAssetId: 'SYN-MOHS-SCC-CHEEK',
  }),
})
