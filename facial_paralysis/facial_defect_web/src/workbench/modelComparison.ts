import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'
import { getWorkbenchAsset } from './catalog'
import {
  deriveClinicalAoiPresentation,
  type ClinicalAoiPresentation,
  type ClinicalAoiSubsiteId,
} from './clinicalAoiPresentation'
import {
  createInferenceBinding,
  runMockEngine,
} from './mockEngine'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import type {
  ApprovedRoiAnnotation,
  InferenceConfiguration,
  MockInferenceOutput,
  WorkspaceState,
} from './types'
import { WorkbenchError } from './types'

export type ClinicalAoiComparisonKey =
  | 'central_triangle'
  | 'patient_left_hemiface'
  | 'patient_right_hemiface'
  | 'outside_template'
  | ClinicalAoiSubsiteId

export type ClinicalAoiComparisonRow = Readonly<{
  readonly key: ClinicalAoiComparisonKey
  readonly label: string
  readonly versionAShare: number
  readonly versionBShare: number
  readonly versionBMinusA: number
}>

export type ClinicalAoiComparisonGroup = Readonly<{
  readonly id:
    | 'subsite_partition'
    | 'hemiface_partition'
    | 'central_triangle_reference'
  readonly label:
    | 'Facial subsite partition'
    | 'Hemiface partition'
    | 'Overlapping reference'
  readonly relationship: 'partition_total_1' | 'overlapping_non_additive'
  readonly rows: readonly ClinicalAoiComparisonRow[]
}>

export type ClinicalAoiMethod = Readonly<{
  readonly schemaVersion: 'synthetic-point-weight-aoi/1'
  readonly template: 'fixed_anatomical_template'
  readonly weight: 'point_intensity'
  readonly assignment: 'point_center'
  readonly radiusContribution: 'ignored'
  readonly purpose: 'ui_rehearsal_only'
}>

export type MockModelComparison = {
  readonly caseId: WorkbenchAssetId
  readonly assetSha256: string
  readonly config: Readonly<InferenceConfiguration>
  readonly roi: Readonly<ApprovedRoiAnnotation>
  readonly left: MockInferenceOutput
  readonly right: MockInferenceOutput
  readonly clinicalAoiMethod: ClinicalAoiMethod
  readonly clinicalAoiGroups: readonly [
    ClinicalAoiComparisonGroup,
    ClinicalAoiComparisonGroup,
    ClinicalAoiComparisonGroup,
  ]
}

const CLINICAL_AOI_METHOD: ClinicalAoiMethod = Object.freeze({
  schemaVersion: 'synthetic-point-weight-aoi/1',
  template: 'fixed_anatomical_template',
  weight: 'point_intensity',
  assignment: 'point_center',
  radiusContribution: 'ignored',
  purpose: 'ui_rehearsal_only',
})

function queryFailure(message: string): never {
  throw new WorkbenchError({
    reason: 'INVALID_OPERATIONAL_ID',
    message,
    field: 'case',
  })
}

export function parseStrictModelComparisonQuery(search: string): WorkbenchAssetId {
  const entries = [...new URLSearchParams(search).entries()]
  if (entries.length !== 1 || entries[0][0] !== 'case' || !entries[0][1]) {
    queryFailure(
      'Simulation comparison requires exactly one canonical case query parameter.',
    )
  }
  const caseId = entries[0][1]
  const asset = getWorkbenchAsset(caseId)
  if (!asset) {
    throw new WorkbenchError({
      reason: 'UNKNOWN_CASE',
      message: `Unknown workbench case: ${caseId}.`,
      field: 'case',
    })
  }
  return asset.id
}

function freezeApprovedRoi(roi: ApprovedRoiAnnotation): ApprovedRoiAnnotation {
  return Object.freeze({
    ...roi,
    geometry: Object.freeze({ ...roi.geometry }),
  })
}

function roundedDelta(right: number, left: number): number {
  return Number((right - left).toFixed(6))
}

export function getExactApprovedComparisonRoi(
  state: WorkspaceState,
  caseId: string,
): ApprovedRoiAnnotation | undefined {
  const asset = getWorkbenchAsset(caseId)
  if (!asset) return undefined

  const roi = state.roisByCase[asset.id]
  if (!roi || !isVerifiedFullImageSourceBinding(asset, roi)) {
    return undefined
  }

  return roi as ApprovedRoiAnnotation
}

type SuccessfulAoiPresentation = Extract<
  ClinicalAoiPresentation,
  { readonly ok: true }
>

function requireAoiPresentation(
  output: MockInferenceOutput,
): SuccessfulAoiPresentation {
  const presentation = deriveClinicalAoiPresentation(
    output.heatmap,
    output.binding.roiGeometry,
  )
  if (!presentation.ok) {
    throw new WorkbenchError({
      reason: 'MALFORMED_RESPONSE',
      message: `Model comparison requires a valid spatial field (${presentation.reason}).`,
      field: 'heatmap',
    })
  }
  return presentation
}

function comparisonRow(
  key: ClinicalAoiComparisonKey,
  label: string,
  versionAShare: number,
  versionBShare: number,
): ClinicalAoiComparisonRow {
  return Object.freeze({
    key,
    label,
    versionAShare,
    versionBShare,
    versionBMinusA: roundedDelta(versionBShare, versionAShare),
  })
}

function comparisonGroup(
  id: ClinicalAoiComparisonGroup['id'],
  label: ClinicalAoiComparisonGroup['label'],
  relationship: ClinicalAoiComparisonGroup['relationship'],
  rows: readonly ClinicalAoiComparisonRow[],
): ClinicalAoiComparisonGroup {
  return Object.freeze({
    id,
    label,
    relationship,
    rows: Object.freeze(rows),
  })
}

function createClinicalAoiGroups(
  versionA: SuccessfulAoiPresentation,
  versionB: SuccessfulAoiPresentation,
): MockModelComparison['clinicalAoiGroups'] {
  const versionASubsites = new Map(
    versionA.subsites.map((subsite) => [subsite.id, subsite]),
  )
  const versionBSubsites = new Map(
    versionB.subsites.map((subsite) => [subsite.id, subsite]),
  )
  const subsiteRows = versionA.subsites.map((versionASubsite) => {
    const versionBSubsite = versionBSubsites.get(versionASubsite.id)
    if (!versionBSubsite) {
      throw new WorkbenchError({
        reason: 'MALFORMED_RESPONSE',
        message: 'Model comparison requires the same registered AOIs in both fields.',
        field: 'heatmap',
      })
    }
    return comparisonRow(
      versionASubsite.id,
      versionASubsite.label,
      versionASubsite.share,
      versionBSubsite.share,
    )
  })

  if (
    versionASubsites.size !== versionBSubsites.size ||
    versionA.orientation.viewerRight !== 'patient_left' ||
    versionB.orientation.viewerRight !== 'patient_left'
  ) {
    throw new WorkbenchError({
      reason: 'MALFORMED_RESPONSE',
      message: 'Model comparison requires matching patient-oriented AOI registration.',
      field: 'attentionSemantics.clinicalAoi',
    })
  }

  return Object.freeze([
    comparisonGroup(
      'subsite_partition',
      'Facial subsite partition',
      'partition_total_1',
      [
        ...subsiteRows,
        comparisonRow(
          'outside_template',
          'Outside fixed template',
          versionA.outsideTemplateShare,
          versionB.outsideTemplateShare,
        ),
      ],
    ),
    comparisonGroup(
      'hemiface_partition',
      'Hemiface partition',
      'partition_total_1',
      [
        comparisonRow(
          'patient_left_hemiface',
          'Patient-left hemiface',
          versionA.hemifaces.patientLeftShare,
          versionB.hemifaces.patientLeftShare,
        ),
        comparisonRow(
          'patient_right_hemiface',
          'Patient-right hemiface',
          versionA.hemifaces.patientRightShare,
          versionB.hemifaces.patientRightShare,
        ),
      ],
    ),
    comparisonGroup(
      'central_triangle_reference',
      'Overlapping reference',
      'overlapping_non_additive',
      [
        comparisonRow(
          'central_triangle',
          'Central facial triangle',
          versionA.centralTriangleShare,
          versionB.centralTriangleShare,
        ),
      ],
    ),
  ])
}

export function createMockModelComparison(input: {
  readonly workspaceState: WorkspaceState
  readonly caseId: string
  readonly config: InferenceConfiguration
}): MockModelComparison {
  const asset = getWorkbenchAsset(input.caseId)
  if (!asset) {
    throw new WorkbenchError({
      reason: 'UNKNOWN_CASE',
      message: `Unknown workbench case: ${input.caseId}.`,
      field: 'caseId',
    })
  }
  const currentRoi = getExactApprovedComparisonRoi(
    input.workspaceState,
    asset.id,
  )
  if (!currentRoi) {
    throw new WorkbenchError({
      reason: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
      message:
        'Same-case simulation comparison requires the current verified full-image source binding.',
      field: 'sourceBinding',
    })
  }

  const roi = freezeApprovedRoi(currentRoi)
  const config = Object.freeze({
    threshold: input.config.threshold,
    smoothing: input.config.smoothing,
  })
  const base = {
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelMode: 'mock_only' as const,
    config,
  }
  const left = runMockEngine(
    createInferenceBinding({
      ...base,
      clientRunId: `compare-${asset.id}-v03`,
      attemptToken: `compare-token-${asset.id}-v03`,
      modelVersion: 'mock-salience-v0.3',
    }),
  )
  const right = runMockEngine(
    createInferenceBinding({
      ...base,
      clientRunId: `compare-${asset.id}-v04`,
      attemptToken: `compare-token-${asset.id}-v04`,
      modelVersion: 'mock-salience-v0.4',
    }),
  )
  const clinicalAoiGroups = createClinicalAoiGroups(
    requireAoiPresentation(left),
    requireAoiPresentation(right),
  )

  return Object.freeze({
    caseId: asset.id,
    assetSha256: asset.sha256,
    config,
    roi,
    left,
    right,
    clinicalAoiMethod: CLINICAL_AOI_METHOD,
    clinicalAoiGroups,
  })
}
