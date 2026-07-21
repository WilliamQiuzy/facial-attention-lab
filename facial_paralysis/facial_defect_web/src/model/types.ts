import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'

export type AttentionOrigin = 'mock_simulation' | 'observed_gaze' | 'model_prediction'

export type ImageRelationship = 'unpaired_demo' | 'paired_research'

export type ApprovedAssetId = WorkbenchAssetId
export type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'

export type CapabilityStatus = 'simulated_ui_only' | 'research_unvalidated'

export type AttentionMetrics = {
  scarGazePercent: number
  timeToFirstFixationMs: number
  fixationDurationMs: number
  fixationCount: number
}

export type AttentionComparison = {
  scarGazeChangePoints: number
  relativeReductionPercent: number
  interpretation: string
}

export type HeatmapPoint = {
  x: number
  y: number
  radius: number
  intensity: number
}

export type RegionOfInterest = {
  label: string
  x: number
  y: number
  width: number
  height: number
  reviewStatus: 'demo_placeholder' | 'reviewed'
}

export type AttentionResult = {
  assetId: ApprovedAssetId
  label: string
  imageUrl: string
  disclosure: string
  metrics: AttentionMetrics
  heatmapPoints: HeatmapPoint[]
  regionOfInterest: RegionOfInterest
}

export type QualityGate = {
  id: string
  label: string
  status: 'pass' | 'blocked' | 'not_applicable'
  detail: string
}

export type AttentionAnalysis = {
  analysisId: string
  caseId: string
  origin: AttentionOrigin
  capabilityStatus: CapabilityStatus
  imageRelationship: ImageRelationship
  watermark: 'SIMULATED — NOT HUMAN GAZE'
  model: { name: string; version: string } | null
  imageA: AttentionResult
  imageB: AttentionResult
  comparison: AttentionComparison
  quality: QualityGate[]
  generatedAt: string
}

export interface DemoAttentionService {
  readonly mode: 'demo'
  getDemoAnalysis(): Promise<AttentionAnalysis>
}

export type ConnectedAnalysisEnvelope = {
  analysisId: string
  origin: Exclude<AttentionOrigin, 'mock_simulation'>
  capabilityStatus: 'research_unvalidated'
  assetId: ApprovedAssetId
  assetSha256: string
  roi: {
    reviewStatus: 'reviewed'
    version: string
  }
  quality: {
    status: 'eligible'
    eligibleSampleCount?: number
    protocolMinimum?: number
  }
  model: { name: string; version: string } | null
  [key: string]: unknown
}

export interface ConnectedAttentionService {
  readonly mode: 'connected'
  requestObservedAnalysis(payload: { assetId: ApprovedAssetId }): Promise<ConnectedAnalysisEnvelope>
  requestModelPrediction(payload: { assetId: ApprovedAssetId }): Promise<ConnectedAnalysisEnvelope>
}

export type AttentionService = DemoAttentionService | ConnectedAttentionService
