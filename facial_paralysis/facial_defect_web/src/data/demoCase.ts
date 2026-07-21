import { compareAttention } from '../model/metrics'
import type { AttentionAnalysis } from '../model/types'
import { workbenchCatalog } from '../workbench/catalog'

const [imageA, imageB] = workbenchCatalog

const imageAMetrics = {
  scarGazePercent: 38,
  timeToFirstFixationMs: 420,
  fixationDurationMs: 920,
  fixationCount: 3.4,
}

const imageBMetrics = {
  scarGazePercent: 21,
  timeToFirstFixationMs: 760,
  fixationDurationMs: 540,
  fixationCount: 2.1,
}

export const demoAnalysis: AttentionAnalysis = {
  analysisId: 'sim-ui-demo-001',
  caseId: 'demo-001',
  origin: 'mock_simulation',
  capabilityStatus: 'simulated_ui_only',
  imageRelationship: 'unpaired_demo',
  watermark: 'SIMULATED — NOT HUMAN GAZE',
  model: null,
  imageA: {
    assetId: imageA.id,
    label: 'Synthetic image A — unpaired',
    imageUrl: imageA.url,
    disclosure: imageA.disclosure,
    metrics: imageAMetrics,
    heatmapPoints: [
      { x: 45, y: 61, radius: 23, intensity: 1 },
      { x: 39, y: 48, radius: 18, intensity: 0.72 },
      { x: 59, y: 45, radius: 16, intensity: 0.5 },
      { x: 50, y: 76, radius: 20, intensity: 0.38 },
    ],
    regionOfInterest: {
      label: 'Illustrative scar-region box',
      x: 36,
      y: 52,
      width: 19,
      height: 21,
      reviewStatus: 'demo_placeholder',
    },
  },
  imageB: {
    assetId: imageB.id,
    label: 'Synthetic image B — unpaired',
    imageUrl: imageB.url,
    disclosure: imageB.disclosure,
    metrics: imageBMetrics,
    heatmapPoints: [
      { x: 46, y: 60, radius: 19, intensity: 0.7 },
      { x: 39, y: 46, radius: 17, intensity: 0.62 },
      { x: 60, y: 46, radius: 17, intensity: 0.56 },
      { x: 51, y: 75, radius: 18, intensity: 0.36 },
    ],
    regionOfInterest: {
      label: 'Illustrative scar-region box',
      x: 38,
      y: 52,
      width: 18,
      height: 20,
      reviewStatus: 'demo_placeholder',
    },
  },
  comparison: compareAttention(
    imageAMetrics,
    imageBMetrics,
    'mock_simulation',
    'unpaired_demo',
  ),
  quality: [
    {
      id: 'asset-source',
      label: 'Synthetic-only asset boundary',
      status: 'pass',
      detail: 'Both files are hash-pinned AI-generated images from the approved allowlist.',
    },
    {
      id: 'human-gaze',
      label: 'Human gaze evidence',
      status: 'not_applicable',
      detail: 'No participant gaze data is present in this interface fixture.',
    },
    {
      id: 'clinical-use',
      label: 'Clinical-use gate',
      status: 'blocked',
      detail: 'External validation, governance, and workflow approval are not complete.',
    },
  ],
  generatedAt: '2026-07-20T12:00:00-04:00',
}
