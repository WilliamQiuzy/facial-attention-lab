import type {
  AttentionComparison,
  AttentionMetrics,
  AttentionOrigin,
  ImageRelationship,
} from './types'

export function compareAttention(
  imageA: Pick<AttentionMetrics, 'scarGazePercent' | 'timeToFirstFixationMs'>,
  imageB: Pick<AttentionMetrics, 'scarGazePercent' | 'timeToFirstFixationMs'>,
  origin: AttentionOrigin,
  relationship: ImageRelationship,
): AttentionComparison {
  const scarGazeChangePoints = Math.round(
    (imageB.scarGazePercent - imageA.scarGazePercent) * 10,
  ) / 10
  const relativeReductionPercent =
    imageA.scarGazePercent === 0
      ? 0
      : Math.round(
          ((imageA.scarGazePercent - imageB.scarGazePercent) /
            imageA.scarGazePercent) *
            100,
        )

  if (origin === 'mock_simulation' || relationship === 'unpaired_demo') {
    return {
      scarGazeChangePoints,
      relativeReductionPercent,
      interpretation:
        `In this simulated, unpaired interface example, Image B has ` +
        `${Math.abs(scarGazeChangePoints)} percentage points ` +
        `${scarGazeChangePoints <= 0 ? 'less' : 'more'} scar-region gaze allocation than Image A. ` +
        'This layout-only comparison is not evidence about a person or an outcome.',
    }
  }

  const sourceLabel = origin === 'observed_gaze' ? 'observed research' : 'model-predicted'
  return {
    scarGazeChangePoints,
    relativeReductionPercent,
    interpretation:
      `The ${sourceLabel} result differs by ${scarGazeChangePoints} percentage points ` +
      'between the paired research images. Interpretation requires protocol and quality review.',
  }
}
