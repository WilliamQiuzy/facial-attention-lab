import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { ResearchInferenceResult } from '../model/inference'
import { frameHasVisibleContent, ResultsView } from './ResultsView'

const ids = [
  'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
  'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
] as const
const v9Actions = ['BROW_RAISE', 'EYE_GENTLE', 'EYE_FORCEFUL', 'SMILE_GENTLE', 'LIP_PUCKER', 'SHOW_BOTTOM_TEETH', 'SMILE_FULL']
const metrics = [
  ['brow_height_asymmetry_iod', 'brow_height_change_from_rest_iod'],
  ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
  ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
  ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
  ['mouth_corner_horizontal_asymmetry_iod', 'mouth_width_change_from_rest_iod'],
  ['mouth_corner_vertical_asymmetry_iod', 'lower_lip_change_from_rest_iod', 'mouth_open_change_from_rest_iod'],
  ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
]

function result(includeOptional: boolean): ResearchInferenceResult {
  const count = includeOptional ? 7 : 6
  return {
    mode: 'research-inference',
    model: {
      modelId: 'broad_literature_shared_v9_blv9_009_ensemble',
      candidateId: 'BLV9-009',
      releaseManifestSha256: '81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64',
      ensembleMembers: 3,
    },
    preprocessing: {
      version: 'faces-to-shared-v9/v1',
      faceLandmarkerSha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
      mirrorMethod: 'horizontal_flip_and_redetect',
      protocol: 'cue_aligned_action',
      timingSource: 'capture_event_log',
    },
    quality: {
      eligible: true,
      actionsUsed: count,
      optionalActionsUnavailable: includeOptional ? [] : ['reanimated_smile'],
      actions: ids.slice(0, count).map((id, index) => ({
        id,
        v9Action: v9Actions[index],
        holdStartMs: index * 4_000 + 4_500,
        holdEndMs: index * 4_000 + 7_500,
        validSamples: 26 + index,
      })),
    },
    prediction: {
      probability: 0.48,
      memberProbabilities: [0.45, 0.48, 0.51],
      predictedClass: 0,
      threshold: 0.5,
      interpretation: 'class_1_research_score_only',
      endpointSemantics: 'meei_facial_palsy_vs_healthy_control_development_head',
      class0Label: 'meei_healthy_control',
      class1Label: 'meei_facial_palsy',
    },
    reportEvidence: {
      normalization: 'original_view_centered_eye_axis_aligned_interocular_scaled',
      interpretation: 'measured_movement_observation_not_causal_or_severity',
      contextFrameMethod: 'registered_hold_midpoint_not_model_selected',
      actions: ids.slice(0, count).map((id, index) => ({
        id,
        contextFrameMs: index * 4_000 + 6_000,
        observations: metrics[index].map((metric, metricIndex) => ({
          metric,
          value: 0.01 * (metricIndex + 1),
          unit: 'interocular_distance' as const,
        })),
      })),
    },
    clinicalUseEligible: false,
  }
}

describe('Research Movement Report', () => {
  it('samples the full frame instead of treating dark corners as a black frame', () => {
    const getImageData = vi.fn((x: number, y: number, width: number, height: number) => ({
      data: new Uint8ClampedArray(width * height * 4).fill(
        x > 30 && x < 70 && y > 30 && y < 70 ? 8 : 0,
      ),
    }))
    expect(frameHasVisibleContent({ getImageData } as unknown as CanvasRenderingContext2D, 100, 100)).toBe(true)
    expect(getImageData).toHaveBeenCalledTimes(5)

    getImageData.mockImplementation((_x: number, _y: number, width: number, height: number) => ({
      data: new Uint8ClampedArray(width * height * 4),
    }))
    expect(frameHasVisibleContent({ getImageData } as unknown as CanvasRenderingContext2D, 100, 100)).toBe(false)
  })

  it('explains a 48 score and presents a six-action evidence report without invented grades', () => {
    const { container } = render(
      <ResultsView
        result={result(false)}
        recording={new File(['video'], 'capture.webm', { type: 'video/webm' })}
        onBack={vi.fn()}
        onReset={vi.fn()}
      />,
    )
    expect(screen.getAllByText('48 / 100')).toHaveLength(1)
    expect(screen.getByText(/2 points below the fixed cutpoint of 50/i)).toBeInTheDocument()
    expect(screen.getByText(/higher values indicate more similarity to the MEEI facial-palsy examples/i)).toBeInTheDocument()
    expect(screen.getByText('Below MEEI research cutpoint')).toBeInTheDocument()
    expect(screen.queryByText('Healthy-control class')).not.toBeInTheDocument()
    expect(screen.queryByText('Facial-palsy class')).not.toBeInTheDocument()
    expect(screen.queryByText(/does not estimate the chance/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Research use only/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Validated response')).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /how the model formed the score/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/auditable movement observations/i)).not.toBeInTheDocument()
    expect(screen.getByText(/measurements are scaled to the same eye-to-eye reference width/i)).toBeInTheDocument()
    expect(screen.queryByText(/no clinical normal range or severity meaning/i)).not.toBeInTheDocument()
    expect(screen.getByText(/measured movement observation — not a cause/i)).toBeInTheDocument()
    expect(screen.getAllByText('Side-to-side difference').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Change from neutral').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/1\.0% of eye-to-eye width/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/smaller means the two sides were more alike in this recording/i).length).toBeGreaterThan(1)
    expect(screen.getByText('26 of 32 points (81%)')).toBeInTheDocument()
    expect(screen.getByText('Brow-height change from rest')).toBeInTheDocument()
    expect(screen.getByText('Lower-lip movement from rest')).toBeInTheDocument()
    expect(screen.getAllByText('Recorded context frame unavailable')).toHaveLength(6)
    expect(screen.getByRole('heading', { name: 'Recording coverage' })).toBeInTheDocument()
    expect(screen.getByText('Neutral baseline + all 6 active movements')).toBeInTheDocument()
    expect(screen.getByText(/all 7 recorded steps in this session were used/i)).toBeInTheDocument()
    expect(screen.getByText('26–31 usable of 32 checkpoints per movement')).toBeInTheDocument()
    expect(screen.getByText(/each active movement is checked at 32 evenly spaced time points/i)).toBeInTheDocument()
    expect(screen.getByText(/neutral recording provides the resting baseline/i)).toBeInTheDocument()
    expect(screen.getByText('Not part of this session')).toBeInTheDocument()
    expect(screen.queryByText('Active movements used')).not.toBeInTheDocument()
    expect(screen.queryByText('Sample support')).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Reanimated smile' })).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Clinical scale status' })).not.toBeInTheDocument()
    expect(screen.queryByText('Not assessed')).not.toBeInTheDocument()
    expect(screen.queryByText('Not collected')).not.toBeInTheDocument()
    expect(screen.queryByText('Validated response provenance')).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Interpretation limits' })).not.toBeInTheDocument()
    expect(screen.queryByText(/not calibrated on FACES recordings/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/House–Brackmann, Sunnybrook, eFACE, and FaCE/i)).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Clinical review note' })).toBeInTheDocument()
    expect(screen.getByText(/review the movement score together with the recorded action images and source video/i)).toBeInTheDocument()
    expect(screen.getByText(/MediaPipe 478-point facial landmarks/i)).toBeInTheDocument()
    expect(container.querySelector('.report-clinical-note svg')).toHaveAttribute('width', '28')
    expect(container.textContent).not.toMatch(/\bcaused\b|contributed|abnormal|affected side|model confidence|BLV9-009/i)
  })

  it('supports all seven actions and report navigation without rerunning inference', async () => {
    const user = userEvent.setup()
    const onBack = vi.fn()
    const onReset = vi.fn()
    render(
      <ResultsView
        result={result(true)}
        recording={new File(['video'], 'capture.webm', { type: 'video/webm' })}
        onBack={onBack}
        onReset={onReset}
      />,
    )
    expect(screen.getByRole('heading', { name: 'Reanimated smile' })).toBeInTheDocument()
    expect(screen.getByText('Neutral baseline + all 7 active movements')).toBeInTheDocument()
    expect(screen.getByText(/all 8 recorded steps in this session were used/i)).toBeInTheDocument()
    expect(screen.getByText('Included')).toBeInTheDocument()
    await user.click(screen.getByRole('link', { name: /back to session summary/i }))
    await user.click(screen.getByRole('button', { name: /start a new session/i }))
    expect(onBack).toHaveBeenCalledTimes(1)
    expect(onReset).toHaveBeenCalledTimes(1)
  })

  it('directly downloads a PDF with evidence images and aligns all three report actions', async () => {
    const user = userEvent.setup()
    const warning = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const print = vi.spyOn(window, 'print').mockImplementation(() => undefined)
    const objectUrls: Blob[] = []
    const filenames: string[] = []
    const createObjectURL = vi.spyOn(URL, 'createObjectURL').mockImplementation((blob) => {
      objectUrls.push(blob as Blob)
      return 'blob:research-report'
    })
    const revokeObjectURL = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
      filenames.push(this.download)
    })
    const { container } = render(
      <ResultsView
        result={result(false)}
        recording={new File(['video'], 'capture.webm', { type: 'video/webm' })}
        onBack={vi.fn()}
        onReset={vi.fn()}
      />,
    )

    expect(container.querySelectorAll('.report-action-control')).toHaveLength(3)
    expect(screen.getByText(/PDF includes the recorded evidence images/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Download recorded video' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Save PDF' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: 'Save PDF' }))
    await waitFor(() => expect(filenames).toContain('faces-research-movement-report.pdf'))
    expect(print).not.toHaveBeenCalled()
    const pdf = objectUrls.find((blob) => blob.type === 'application/pdf')
    expect(pdf).toBeDefined()
    expect(pdf?.size).toBeGreaterThan(1_000)
    expect(warning).not.toHaveBeenCalled()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:research-report')
    print.mockRestore()
    createObjectURL.mockRestore()
    revokeObjectURL.mockRestore()
    anchorClick.mockRestore()
    warning.mockRestore()
  })
})
