import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { createHash } from 'node:crypto'

import { App } from './App'
import type { ResearchInferenceResult } from './model/inference'

async function uploadVideo(user: ReturnType<typeof userEvent.setup>) {
  const file = new File(['synthetic-video'], 'faces-session.webm', { type: 'video/webm' })
  await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)
  return file
}

async function uploadTimeline(user: ReturnType<typeof userEvent.setup>) {
  const digest = createHash('sha256').update('synthetic-video').digest('hex')
  const ids = [
    'neutral_repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
    'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
  ]
  const sidecar = new File([JSON.stringify({
    schema_version: 'faces-action-timeline/v1',
    script_version: 'faces-script/24-004956-v1',
    recording_sha256: digest,
    timing_source: 'capture_event_log',
    recording_duration_ms: 32_000,
    actions: ids.map((action, index) => ({
      action,
      status: 'completed',
      prompt_start_ms: index * 4_000,
      hold_start_ms: index * 4_000 + 500,
      hold_end_ms: index * 4_000 + 3_500,
      completion_ms: index * 4_000 + 3_750,
    })),
  })], 'faces-session.timeline.json', { type: 'application/json' })
  await user.upload(screen.getByLabelText('Choose FACES action timeline'), sidecar)
}

function acceptedResult(): ResearchInferenceResult {
  return {
    mode: 'research-inference',
    model: {
      modelId: 'broad_literature_shared_v9_blv9_009_ensemble',
      candidateId: 'BLV9-009',
      releaseManifestSha256: 'c4fdaf054f3076a2e31b0e1ae93d1e91a45212817eb39d1c4a53620a4007b18f',
      ensembleMembers: 3,
    },
    preprocessing: {
      version: 'faces-to-shared-v9/v1',
      faceLandmarkerSha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
      mirrorMethod: 'horizontal_flip_and_redetect',
      protocol: 'cue_aligned_action',
      timingSource: 'capture_event_log',
    },
    quality: { eligible: true, actionsUsed: 7, actions: [] },
    prediction: {
      probability: 0.73,
      memberProbabilities: [0.71, 0.74, 0.74],
      predictedClass: 1,
      threshold: 0.5,
      interpretation: 'research_score_only',
    },
    clinicalUseEligible: false,
  } as ResearchInferenceResult
}

describe('App', () => {
  it('states the research boundary and never claims unsupported HB or heatmap output', () => {
    render(<App demonstrationEnabled />)
    expect(screen.getByText('Research use only')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /capture the full facial movement story/i })).toBeInTheDocument()
    expect(screen.getByText(/FACES protocol · Source script v0.01/i)).toBeInTheDocument()
    expect(screen.queryByText(/IRB/i)).not.toBeInTheDocument()
    expect(screen.getByText(/Eye or mouth severity, House-Brackmann/i)).toBeInTheDocument()
    expect(screen.queryByText(/heatmap/i)).not.toBeInTheDocument()
  })

  it('runs demonstration only after a separate explicit user action', async () => {
    const user = userEvent.setup()
    render(<App demonstrationEnabled />)
    await uploadVideo(user)

    const demoButton = screen.getByRole('button', { name: 'Preview demonstration results' })
    await user.click(demoButton)

    expect(screen.getByText('DEMONSTRATION - NOT MODEL OUTPUT')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/no model processed this video/i)
    expect(screen.getByText('Demonstration probability layout')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Eye region' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Mouth region' })).toBeInTheDocument()
    expect(screen.queryByText('Binary model output')).not.toBeInTheDocument()
    expect(screen.queryByText('Ordinal region output')).not.toBeInTheDocument()
  })

  it('clears session media and results when starting a new session', async () => {
    const user = userEvent.setup()
    render(<App demonstrationEnabled />)
    await uploadVideo(user)
    await user.click(screen.getByRole('button', { name: 'Preview demonstration results' }))

    await user.click(screen.getByRole('button', { name: 'Start a new session' }))

    expect(screen.queryByText('faces-session.webm')).not.toBeInTheDocument()
    expect(screen.queryByText('DEMONSTRATION - NOT MODEL OUTPUT')).not.toBeInTheDocument()
    expect(screen.getByText('Choose a LifeLink Face recording')).toBeInTheDocument()
  })

  it('avoids smooth scrolling on reset when reduced motion is requested', async () => {
    const user = userEvent.setup()
    vi.spyOn(window, 'matchMedia').mockReturnValue({ matches: true } as MediaQueryList)
    render(<App demonstrationEnabled />)
    await uploadVideo(user)
    await user.click(screen.getByRole('button', { name: 'Preview demonstration results' }))
    await user.click(screen.getByRole('button', { name: 'Start a new session' }))

    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'auto' })
  })

  it('warns before identifiable video can be sent to a research endpoint', async () => {
    const user = userEvent.setup()
    render(<App apiEndpoint="https://research.example.test/infer" demonstrationEnabled />)
    await uploadVideo(user)
    expect(screen.getByText(/facial video is identifiable/i)).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /authorized research endpoint/i })).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeDisabled()
  })

  it('shows API failure without silently producing demonstration output', async () => {
    const user = userEvent.setup()
    const analyze = vi.fn().mockRejectedValue(new Error('Segmentation coverage incomplete'))
    render(
      <App
        apiEndpoint="https://research.example.test/infer"
        demonstrationEnabled
        analyze={analyze}
      />,
    )
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/segmentation coverage incomplete/i)
    expect(screen.queryByText('DEMONSTRATION - NOT MODEL OUTPUT')).not.toBeInTheDocument()
  })

  it('discards a stale response after the recording changes', async () => {
    const user = userEvent.setup()
    let resolveAnalysis: ((result: ResearchInferenceResult) => void) | undefined
    const analyze = vi.fn(
      () =>
        new Promise<ResearchInferenceResult>((resolve) => {
          resolveAnalysis = resolve
        }),
    )
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} />)
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    const replacement = new File(['replacement'], 'replacement.webm', { type: 'video/webm' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), replacement)
    resolveAnalysis?.(acceptedResult())

    await waitFor(() => expect(screen.getByText('replacement.webm')).toBeInTheDocument())
    expect(screen.queryByText('Accepted research inference')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /include step 8/i })).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeDisabled()
  })

  it('renders accepted research output with pinned provenance', async () => {
    const user = userEvent.setup()
    const result = acceptedResult()
    const analyze = vi.fn().mockResolvedValue(result)
    render(
      <App apiEndpoint="https://research.example.test/infer" analyze={analyze} />,
    )
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Accepted research inference')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/video hash, action timeline.*Shared V9/i)
    expect(screen.getByText('Uncalibrated research probability')).toBeInTheDocument()
    expect(screen.queryByText('Ordinal region output')).not.toBeInTheDocument()
    const provenance = screen.getByText('Validated response provenance').parentElement
    expect(provenance).not.toBeNull()
    expect(within(provenance!).getByText('BLV9-009')).toBeInTheDocument()
  })
})
