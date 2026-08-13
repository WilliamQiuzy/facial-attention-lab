import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { App } from './App'
import type { ResearchInferenceResult } from './model/inference'

async function uploadVideo(user: ReturnType<typeof userEvent.setup>) {
  const file = new File(['synthetic-video'], 'faces-session.webm', { type: 'video/webm' })
  await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)
  return file
}

function acceptedResult(): ResearchInferenceResult {
  return {
    mode: 'research-inference',
    provenance: {
      modelFile: 'warmstart_v4_expanded.pt',
      modelSha256: '6310052121ed8a9a9e746716cb9c0d178eb252b438b6de7d33160eb555f6417b',
      preprocessingVersion: 'predict-pipeline/v1',
      segmentationVersion: 'faces-segmentation/v1',
    },
    segmentation: { durationMs: 42_000, actions: [] },
    scores: {
      palsyProbability: 0.73,
      eyes: { level: 1, expected: 1.2, pGt: [0.82, 0.38], label: 'Slight' },
      mouth: { level: 2, expected: 1.7, pGt: [0.91, 0.79], label: 'Strong' },
    },
  } as unknown as ResearchInferenceResult
}

describe('App', () => {
  it('states the research boundary and never claims unsupported HB or heatmap output', () => {
    render(<App demonstrationEnabled />)
    expect(screen.getByText('Research use only')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /capture the full facial movement story/i })).toBeInTheDocument()
    expect(screen.getByText(/FACES protocol · Source script v0.01/i)).toBeInTheDocument()
    expect(screen.queryByText(/IRB/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/House-Brackmann/i)).not.toBeInTheDocument()
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
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('radio', { name: /step 8 not applicable/i }))
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
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeDisabled()
    await user.click(screen.getByRole('radio', { name: /step 8 not applicable/i }))
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    const replacement = new File(['replacement'], 'replacement.webm', { type: 'video/webm' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), replacement)
    resolveAnalysis?.(acceptedResult())

    await waitFor(() => expect(screen.getByText('replacement.webm')).toBeInTheDocument())
    expect(screen.queryByText('Accepted research inference')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /step 8 not applicable/i })).not.toBeChecked()
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
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('radio', { name: /step 8 not applicable/i }))
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Accepted research inference')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/schema, segmentation, and checkpoint/i)
    expect(screen.getByText('Uncalibrated research probability')).toBeInTheDocument()
    expect(screen.getAllByText('Ordinal region output')).toHaveLength(2)
    const provenance = screen.getByText('Validated response provenance').parentElement
    expect(provenance).not.toBeNull()
    expect(within(provenance!).getByText('warmstart_v4_expanded.pt')).toBeInTheDocument()
  })
})
