import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { seedTask6State, TestRenderBoundary } from './task6TestSupport'

function renderPatient(path: string, state = seedTask6State().state) {
  return render(
    <TestRenderBoundary>
      <MemoryRouter initialEntries={[path]}>
        <App initialState={state} />
      </MemoryRouter>
    </TestRenderBoundary>,
  )
}

function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.addEventListener('load', () => resolve(String(reader.result)))
    reader.addEventListener('error', () => reject(reader.error))
    reader.readAsText(blob)
  })
}

afterEach(() => vi.restoreAllMocks())

describe('gated patient explanation and export', () => {
  it.each([
    ['/patient-report', 'No exact review ID was supplied'],
    ['/patient-report?review=', 'No exact review ID was supplied'],
    ['/patient-report?review=%20%20', 'No exact review ID was supplied'],
    [
      '/patient-report?review=%20review-task6-1%20',
      'The review ID must match exactly',
    ],
    [
      '/patient-report?review=review-task6-1&review=review-task6-1',
      'Duplicate review parameters were supplied',
    ],
    ['/patient-report?review=missing-review', 'missing-review'],
    ['/patient-report?review=toString', 'toString'],
    ['/patient-report?review=constructor', 'constructor'],
    ['/patient-report?review=__proto__', '__proto__'],
    ['/patient-report?review=prototype', 'prototype'],
    ['/patient-report?review=valueOf', 'valueOf'],
  ])('fails closed for malformed or unavailable binding %s', (path, reason) => {
    renderPatient(path)

    expect(
      screen.getByRole('heading', { name: 'Patient preview unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText(reason)).toBeVisible()
    expect(
      screen.queryByRole('button', { name: 'Download safe JSON manifest' }),
    ).not.toBeInTheDocument()
  })

  it.each([
    'case=SYN-HNC-CHEEK-TUMOUR',
    'extra=',
    '__proto__=polluted',
    'constructor=unsafe',
  ])('rejects every additional query key even with an otherwise eligible review: %s', (extra) => {
    const seed = seedTask6State({ reviewStatus: 'approved_for_research' })
    renderPatient(`/patient-report?review=${seed.reviewId}&${extra}`, seed.state)

    expect(
      screen.getByRole('heading', { name: 'Patient preview unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText('Unexpected query parameters were supplied')).toBeVisible()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Download safe JSON manifest' }),
    ).not.toBeInTheDocument()
  })

  it.each([
    ['awaiting review', { reviewStatus: 'awaiting_review' }],
    ['changes requested', { reviewStatus: 'changes_requested' }],
    ['revoked', { reviewStatus: 'revoked' }],
    [
      'stale result',
      { reviewStatus: 'approved_for_research', freshness: 'stale' },
    ],
    [
      'connected research result',
      { reviewStatus: 'approved_for_research', origin: 'model_prediction' },
    ],
  ] as const)(
    'blocks %s from patient preview and export',
    (_label, options) => {
      const seed = seedTask6State(options)
      renderPatient(`/patient-report?review=${seed.reviewId}`, seed.state)

      expect(
        screen.getByRole('heading', { name: 'Patient preview unavailable', level: 1 }),
      ).toBeVisible()
      expect(screen.getByText(/patient explanation eligibility is blocked/i)).toBeVisible()
      expect(screen.queryByRole('img')).not.toBeInTheDocument()
      expect(
        screen.queryByRole('button', { name: 'Download safe JSON manifest' }),
      ).not.toBeInTheDocument()
    },
  )

  it('opens an exact eligible review on AOI summary while keeping Overlay and export available', async () => {
    const seed = seedTask6State({ reviewStatus: 'approved_for_research' })
    const user = userEvent.setup()
    const { container } = renderPatient(
      `/patient-report?review=${seed.reviewId}`,
      seed.state,
    )

    expect(
      screen.getByRole('heading', {
        name: 'Simulated attention explanation',
        level: 1,
      }),
    ).toBeVisible()
    const preview = screen.getByRole('region', {
      name: 'Approved simulated patient explanation',
    })
    expect(
      within(preview).getByRole('region', { name: 'Simulated attention result' }),
    ).toHaveAttribute('data-layout', 'patient-compact')
    expect(
      within(preview).getByRole('heading', {
        name: 'Result',
      }),
    ).toBeVisible()
    const guidance = within(preview).getByText('How to discuss this result').closest(
      'details',
    )
    expect(guidance).not.toHaveAttribute('open')
    expect(container.querySelector('.task6-patient-hero')).not.toBeInTheDocument()
    expect(container.querySelector('.task6-patient-meaning-band')).not.toBeInTheDocument()
    expect(container.querySelector('.task6-patient-final')).not.toBeInTheDocument()
    expect(screen.queryByText(/where and when gaze landed/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/fixations?|fixation timing/i)).not.toBeInTheDocument()
    expect(within(preview).queryByText(/warmer areas/i)).not.toBeInTheDocument()
    expect(
      within(preview).getByText(
        'Begin with the AOI summary. Open the density field or overlay only when useful.',
      ),
    ).toBeVisible()
    expect(
      within(preview).getByRole('radio', { name: 'AOI summary' }),
    ).toBeChecked()
    expect(
      within(preview).getByRole('heading', { name: 'Clinical AOI summary' }),
    ).toBeVisible()
    expect(
      within(preview).getByRole('region', {
        name: 'Anatomical template shares',
      }),
    ).toBeVisible()
    expect(
      within(preview).getByRole('region', { name: 'Hemiface shares' }),
    ).toBeVisible()
    expect(
      within(preview).getByText(
        'Simulated attention-density interface structure · not observed or human gaze · not a patient prediction or result',
      ),
    ).toBeVisible()
    expect(within(preview).getAllByRole('img')).toHaveLength(1)
    expect(within(preview).getByRole('radio', { name: 'Overlay' })).toBeVisible()
    expect(within(preview).queryByRole('checkbox')).not.toBeInTheDocument()
    expect(preview).not.toHaveTextContent(/selected region|ROI/i)
    const technicalOptions = within(preview).getByText('Technical options').closest(
      'details',
    )
    expect(technicalOptions).not.toHaveAttribute('open')

    await user.click(within(preview).getByRole('radio', { name: 'Overlay' }))

    expect(within(preview).getAllByRole('img')).toHaveLength(1)
    expect(within(preview).getByText('SIMULATED — NOT HUMAN GAZE')).toBeVisible()
    expect(
      within(
        screen.getByRole('status', { name: 'Workspace environment' }),
      ).getByText(
        'Research prototype · sample data only · session data resets on refresh',
      ),
    ).toBeVisible()
    expect(
      screen.queryByText(/one independent, unpaired synthetic image/i),
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: 'Return to research review' }),
    ).toHaveAttribute(
      'href',
      `/research/reviews/${seed.reviewId}`,
    )
    expect(screen.queryByRole('button', { name: /print/i })).not.toBeInTheDocument()
    expect(document.body).not.toHaveTextContent(/demo_author|demo_reviewer/)
  })

  it('downloads an application/json whitelist manifest and revokes its Blob URL', async () => {
    const seed = seedTask6State({ reviewStatus: 'approved_for_research' })
    const createObjectUrl = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:task6-export')
    const revokeObjectUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {})
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {})
    const user = userEvent.setup()
    renderPatient(`/patient-report?review=${seed.reviewId}`, seed.state)

    await user.click(screen.getByText('Technical options'))
    await user.click(
      screen.getByRole('button', { name: 'Download safe JSON manifest' }),
    )

    expect(createObjectUrl).toHaveBeenCalledOnce()
    const blob = createObjectUrl.mock.calls[0][0]
    expect(blob).toBeInstanceOf(Blob)
    if (!(blob instanceof Blob)) throw new Error('Expected an export Blob.')
    expect(blob.type).toBe('application/json')
    expect(click).toHaveBeenCalledOnce()
    expect(revokeObjectUrl).toHaveBeenCalledWith('blob:task6-export')

    const serialized = await readBlob(blob)
    const manifest = JSON.parse(serialized) as {
      readonly roi: { readonly version: number }
    }
    expect(serialized).toContain('mock_simulation')
    expect(serialized).toContain(seed.assetSha256)
    expect(serialized).toContain(seed.modelVersion)
    expect(manifest.roi.version).toBe(seed.roiVersion)
    expect(serialized).toContain(seed.resultDigest)
    expect(serialized).toContain('approved_for_research')
    expect(serialized).toContain('clinicalUseEligible')
    expect(serialized).toContain('disclaimers')
    expect(serialized).not.toContain('Suitable for a synthetic research demonstration.')
    expect(serialized).not.toMatch(
      /demo_author|demo_reviewer|run-task6|attempt-task6|review-task6|patientId|name|dateOfBirth|medicalRecord/i,
    )
  })
})
