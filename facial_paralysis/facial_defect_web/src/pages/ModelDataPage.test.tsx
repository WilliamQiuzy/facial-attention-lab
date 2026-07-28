import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import type { WorkbenchGateway } from '../workbench/WorkbenchGateway'

describe('model and data readiness', () => {
  it('describes the active default mock gateway and one exact HTTP contract', () => {
    render(
      <MemoryRouter initialEntries={['/integration']}>
        <App />
      </MemoryRouter>,
    )

    const page = within(screen.getByRole('main'))
    const status = page.getByRole('status', { name: 'Current gateway mode' })
    expect(within(status).getByText(/local mock gateway active/i)).toBeVisible()
    expect(within(status).getByText(/default mode · no inference network requests/i)).toBeVisible()
    expect(page.getByText('WorkbenchGateway')).toBeVisible()
    expect(page.getByText('MockWorkbenchGateway')).toBeVisible()
    expect(page.getByText(/10 hash-pinned standalone synthetic cases/i)).toBeVisible()
    expect(page.getByText('/api/v1/workbench/inference')).toBeVisible()
    expect(page.getByText(/connected wire sends the request-contract version/i)).toBeVisible()
    expect(page.getByText(/response must echo that request identity/i)).toBeVisible()
    expect(page.getByText(/connected mode never falls back to mock/i)).toBeVisible()
    expect(page.getByText('mock_simulation')).toBeVisible()
    expect(page.getByText('model_prediction')).toBeVisible()
    expect(page.getByText('observed_gaze')).toBeVisible()
    const compatibility = page.getByRole('note', { name: 'Current model compatibility' })
    expect(
      within(compatibility).getByText(
        /functional-assessment research outputs are non-spatial severity or regional summaries/i,
      ),
    ).toBeVisible()
    expect(
      within(compatibility).getByText(
        /not connected to this web workbench/i,
      ),
    ).toBeVisible()
    expect(within(compatibility).getByText(/does not emit a spatial heatmap/i)).toBeVisible()
    expect(
      within(compatibility).getByText(
        /non-spatial severity or ordinal response cannot be converted into one/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /population-level predicted observer-attention spatial density/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /a future extended contract may add post-inference AOI summaries without changing the prediction/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(/surgical-site mask is not part of the current request/i),
    ).toBeVisible()
    expect(
      page.getByText(
        /no attention checkpoint or output is implemented/i,
      ),
    ).toBeVisible()
    expect(page.getByText('registration_geometry_unavailable_v1')).toBeVisible()
    expect(
      page.getByText(
        /does not carry landmarks or polygons, source dimensions, orientation or mirror metadata, or registration quality control/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /connected AOI reporting is unavailable and fails closed until the contract is extended/i,
      ),
    ).toBeVisible()
    expect(
      page.queryByText(/model-supplied face registration/i),
    ).not.toBeInTheDocument()
    expect(page.getByText(/capped at 4,096 display points/i)).toBeVisible()
    expect(page.queryByText('/api/v1/attention-analyses')).not.toBeInTheDocument()
    expect(page.queryByText('/api/v1/salience-predictions')).not.toBeInTheDocument()
    expect(page.queryByText(/two approved image assets/i)).not.toBeInTheDocument()
  })

  it('reports connected mode as explicitly enabled and remains zero-call until execution', () => {
    const runInference = vi.fn()
    const gateway = {
      mode: 'connected',
      runInference,
    } satisfies WorkbenchGateway
    render(
      <MemoryRouter initialEntries={['/integration']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    const page = within(screen.getByRole('main'))
    const status = page.getByRole('status', { name: 'Current gateway mode' })
    expect(within(status).getByText(/research HTTP seam enabled/i)).toBeVisible()
    expect(within(status).getByText(/explicit opt-in/i)).toBeVisible()
    expect(within(status).getByText(/network requests occur only when a run executes/i)).toBeVisible()
    expect(page.getByText('VITE_ENABLE_CONNECTED_MODE=true')).toBeVisible()
    expect(page.getByText('VITE_ATTENTION_API_URL')).toBeVisible()
    expect(page.getByText(/connected mode never falls back to mock/i)).toBeVisible()
    expect(page.getByText('research_unvalidated')).toBeVisible()
    const compatibility = page.getByRole('note', { name: 'Current model compatibility' })
    expect(
      within(compatibility).getByText(
        /current functional-assessment research system is still not connected/i,
      ),
    ).toBeVisible()
    expect(runInference).not.toHaveBeenCalled()
  })
})
