import { readFileSync } from 'node:fs'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'
import { listWorkbenchAssets } from '../workbench/catalog'
import { createInitialWorkspaceState } from '../workbench/reducer'
import type { WorkbenchGateway } from '../workbench/WorkbenchGateway'
import type { WorkspaceState } from '../workbench/types'

function renderSourceBinding(
  caseId: string,
  initialState?: WorkspaceState,
  gateway?: WorkbenchGateway,
) {
  return render(
    <MemoryRouter initialEntries={[`/cases/${caseId}/roi`]}>
      <App initialState={initialState} gateway={gateway} />
    </MemoryRouter>,
  )
}

function createPartialState(caseId: WorkbenchAssetId): WorkspaceState {
  const initial = createInitialWorkspaceState()
  const current = initial.roisByCase[caseId]!
  return {
    ...initial,
    roisByCase: {
      ...initial.roisByCase,
      [caseId]: {
        ...current,
        status: 'approved' as const,
        geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
      },
    },
  }
}

describe('source image binding page', () => {
  it('keeps the source image square and collapses the recovery layout on narrow screens', () => {
    const css = readFileSync('src/styles/workbench.css', 'utf8')

    expect(css).toMatch(
      /\.source-binding-image figure\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1/s,
    )
    expect(css).toMatch(
      /\.source-binding-image img\s*\{[^}]*object-fit:\s*contain/s,
    )
    expect(css).toMatch(
      /@media \(max-width:\s*900px\)[\s\S]*?\.source-binding-layout\s*\{[^}]*grid-template-columns:\s*1fr/s,
    )
  })

  it('fails closed for an unknown exact case ID without substituting a fixture', () => {
    renderSourceBinding('UNKNOWN-CASE')

    expect(
      screen.getByRole('heading', { name: 'Case unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText('UNKNOWN-CASE')).toBeVisible()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Restore full-image binding' }),
    ).not.toBeInTheDocument()
  })

  it('shows one calm, verified full-image input identity without ROI editing controls', () => {
    const asset = listWorkbenchAssets()[0]
    renderSourceBinding(asset.id)

    expect(
      screen.getByRole('heading', { name: 'Source image binding', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText(asset.id)).toBeVisible()
    expect(screen.getByRole('img')).toHaveAttribute('src', asset.url)
    expect(screen.getByRole('img')).toHaveAttribute('width', '1024')
    expect(screen.getByRole('status', { name: 'Source binding status' })).toHaveTextContent(
      'Verified',
    )
    expect(
      screen.getByText(
        'Internal full-image source binding — not an anatomical AOI and not a surgical-site mask.',
      ),
    ).toBeVisible()
    expect(screen.getByRole('link', { name: 'Continue to Run' })).toHaveAttribute(
      'href',
      `/analysis?case=${asset.id}`,
    )

    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    expect(screen.queryByText('Normalized region')).not.toBeInTheDocument()
    expect(screen.queryByText('Annotation author')).not.toBeInTheDocument()
    expect(screen.queryByText('Independent reviewer')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Approve/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Supersede/i })).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Restore full-image binding' }),
    ).not.toBeInTheDocument()
  })

  it('restores an abnormal binding without inference and requires a separate Run action', async () => {
    const user = userEvent.setup()
    const asset = listWorkbenchAssets()[0]
    const runInference = vi.fn()
    const gateway: WorkbenchGateway = { mode: 'mock', runInference }
    renderSourceBinding(asset.id, createPartialState(asset.id), gateway)

    expect(screen.getByRole('status', { name: 'Source binding status' })).toHaveTextContent(
      'Needs restoration',
    )
    expect(
      screen.getByText(/incompatible current results become stale/i),
    ).toBeVisible()
    expect(screen.queryByRole('link', { name: 'Continue to Run' })).not.toBeInTheDocument()

    await user.click(
      screen.getByRole('button', { name: 'Restore full-image binding' }),
    )

    expect(screen.getByRole('status', { name: 'Source binding status' })).toHaveTextContent(
      'Verified',
    )
    expect(
      screen.queryByRole('button', { name: 'Restore full-image binding' }),
    ).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Continue to Run' })).toBeVisible()
    expect(runInference).not.toHaveBeenCalled()
  })
})
