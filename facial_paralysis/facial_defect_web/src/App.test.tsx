import { render, screen, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from './App'
import type { WorkbenchGateway } from './workbench/WorkbenchGateway'
import { listWorkbenchAssets } from './workbench/catalog'

const approvedCaseId = listWorkbenchAssets()[2].id

function renderApp(path = '/') {
  function LocationProbe() {
    const location = useLocation()
    return (
      <output data-testid="current-location">
        {location.pathname}
        {location.search}
      </output>
    )
  }

  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <LocationProbe />
    </MemoryRouter>,
  )
}

describe('workspace application shell', () => {
  it('opens Patients first and keeps only the clinician workflow in primary navigation', () => {
    renderApp()

    expect(screen.getByRole('heading', { name: 'Patients', level: 1 })).toBeVisible()
    expect(screen.getByRole('link', { name: /facial reconstruction imaging/i })).toHaveAttribute(
      'href',
      '/patients',
    )

    const primary = screen.getByRole('navigation', { name: 'Primary navigation' })
    const primaryLinks = within(primary).getAllByRole('link')
    expect(primaryLinks.map((link) => link.textContent)).toEqual(['Patients', 'Reviews', 'Help'])
    expect(primaryLinks.map((link) => link.getAttribute('href'))).toEqual([
      '/patients',
      '/reviews',
      '/about',
    ])
  })

  it('keeps every advanced route available from Help under Research tools', () => {
    renderApp('/about')

    expect(
      screen.getByRole('heading', { name: 'Help & research information', level: 1 }),
    ).toBeVisible()

    const researchTools = screen.getByRole('navigation', { name: 'Research tools' })
    expect(within(researchTools).getAllByRole('link').map((link) => link.textContent)).toEqual([
      'Synthetic cases',
      'Research reviews',
      'Runs',
      'Jobs',
      'Models',
      'Methods',
      'Integration',
    ])
    expect(within(researchTools).getAllByRole('link').map((link) => link.getAttribute('href'))).toEqual([
      '/cases',
      '/research/reviews',
      '/runs',
      '/jobs',
      '/models',
      '/methods',
      '/integration',
    ])
  })

  it('keeps the compact prototype boundary visible', () => {
    renderApp('/patients')

    const environments = screen.getAllByRole('status', {
      name: 'Workspace environment',
    })
    expect(environments).toHaveLength(1)
    const environment = environments[0]
    expect(
      within(environment).getByText(
        'Research prototype · synthetic/test records only · session data resets on refresh · clinical use blocked',
      ),
    ).toBeVisible()
    expect(
      screen.queryByRole('status', { name: 'Research use status' }),
    ).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: /skip to main content/i })).toHaveAttribute(
      'href',
      '#main-content',
    )
  })

  it('does not weaken the prototype boundary when research tools use a connected gateway', () => {
    const gateway = {
      mode: 'connected',
      runInference: async () => {
        throw new Error('Not invoked while rendering the shell.')
      },
    } satisfies WorkbenchGateway

    render(
      <MemoryRouter initialEntries={['/']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    const environment = screen.getByRole('status', {
      name: 'Workspace environment',
    })
    expect(
      within(environment).getByText(
        'Research prototype · synthetic/test records only · session data resets on refresh · clinical use blocked',
      ),
    ).toBeVisible()
  })

  it('keeps legacy research-review deep links compatible but redirects them out of the clinical namespace', () => {
    renderApp('/reviews/review-from-old-session')

    expect(
      screen.getByRole('heading', {
        name: 'Review unavailable in this session',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.getByText('review-from-old-session')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Back to reviews' })).toHaveAttribute(
      'href',
      '/research/reviews',
    )
    expect(screen.getByTestId('current-location')).toHaveTextContent(
      '/research/reviews/review-from-old-session',
    )
  })

  it('preserves the exact query while redirecting a legacy research-review creation link', () => {
    renderApp('/reviews/new?run=run-legacy&attempt=attempt-legacy')

    expect(screen.getByTestId('current-location')).toHaveTextContent(
      '/research/reviews/new?run=run-legacy&attempt=attempt-legacy',
    )
  })

  it.each([
    ['/', 'Patients'],
    ['/patients', 'Patients'],
    ['/patients/new', 'New patient'],
    ['/cases', 'Cases'],
    ['/runs', 'Recent simulations'],
    ['/jobs', 'Run several cases'],
    [`/models?case=${approvedCaseId}`, 'Compare simulation versions'],
    ['/reviews', 'Reviews'],
    ['/research/reviews', 'Review results'],
    ['/research/reviews/new', 'Review target unavailable'],
    ['/about', 'Help & research information'],
    ['/methods', 'Methods, provenance & safeguards'],
    ['/integration', 'Model & data readiness'],
    [`/analysis?case=${approvedCaseId}`, 'Simulated observer-attention density'],
    ['/patient-report', 'Patient preview unavailable'],
  ])('declares the internal route %s', async (path, heading) => {
    renderApp(path)

    expect(
      await screen.findByRole('heading', { name: heading, level: 1 }),
    ).toBeVisible()
    expect(
      screen.queryByRole('heading', { name: 'This route is not available' }),
    ).not.toBeInTheDocument()
  })

  it('resolves the legacy model route to Integration and fails closed elsewhere', () => {
    const legacy = renderApp('/model')
    expect(
      screen.getByRole('heading', { name: 'Model & data readiness', level: 1 }),
    ).toBeVisible()

    legacy.unmount()
    renderApp('/not-a-workspace-route')
    expect(
      screen.getByRole('heading', { name: 'This route is not available', level: 1 }),
    ).toBeVisible()
  })
})
