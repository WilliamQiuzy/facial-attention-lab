import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { ModelComparePage } from './ModelComparePage'
import { listWorkbenchAssets } from '../workbench/catalog'
import { MockWorkbenchGateway } from '../workbench/MockWorkbenchGateway'
import { WorkspaceProvider } from '../workbench/WorkspaceProvider'
import { createInitialWorkspaceState } from '../workbench/reducer'
import type { RoiAnnotation, WorkspaceState } from '../workbench/types'

const catalog = listWorkbenchAssets()
const approvedCases = catalog
const approvedCase = catalog[2]
const exactPartitionTotalCases = [
  'SYN-MOHS-SCC-CHEEK',
  'SYN-HNC-CHEEK-FREEFLAP',
  ...approvedCases
    .map(({ id }) => id)
    .filter(
      (id) =>
        id !== 'SYN-MOHS-SCC-CHEEK' &&
        id !== 'SYN-HNC-CHEEK-FREEFLAP',
    ),
]

function createExplicitDraftState(): WorkspaceState {
  const state = createInitialWorkspaceState()
  const defaultRoi = state.roisByCase[catalog[0].id]!
  const { reviewerId: _reviewerId, ...draftBase } = defaultRoi
  const draftRoi: RoiAnnotation = { ...draftBase, status: 'draft' }
  return {
    ...state,
    roisByCase: { ...state.roisByCase, [catalog[0].id]: draftRoi },
  }
}

function LocationProbe() {
  const location = useLocation()
  return <output aria-label="Current route">{location.pathname}{location.search}</output>
}

function renderPage(
  path: string,
  gateway = new MockWorkbenchGateway(),
  initialState?: WorkspaceState,
) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <WorkspaceProvider gateway={gateway} initialState={initialState}>
        <ModelComparePage />
        <LocationProbe />
      </WorkspaceProvider>
    </MemoryRouter>,
  )
}

function displayedComparisonPercentages(
  group: HTMLElement,
  versionColumn: 1 | 2,
): number[] {
  return within(group)
    .getAllByTestId('model-aoi-row')
    .map((row) => {
      const value = row.children.item(versionColumn)?.textContent ?? ''
      return Number(value.replace('%', ''))
    })
}

describe('model compare page', () => {
  it.each([
    '/models?case=',
    '/models?case=UNKNOWN-CASE',
    `/models?case=${approvedCase.id}&case=${approvedCase.id}`,
    `/models?case=${approvedCase.id}&case=${approvedCases[1].id}`,
    `/models?case=${approvedCase.id}&extra=true`,
  ])('fails closed without substituting a case for an invalid exact query: %s', (path) => {
    renderPage(path)

    expect(
      screen.getByRole('heading', { name: 'Simulation comparison unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.queryByTestId('model-comparison-panel')).not.toBeInTheDocument()
    expect(
      screen.queryByRole('combobox', { name: 'Available synthetic case' }),
    ).not.toBeInTheDocument()
  })

  it('leads the bare route with an approved-only case picker and no model registry cards', () => {
    renderPage('/models')

    expect(
      screen.getByRole('heading', { name: 'Compare simulation versions', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText('MOCK ONLY · PROMOTION BLOCKED')).toBeVisible()

    const selector = screen.getByRole('combobox', {
      name: 'Available synthetic case',
    })
    expect(
      within(selector).getAllByRole('option').map((option) =>
        (option as HTMLOptionElement).value),
    ).toEqual(['', ...approvedCases.map((asset) => asset.id)])
    expect(screen.getByRole('button', { name: 'Compare versions' })).toBeDisabled()

    expect(screen.queryByText('mock-salience-v0.3')).not.toBeInTheDocument()
    expect(screen.queryByText('mock-salience-v0.4')).not.toBeInTheDocument()
    expect(screen.queryByTestId('model-case-row')).not.toBeInTheDocument()
    expect(screen.queryByTestId('model-comparison-panel')).not.toBeInTheDocument()
    expect(
      screen.queryByText(/0 cases require source binding restoration/i),
    ).not.toBeInTheDocument()
  })

  it('navigates from the explicit case choice to one strict case query', async () => {
    const user = userEvent.setup()
    const selectedCase = approvedCases[2]
    renderPage('/models')

    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Available synthetic case' }),
      selectedCase.id,
    )
    await user.click(screen.getByRole('button', { name: 'Compare versions' }))

    expect(screen.getByLabelText('Current route')).toHaveTextContent(
      `/models?case=${selectedCase.id}`,
    )
    expect(
      screen.getByRole('heading', { name: 'Compare simulation versions', level: 1 }),
    ).toBeVisible()
    expect(
      within(screen.getByRole('region', { name: 'Selected comparison case' }))
        .getByText(selectedCase.id),
    ).toBeVisible()
  })

  it('shows center-assigned point-weight summaries as two partitions and one overlapping reference', async () => {
    const user = userEvent.setup()
    const { container } = renderPage(`/models?case=${approvedCase.id}`)

    expect(
      screen.getByRole('heading', { name: 'Compare simulation versions', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText('MOCK ONLY · UNVALIDATED · NOT A PATIENT RESULT')).toBeVisible()
    expect(screen.getByText('PROMOTION BLOCKED')).toBeVisible()

    const selectedCase = screen.getByRole('region', { name: 'Selected comparison case' })
    expect(within(selectedCase).getByText(approvedCase.label)).toBeVisible()
    expect(within(selectedCase).getByText(approvedCase.id)).toBeVisible()

    const panels = screen.getAllByTestId('model-comparison-panel')
    expect(
      screen.getByRole('region', { name: 'Same-case spatial simulations' }),
    ).toBeVisible()
    expect(panels).toHaveLength(2)
    expect(within(panels[0]).getByRole('heading', { name: 'Version A' })).toBeVisible()
    expect(within(panels[1]).getByRole('heading', { name: 'Version B' })).toBeVisible()
    expect(
      within(panels[0]).getByText('mock-salience-v0.3'),
    ).toBeVisible()
    expect(
      within(panels[1]).getByText('mock-salience-v0.4'),
    ).toBeVisible()
    expect(within(panels[0]).getByRole('img')).toHaveAttribute('src', approvedCase.url)
    expect(within(panels[1]).getByRole('img')).toHaveAttribute('src', approvedCase.url)
    expect(within(panels[0]).getByText(new RegExp(approvedCase.id))).toBeVisible()
    expect(within(panels[1]).getByText(new RegExp(approvedCase.id))).toBeVisible()
    expect(container.querySelectorAll('.roi-box')).toHaveLength(0)

    expect(screen.getAllByTestId('model-aoi-row')).toHaveLength(8)
    for (const label of [
      'Brow / forehead',
      'Orbital / eyes',
      'Nasal / midface',
      'Perioral / mouth',
      'Outside fixed template',
      'Patient-left hemiface',
      'Patient-right hemiface',
      'Central facial triangle',
    ]) {
      expect(screen.getByText(label)).toBeVisible()
    }

    const subsiteGroup = screen.getByRole('region', {
      name: 'Facial subsite partition',
    })
    const hemifaceGroup = screen.getByRole('region', {
      name: 'Hemiface partition',
    })
    const centralGroup = screen.getByRole('region', {
      name: 'Overlapping reference',
    })
    expect(within(subsiteGroup).getAllByTestId('model-aoi-row')).toHaveLength(5)
    expect(within(hemifaceGroup).getAllByTestId('model-aoi-row')).toHaveLength(2)
    expect(within(centralGroup).getAllByTestId('model-aoi-row')).toHaveLength(1)

    expect(
      screen.getByText(/fixed-template simulated point-weight shares/i),
    ).toBeVisible()
    expect(
      screen.getByText(
        /each point's intensity weight is assigned by its center.*radius is ignored.*not raster or density-kernel integrals.*UI rehearsal only/i,
      ),
    ).toBeVisible()
    expect(
      within(subsiteGroup).getByText(/five rows form one partition and sum to 100%/i),
    ).toBeVisible()
    expect(
      within(hemifaceGroup).getByText(/two hemifaces form one partition and sum to 100%/i),
    ).toBeVisible()
    expect(
      within(centralGroup).getByText(
        /overlaps both partitions.*not additive to either total/i,
      ),
    ).toBeVisible()
    expect(
      screen.queryByText(/completed spatial field shares|registered field shares/i),
    ).not.toBeInTheDocument()
    expect(screen.queryByText('ROI coverage')).not.toBeInTheDocument()
    expect(screen.queryByText('Peak intensity')).not.toBeInTheDocument()
    expect(screen.queryByText('Mean point intensity')).not.toBeInTheDocument()
    expect(screen.queryByText('Focus score')).not.toBeInTheDocument()
    expect(screen.queryByText('Advanced settings')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()

    const technical = screen.getByText('Technical details').closest('details')
    expect(technical).not.toBeNull()
    expect(technical).not.toHaveAttribute('open')

    const exactTechnical = within(technical!)
    expect(exactTechnical.getByText(approvedCase.sha256)).toBeInTheDocument()
    expect(exactTechnical.getByText(/roi-demo-03 · version 3/i)).toBeInTheDocument()
    expect(
      exactTechnical.getByText(/same synthetic asset · same full-image source binding/i),
    ).toBeInTheDocument()
    expect(exactTechnical.getByText(/synthetic_template_v1/i)).toBeInTheDocument()
    expect(exactTechnical.getByText('point_center')).toBeInTheDocument()
    expect(exactTechnical.getByText(/radius ignored/i)).toBeInTheDocument()
    expect(exactTechnical.getByText('mock-salience-v0.3')).toBeInTheDocument()
    expect(exactTechnical.getByText('mock-salience-v0.4')).toBeInTheDocument()
    expect(exactTechnical.getAllByText('mock_only')).toHaveLength(2)
    expect(
      exactTechnical.getByLabelText('Result digest mock-salience-v0.3'),
    ).toBeInTheDocument()
    expect(
      exactTechnical.getByLabelText('Result digest mock-salience-v0.4'),
    ).toBeInTheDocument()

    await user.click(screen.getByText('Technical details'))
    expect(technical).toHaveAttribute('open')
    expect(exactTechnical.getByText(approvedCase.sha256)).toBeVisible()
  })

  it.each(exactPartitionTotalCases)(
    'uses exact displayed AOI partition totals for %s',
    (caseId) => {
      renderPage(`/models?case=${caseId}`)

      for (const groupName of [
        'Facial subsite partition',
        'Hemiface partition',
      ]) {
        const group = screen.getByRole('region', { name: groupName })
        for (const versionColumn of [1, 2] as const) {
          const percentages = displayedComparisonPercentages(
            group,
            versionColumn,
          )
          expect(
            percentages.reduce(
              (sum, value) => sum + Math.round(value * 10),
              0,
            ),
          ).toBe(1000)
        }
      }

      const nonAdditive = screen.getByRole('region', {
        name: 'Overlapping reference',
      })
      expect(nonAdditive).toHaveTextContent(
        'not additive to either total',
      )
    },
  )

  it('uses one fixed comparison profile without clinician tuning controls', () => {
    renderPage(`/models?case=${approvedCase.id}`)

    expect(
      screen.getByLabelText('Result digest mock-salience-v0.3'),
    ).toBeInTheDocument()
    expect(
      screen.getByLabelText('Result digest mock-salience-v0.4'),
    ).toBeInTheDocument()
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    expect(screen.queryByText(/threshold/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/smoothing/i)).not.toBeInTheDocument()
  })

  it('blocks a canonical case whose source binding is not verified', () => {
    renderPage(
      `/models?case=${catalog[0].id}`,
      new MockWorkbenchGateway(),
      createExplicitDraftState(),
    )

    expect(
      screen.getByRole('heading', {
        name: 'Source image binding required',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.getByText(catalog[0].id)).toBeVisible()
    expect(screen.queryByTestId('model-comparison-panel')).not.toBeInTheDocument()
  })

  it.each([
    { caseId: catalog[0].id },
    { assetId: catalog[0].id },
    { id: '' },
    { version: 0 },
    { geometry: { x: -0.1, y: 0.2, width: 0.4, height: 0.4 } },
    { authorId: 'unexpected_author' },
    { reviewerId: undefined },
  ])('shows an explicit blocked state for a malformed source binding: %o', (patch) => {
    const state = createInitialWorkspaceState()
    const malformedState = {
      ...state,
      roisByCase: {
        ...state.roisByCase,
        [approvedCase.id]: {
          ...state.roisByCase[approvedCase.id]!,
          ...patch,
          status: 'approved',
        },
      },
    } as WorkspaceState

    renderPage(
      `/models?case=${approvedCase.id}`,
      new MockWorkbenchGateway(),
      malformedState,
    )

    expect(
      screen.getByRole('heading', {
        name: 'Source image binding required',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.queryByTestId('model-comparison-panel')).not.toBeInTheDocument()
  })

  it('does not invoke the gateway, fetch, or browser storage while comparing', () => {
    const gateway = new MockWorkbenchGateway()
    const gatewaySpy = vi.spyOn(gateway, 'runInference')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const storageReadSpy = vi.spyOn(Storage.prototype, 'getItem')
    const storageWriteSpy = vi.spyOn(Storage.prototype, 'setItem')

    try {
      renderPage(`/models?case=${approvedCase.id}`, gateway)

      expect(gatewaySpy).not.toHaveBeenCalled()
      expect(fetchSpy).not.toHaveBeenCalled()
      expect(storageReadSpy).not.toHaveBeenCalled()
      expect(storageWriteSpy).not.toHaveBeenCalled()
    } finally {
      gatewaySpy.mockRestore()
      fetchSpy.mockRestore()
      storageReadSpy.mockRestore()
      storageWriteSpy.mockRestore()
    }
  })

  it('can transition from a valid binding to the bare selector without changing hook order', async () => {
    const user = userEvent.setup()
    function NavigationHarness() {
      const navigate = useNavigate()
      return (
        <>
          <button type="button" onClick={() => navigate('/models')}>Remove case query</button>
          <ModelComparePage />
        </>
      )
    }

    render(
      <MemoryRouter initialEntries={[`/models?case=${approvedCase.id}`]}>
        <WorkspaceProvider gateway={new MockWorkbenchGateway()}>
          <NavigationHarness />
        </WorkspaceProvider>
      </MemoryRouter>,
    )
    expect(screen.getAllByTestId('model-comparison-panel')).toHaveLength(2)

    await user.click(screen.getByRole('button', { name: 'Remove case query' }))

    expect(
      screen.getByRole('heading', { name: 'Compare simulation versions', level: 1 }),
    ).toBeVisible()
    expect(
      screen.getByRole('combobox', { name: 'Available synthetic case' }),
    ).toHaveValue('')
  })
})
