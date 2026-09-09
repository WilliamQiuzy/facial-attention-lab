import { readFileSync } from 'node:fs'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'
import { listWorkbenchAssets } from '../workbench/catalog'

function HistoryControls() {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <aside aria-label="Test history controls">
      <output aria-label="Current location">{`${location.pathname}${location.search}`}</output>
      <button type="button" onClick={() => navigate(-1)}>Browser back</button>
      <button type="button" onClick={() => navigate(1)}>Browser forward</button>
    </aside>
  )
}

function renderWorklist(path = '/cases') {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
      <HistoryControls />
    </MemoryRouter>,
  )
}

function renderCaseRoute(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

function visibleCaseCards() {
  return screen.queryAllByTestId('case-card')
}

function caseTitle(label: string) {
  return label.replace(/^Standalone synthetic case — /, '')
}

async function openFilters(
  user: ReturnType<typeof userEvent.setup>,
  currentCategory = 'All',
) {
  const disclosure = document.querySelector<HTMLDetailsElement>('details.case-filters')
  expect(disclosure).not.toBeNull()
  expect(disclosure).not.toHaveAttribute('open')
  await user.click(
    screen.getByText(`Category: ${currentCategory}`, {
      selector: 'summary',
    }),
  )
  expect(disclosure).toHaveAttribute('open')
}

describe('synthetic case worklist', () => {
  it('shows ten concise single-image cases with one workflow action each', () => {
    const { container } = renderWorklist()
    const assets = listWorkbenchAssets()
    const cards = visibleCaseCards()

    expect(cards).toHaveLength(10)
    expect(screen.getByRole('heading', { name: 'Cases', level: 1 })).toBeVisible()
    expect(screen.queryByText(/Closed synthetic boundary/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Case data boundary' })).not.toBeInTheDocument()
    expect(screen.queryByText(/upload is unavailable/i)).not.toBeInTheDocument()

    for (const [index, asset] of assets.entries()) {
      const card = cards[index]
      const [image] = within(card).getAllByRole('img')
      expect(within(card).getAllByRole('img')).toHaveLength(1)
      expect(image).toHaveAttribute('width', '1024')
      expect(image).toHaveAttribute('height', '1024')
      expect(image).toHaveAttribute('loading', 'lazy')
      expect(image).not.toHaveAttribute('fetchpriority')
      expect(image).toHaveAttribute('decoding', 'async')
      expect(within(card).queryByText(asset.id)).not.toBeInTheDocument()
      expect(within(card).queryByText('Synthetic')).not.toBeInTheDocument()
      expect(within(card).getAllByRole('link')).toHaveLength(1)
      expect(within(card).queryByText(/Generation metadata/i)).not.toBeInTheDocument()
      expect(within(card).queryByText(/SHA-256/i)).not.toBeInTheDocument()
      expect(within(card).queryByText(/^Run$/i)).not.toBeInTheDocument()
      expect(within(card).queryByText(/^Review$/i)).not.toBeInTheDocument()
      expect(within(card).queryByRole('status')).not.toBeInTheDocument()
      expect(within(card).queryByText(/ROI/i)).not.toBeInTheDocument()
      expect(
        within(card).getByRole('link', {
          name: `Run simulation for ${caseTitle(asset.label)}`,
        }),
      ).toHaveAttribute('href', `/analysis?case=${asset.id}`)
    }

    const resultCount = screen.getByRole('status', { name: 'Case result count' })
    expect(resultCount).toHaveAttribute('aria-live', 'polite')
    expect(resultCount).toHaveAttribute('aria-atomic', 'true')
    expect(resultCount).toHaveTextContent('10 of 10 cases')

    expect(container).not.toHaveTextContent(/before[- /]?after/i)
    expect(container).not.toHaveTextContent(/paired (?:case|image|comparison)/i)
    expect(container).not.toHaveTextContent(/treatment (?:outcome|improvement)/i)
  })

  it('keeps search visible and puts only Category in a closed native Filters disclosure', async () => {
    const user = userEvent.setup()
    renderWorklist()

    const search = screen.getByRole('searchbox', { name: 'Search synthetic cases' })
    expect(search).toHaveAttribute('name', 'q')
    expect(search).toHaveAttribute('autocomplete', 'off')
    expect(search).toHaveAttribute('placeholder', 'Try SYN-MOHS or trauma…')
    expect(search).toBeVisible()
    const category = screen.getByRole('combobox', { name: 'Category', hidden: true })
    expect(category).not.toBeVisible()
    expect(category).toHaveAttribute(
      'name',
      'category',
    )
    expect(screen.queryByRole('combobox', { name: /ROI/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/ROI status|All ROI states/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Clear filters' })).not.toBeInTheDocument()

    await user.type(search, 'SYN-VASCULAR-PWS')
    expect(visibleCaseCards()).toHaveLength(1)
    const vascularCase = listWorkbenchAssets().find(
      (asset) => asset.id === 'SYN-VASCULAR-PWS',
    )
    expect(vascularCase).toBeDefined()
    expect(
      screen.getByRole('heading', {
        name: caseTitle(vascularCase!.label),
        level: 2,
      }),
    ).toBeVisible()
    await waitFor(() => {
      expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
        '/cases?q=SYN-VASCULAR-PWS',
      )
    })

    await user.clear(search)
    await waitFor(() => expect(visibleCaseCards()).toHaveLength(10))
    await openFilters(user)
    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Category' }),
      'trauma',
    )
    expect(screen.getByText('Category: Trauma', { selector: 'summary' })).toBeVisible()
    expect(visibleCaseCards()).toHaveLength(2)
    expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
      '/cases?category=trauma',
    )

    await user.selectOptions(
      screen.getByRole('combobox', { name: 'Category' }),
      'mohs',
    )
    expect(visibleCaseCards()).toHaveLength(2)
    expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
      '/cases?category=mohs',
    )

    await user.click(screen.getByRole('button', { name: 'Clear filters' }))
    expect(search).toHaveValue('')
    expect(visibleCaseCards()).toHaveLength(10)
    expect(screen.getByText('10 of 10 cases')).toBeVisible()
    expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
      '/cases',
    )
    expect(screen.queryByRole('button', { name: 'Clear filters' })).not.toBeInTheDocument()
  })

  it('removes legacy ROI queries while preserving search and category history', async () => {
    const user = userEvent.setup()
    const deepLink = renderWorklist('/cases?q=SYN-HNC&category=hn_cancer&roi=approved')

    expect(screen.getByRole('searchbox', { name: 'Search synthetic cases' })).toHaveValue('SYN-HNC')
    expect(screen.getByRole('combobox', { name: 'Category', hidden: true })).not.toBeVisible()
    expect(screen.getByRole('combobox', { name: 'Category', hidden: true })).toHaveValue('hn_cancer')
    expect(screen.queryByRole('combobox', { name: /ROI/i })).not.toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
        '/cases?q=SYN-HNC&category=hn_cancer',
      )
    })

    await openFilters(user, 'Head & neck')
    await user.selectOptions(screen.getByRole('combobox', { name: 'Category' }), 'all')
    expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
      '/cases?q=SYN-HNC',
    )
    await user.click(screen.getByRole('button', { name: 'Browser back' }))
    expect(screen.getByRole('combobox', { name: 'Category' })).toHaveValue('hn_cancer')
    expect(screen.getByRole('searchbox', { name: 'Search synthetic cases' })).toHaveValue('SYN-HNC')
    expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
      '/cases?q=SYN-HNC&category=hn_cancer',
    )
    await user.click(screen.getByRole('button', { name: 'Browser forward' }))
    expect(screen.getByRole('combobox', { name: 'Category' })).toHaveValue('all')
    expect(screen.getByRole('searchbox', { name: 'Search synthetic cases' })).toHaveValue('SYN-HNC')

    deepLink.unmount()
    renderWorklist('/cases?q=SYN&category=unknown&roi=not-a-state')
    expect(screen.getByRole('combobox', { name: 'Category', hidden: true })).toHaveValue('all')
    await waitFor(() => {
      expect(screen.getByRole('status', { name: 'Current location' })).toHaveTextContent(
        '/cases?q=SYN',
      )
    })
  })

  it('provides a useful empty state without repeating the global data boundary', async () => {
    const user = userEvent.setup()
    const { container } = renderWorklist()

    await user.type(
      screen.getByRole('searchbox', { name: 'Search synthetic cases' }),
      'missing-case',
    )
    expect(visibleCaseCards()).toHaveLength(0)
    expect(screen.getByLabelText('No matching cases')).toBeVisible()
    expect(screen.getAllByRole('button', { name: 'Clear filters' })).toHaveLength(1)
    await user.click(screen.getByRole('button', { name: 'Clear filters' }))
    expect(visibleCaseCards()).toHaveLength(10)

    expect(screen.queryByText(/upload is unavailable in this workspace/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Case data boundary' })).not.toBeInTheDocument()
    expect(container.querySelector('input[type="file"]')).not.toBeInTheDocument()
  })

  it('preserves an uncropped square coordinate plane for every case preview', () => {
    const css = readFileSync('src/styles/workbench.css', 'utf8')

    expect(css).toMatch(/\.case-card__preview\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1/s)
    expect(css).toMatch(/\.case-card__preview img\s*\{[^}]*object-fit:\s*contain/s)
    expect(css).not.toMatch(/\.case-card__preview\s*\{[^}]*min-height:\s*(?:100%|\d+px)/s)
    expect(css).not.toMatch(/\.case-card__preview img\s*\{[^}]*(?:object-fit:\s*cover|min-height:\s*\d+px)/s)
    expect(css).not.toMatch(
      /\.case-card\s*\{[^}]*min-height:\s*248px/s,
    )
    expect(css).toMatch(
      /\.worklist-page \.workspace-page__header h1\s*\{[^}]*font-family:\s*var\(--research-font-sans\)/s,
    )
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)[\s\S]*\.case-card\s*\{[^}]*grid-template-columns:\s*108px minmax\(0,\s*1fr\)/s,
    )
    expect(css).toMatch(
      /@media \(max-width:\s*599px\)[\s\S]*\.case-card__preview\s*\{[^}]*width:\s*108px[^}]*height:\s*108px/s,
    )
  })

  it('opens only canonical source-binding IDs and fails closed for every unknown case', () => {
    const canonicalId = listWorkbenchAssets()[0].id
    const canonical = renderCaseRoute(`/cases/${canonicalId}/roi`)
    expect(
      screen.getByRole('heading', { name: 'Source image binding', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText(canonicalId)).toBeVisible()

    canonical.unmount()
    renderCaseRoute('/cases/not-a-canonical-case/roi')
    expect(
      screen.getByRole('heading', { name: 'Case unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText('not-a-canonical-case')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Back to cases' })).toHaveAttribute(
      'href',
      '/cases',
    )
    expect(screen.queryByText(canonicalId)).not.toBeInTheDocument()
  })
})
