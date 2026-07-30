import { Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { CaseCard, humanCategory } from '../components/CaseCard'
import { listWorkbenchAssets } from '../workbench/catalog'

function canonicalFilters(
  query: string,
  category: string,
): URLSearchParams {
  const next = new URLSearchParams()
  const cleanQuery = query.trim()
  if (cleanQuery) next.set('q', cleanQuery)
  if (category !== 'all') next.set('category', category)
  return next
}

export function WorklistPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const catalog = listWorkbenchAssets()
  const categories = [...new Set(catalog.map((asset) => asset.category))]
  const urlQuery = (searchParams.get('q') ?? '').trim()
  const requestedCategory = searchParams.get('category') ?? 'all'
  const urlCategory = requestedCategory === 'all' || categories.some((value) => value === requestedCategory)
    ? requestedCategory
    : 'all'
  const [query, setQuery] = useState(urlQuery)
  const [category, setCategory] = useState(urlCategory)
  const normalizedQuery = query.trim().toLowerCase()
  const filtersActive = category !== 'all'
  const categoryLabel =
    category === 'all' ? 'All' : humanCategory(category)

  useEffect(() => {
    setQuery(urlQuery)
    setCategory(urlCategory)
    const canonical = canonicalFilters(urlQuery, urlCategory)
    if (canonical.toString() !== searchParams.toString()) {
      setSearchParams(canonical, { replace: true })
    }
  }, [searchParams, setSearchParams, urlCategory, urlQuery])

  useEffect(() => {
    if (query.trim() === urlQuery) return
    const timeout = window.setTimeout(() => {
      setSearchParams(canonicalFilters(query, category), { replace: true })
    }, 120)
    return () => window.clearTimeout(timeout)
  }, [category, query, setSearchParams, urlQuery])

  const updateFilters = (updates: {
    readonly query?: string
    readonly category?: string
  }) => {
    if (updates.query !== undefined) {
      setQuery(updates.query)
      return
    }
    const nextCategory = updates.category ?? category
    setCategory(nextCategory)
    setSearchParams(canonicalFilters(query, nextCategory))
  }

  const cases = useMemo(
    () =>
      catalog.filter((asset) => {
        const matchesQuery =
          normalizedQuery.length === 0 ||
          [asset.id, asset.label, asset.category, humanCategory(asset.category)]
            .join(' ')
            .toLowerCase()
            .includes(normalizedQuery)
        const matchesCategory = category === 'all' || asset.category === category
        return matchesQuery && matchesCategory
      }),
    [catalog, category, normalizedQuery],
  )

  const clearFilters = () => {
    setQuery('')
    setCategory('all')
    setSearchParams(new URLSearchParams())
  }

  return (
    <div className="workspace-page worklist-page">
      <header className="workspace-page__header page-shell">
        <div>
          <h1>Cases</h1>
          <p>Find a synthetic case and continue its current step.</p>
        </div>
      </header>

      <section className="case-catalog page-shell" aria-labelledby="case-results-title">
        <div className="case-toolbar">
          <label className="case-search">
            <span>Search synthetic cases</span>
            <span className="case-search__control">
              <Search aria-hidden="true" />
              <input
                type="search"
                placeholder="Try SYN-MOHS or trauma…"
                name="q"
                autoComplete="off"
                value={query}
                onChange={(event) => updateFilters({ query: event.currentTarget.value })}
              />
            </span>
          </label>
          <details className="case-filters">
            <summary>Category: {categoryLabel}</summary>
            <div className="case-filters__controls">
              <label>
                <span>Category</span>
                <select
                  name="category"
                  value={category}
                  onChange={(event) => updateFilters({ category: event.currentTarget.value })}
                >
                  <option value="all">All categories</option>
                  {categories.map((value) => (
                    <option key={value} value={value}>{humanCategory(value)}</option>
                  ))}
                </select>
              </label>
              {filtersActive ? (
                <button className="workspace-button workspace-button--quiet" type="button" onClick={clearFilters}>
                  Clear filters
                </button>
              ) : null}
            </div>
          </details>
        </div>

        <div className="case-results-heading">
          <div>
            <h2 id="case-results-title">Case list</h2>
          </div>
          <output
            aria-label="Case result count"
            aria-live="polite"
            aria-atomic="true"
          >
            {cases.length} of {catalog.length} cases
          </output>
        </div>

        {cases.length > 0 ? (
          <div className="case-card-grid">
            {cases.map((asset) => (
              <CaseCard
                asset={asset}
                key={asset.id}
              />
            ))}
          </div>
        ) : (
          <div className="workspace-empty" aria-label="No matching cases">
            <Search aria-hidden="true" />
            <h3>No cases match these filters</h3>
            <p>Clear the filters to see all ten cases.</p>
            <button
              className="workspace-button workspace-button--quiet"
              type="button"
              onClick={clearFilters}
            >
              Clear filters
            </button>
          </div>
        )}
      </section>
    </div>
  )
}
