import { ArrowRight, Database, Search, ShieldCheck } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { approvedAssets } from '../data/approvedAssetManifest'

const demoCase = {
  id: 'D-001',
  caseId: 'demo-001',
  title: 'Nasal-region attention layout',
  description: 'Two approved synthetic faces prepared as an unpaired interface fixture.',
  state: 'Ready for demo',
}

export function WorklistPage() {
  const [query, setQuery] = useState('')
  const normalizedQuery = query.trim().toLowerCase()
  const cases = useMemo(
    () =>
      !normalizedQuery ||
      `${demoCase.id} ${demoCase.title} ${demoCase.description}`
        .toLowerCase()
        .includes(normalizedQuery)
        ? [demoCase]
        : [],
    [normalizedQuery],
  )

  return (
    <div className="worklist-page">
      <section className="worklist-hero">
        <div className="page-shell worklist-hero__inner">
          <div>
            <p className="eyebrow">Synthetic worklist</p>
            <h1>Synthetic case worklist</h1>
            <p className="lede">
              A deliberately small, hash-pinned library for testing the clinician workflow
              before any real research data or model output is allowed in the interface.
            </p>
          </div>
          <div className="worklist-boundary">
            <ShieldCheck aria-hidden="true" />
            <div>
              <strong>Closed demo boundary</strong>
              <span>Uploads and external folders are unavailable.</span>
            </div>
          </div>
        </div>
      </section>

      <section className="page-shell worklist-stats" aria-label="Worklist summary">
        <div><strong>1</strong><span>approved demo set</span></div>
        <div><strong>{approvedAssets.length}</strong><span>synthetic images</span></div>
        <div><strong>0</strong><span>patient records</span></div>
        <div><strong>0</strong><span>network sources</span></div>
      </section>

      <section className="page-shell worklist-content" aria-labelledby="approved-cases-title">
        <div className="worklist-toolbar">
          <div>
            <p className="eyebrow">Approved fixtures</p>
            <h2 id="approved-cases-title">Cases available now</h2>
          </div>
          <label className="search-field">
            <span className="sr-only">Search synthetic cases</span>
            <Search aria-hidden="true" />
            <input
              type="search"
              aria-label="Search synthetic cases"
              placeholder="Search case ID or description"
              value={query}
              onChange={(event) => setQuery(event.currentTarget.value)}
            />
          </label>
        </div>

        {cases.length ? (
          <div className="case-row">
            <div className="case-row__media" aria-hidden="true">
              <img src={approvedAssets[0].url} alt="" />
              <img src={approvedAssets[1].url} alt="" />
            </div>
            <div className="case-row__main">
              <div className="case-row__meta">
                <span>{demoCase.id}</span>
                <span className="state-chip">{demoCase.state}</span>
              </div>
              <h3>{demoCase.title}</h3>
              <p>{demoCase.description}</p>
              <div className="case-row__tags">
                <span>AI-generated · unpaired</span>
                <span>mock_simulation</span>
                <span>simulated_ui_only</span>
              </div>
            </div>
            <Link
              className="case-row__action"
              to={`/analysis?case=${demoCase.caseId}`}
              aria-label={`Open case ${demoCase.id}`}
            >
              <span>Open demo</span>
              <ArrowRight aria-hidden="true" />
            </Link>
          </div>
        ) : (
          <div className="empty-state" role="status">
            <Search aria-hidden="true" />
            <h3>No approved synthetic cases match “{query}”.</h3>
            <p>Clear the search to return to the closed, one-case demo allowlist.</p>
            <button type="button" onClick={() => setQuery('')}>Clear search</button>
          </div>
        )}

        <aside className="upload-boundary">
          <div>
            <Database aria-hidden="true" />
            <div>
              <h3>Upload is unavailable in this prototype.</h3>
              <p>
                Future study ingestion will require consent, access control, de-identification,
                retention policy, and an approved API—not a browser-only file picker.
              </p>
            </div>
          </div>
          <span aria-label="Unavailable">Not enabled</span>
        </aside>
      </section>
    </div>
  )
}
