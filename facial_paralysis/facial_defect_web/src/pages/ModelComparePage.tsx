import {
  ArrowRight,
  GitCompareArrows,
  LockKeyhole,
  ShieldAlert,
} from 'lucide-react'
import { useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { AttentionMap } from '../components/AttentionMap'
import { FailClosedState } from '../components/FailClosedState'
import { getWorkbenchAsset, listWorkbenchAssets } from '../workbench/catalog'
import {
  createMockModelComparison,
  getExactApprovedComparisonRoi,
  parseStrictModelComparisonQuery,
} from '../workbench/modelComparison'
import { useWorkspace } from '../workbench/WorkspaceProvider'
import type {
  ApprovedRoiAnnotation,
  MockInferenceOutput,
} from '../workbench/types'
import '../styles/task5.css'

const COMPARISON_CONFIG = Object.freeze({
  threshold: 0.42,
  smoothing: 0.27,
})

function formatShare(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatPartitionShares(shares: readonly number[]): string[] {
  const total = shares.reduce((sum, share) => sum + share, 0)
  if (!(total > 0)) return shares.map(() => '0.0%')

  const quotas = shares.map((share) => (share / total) * 1000)
  const tenths = quotas.map(Math.floor)
  const remaining = 1000 - tenths.reduce((sum, value) => sum + value, 0)
  const priority = quotas
    .map((quota, index) => ({
      index,
      remainder: quota - tenths[index],
    }))
    .sort(
      (first, second) =>
        second.remainder - first.remainder || first.index - second.index,
    )

  for (let index = 0; index < remaining; index += 1) {
    tenths[priority[index].index] += 1
  }

  return tenths.map((value) => `${(value / 10).toFixed(1)}%`)
}

function formatDelta(value: number): string {
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)} pp`
}

export function ModelComparePage() {
  const { search } = useLocation()
  const { state } = useWorkspace()

  if (search === '') return <ModelRegistryLanding />

  let caseId: ReturnType<typeof parseStrictModelComparisonQuery>
  try {
    caseId = parseStrictModelComparisonQuery(search)
  } catch {
    return (
      <FailClosedState
        eyebrow="Strict comparison binding"
        title="Simulation comparison unavailable"
        requestedId={search || 'Missing ?case='}
        description="Provide exactly one current canonical case ID. Missing, unknown, duplicate, or multi-case inputs are never replaced with a fixture."
        backTo="/cases"
        backLabel="Choose a canonical case"
      />
    )
  }

  const roi = getExactApprovedComparisonRoi(state, caseId)
  if (!roi) {
    return (
      <FailClosedState
        eyebrow="Evaluation binding blocked"
        title="Source image binding required"
        requestedId={caseId}
        description="This exact canonical case does not have its verified full-image source binding, so simulation comparison is unavailable."
        backTo={`/cases/${caseId}/roi`}
        backLabel="Restore source binding"
      />
    )
  }

  const asset = getWorkbenchAsset(caseId)!
  const comparison = createMockModelComparison({
    workspaceState: state,
    caseId,
    config: COMPARISON_CONFIG,
  })

  return (
    <div className="workspace-page task5-page task5-model-page">
      <header className="workspace-page__header page-shell">
        <div>
          <p className="workspace-kicker">Models</p>
          <h1>Compare simulation versions</h1>
          <p>
            Compare two deterministic spatial-output rehearsals on the same synthetic
            case. No trained facial-defect attention model is present.
          </p>
          <section className="task5-selected-case" aria-label="Selected comparison case">
            <span>Selected case</span>
            <strong>{asset.label}</strong>
            <code>{asset.id}</code>
          </section>
        </div>
        <div className="task5-model-boundary" role="note">
          <ShieldAlert aria-hidden="true" />
          <div>
            <strong>MOCK ONLY · UNVALIDATED · NOT A PATIENT RESULT</strong>
            <span>PROMOTION BLOCKED</span>
          </div>
        </div>
      </header>

      <section className="task5-model-grid page-shell" aria-label="Same-case spatial simulations">
        <ModelPanel asset={asset} roi={comparison.roi} output={comparison.left} label="Version A" />
        <ModelPanel asset={asset} roi={comparison.roi} output={comparison.right} label="Version B" />
      </section>

      <section className="task5-metric-comparison page-shell" aria-labelledby="model-metrics-title">
        <div className="task5-section-heading task5-metric-comparison__heading">
          <div>
            <p className="workspace-kicker">Fixed-template AOI rehearsal</p>
            <h2 id="model-metrics-title">Compare simulated point-weight summaries</h2>
          </div>
          <p>
            Fixed-template simulated point-weight shares—not gaze duration, fixation
            count, severity, or treatment change.
          </p>
        </div>
        <p className="task5-aoi-note">
          Each point&apos;s intensity weight is assigned by its center. Radius is ignored,
          so these values are not raster or density-kernel integrals. UI rehearsal only.
          Patient left appears on the viewer&apos;s right.
        </p>
        <div className="task5-metric-table">
          <div className="task5-metric-table__labels" aria-hidden="true">
            <span>Fixed-template area</span><span>Version A</span><span>Version B</span><span>B − A</span>
          </div>
          {comparison.clinicalAoiGroups.map((group) => {
            const headingId = `model-aoi-group-${group.id}`
            const versionADisplay =
              group.relationship === 'partition_total_1'
                ? formatPartitionShares(
                    group.rows.map((row) => row.versionAShare),
                  )
                : group.rows.map((row) => formatShare(row.versionAShare))
            const versionBDisplay =
              group.relationship === 'partition_total_1'
                ? formatPartitionShares(
                    group.rows.map((row) => row.versionBShare),
                  )
                : group.rows.map((row) => formatShare(row.versionBShare))
            const note =
              group.id === 'subsite_partition'
                ? 'These five rows form one partition and sum to 100%.'
                : group.id === 'hemiface_partition'
                  ? 'These two hemifaces form one partition and sum to 100%.'
                  : 'This reference overlaps both partitions and is not additive to either total.'

            return (
              <section
                className="task5-aoi-group"
                aria-labelledby={headingId}
                key={group.id}
              >
                <div className="task5-aoi-group__heading">
                  <h3 id={headingId}>{group.label}</h3>
                  <p>{note}</p>
                </div>
                <div className="task5-aoi-group__rows">
                  {group.rows.map((row, rowIndex) => (
                    <div data-testid="model-aoi-row" key={row.key}>
                      <strong>{row.label}</strong>
                      <span>{versionADisplay[rowIndex]}</span>
                      <span>{versionBDisplay[rowIndex]}</span>
                      <code>{formatDelta(row.versionBMinusA)}</code>
                    </div>
                  ))}
                </div>
              </section>
            )
          })}
        </div>
      </section>

      <div className="page-shell task5-model-disclosures task5-model-disclosures--technical">
        <ModelTechnicalDetails comparison={comparison} />
        <Link className="workspace-button workspace-button--secondary" to="/models">
          Choose another case
        </Link>
      </div>
    </div>
  )
}

function ModelRegistryLanding() {
  const { state } = useWorkspace()
  const assets = listWorkbenchAssets()
  const navigate = useNavigate()
  const [selectedCaseId, setSelectedCaseId] = useState('')
  const approvedAssets = assets.filter((asset) => {
    return getExactApprovedComparisonRoi(state, asset.id) !== undefined
  })
  const needsReviewCount = assets.length - approvedAssets.length

  return (
    <div className="workspace-page task5-page task5-registry-page">
      <header className="workspace-page__header page-shell">
        <div>
          <p className="workspace-kicker">Models</p>
          <h1>Compare simulation versions</h1>
          <p>
            Choose one synthetic case to compare two deterministic spatial-output
            rehearsals. This is interface testing, not model evaluation.
          </p>
        </div>
        <div className="task5-model-boundary" role="note">
          <ShieldAlert aria-hidden="true" />
          <div>
            <strong>MOCK ONLY · PROMOTION BLOCKED</strong>
            <span>No human gaze or patient result</span>
          </div>
        </div>
      </header>

      <section className="page-shell task5-model-picker-wrap" aria-label="Choose comparison case">
        <form
          className="task5-model-picker"
          onSubmit={(event) => {
            event.preventDefault()
            if (selectedCaseId) {
              navigate(`/models?case=${encodeURIComponent(selectedCaseId)}`)
            }
          }}
        >
          <div>
            <h2>Choose a case</h2>
            <p>Only cases with a verified full-image source binding are available.</p>
          </div>
          <label>
            <span>Available synthetic case</span>
            <select
              name="availableSyntheticCase"
              aria-label="Available synthetic case"
              value={selectedCaseId}
              onChange={(event) => setSelectedCaseId(event.currentTarget.value)}
            >
              <option value="">Select a case</option>
              {approvedAssets.map((asset) => (
                <option value={asset.id} key={asset.id}>{asset.label} · {asset.id}</option>
              ))}
            </select>
          </label>
          <button
            className="workspace-button workspace-button--primary"
            type="submit"
            disabled={!selectedCaseId}
          >
            Compare versions <ArrowRight aria-hidden="true" />
          </button>
          {needsReviewCount > 0 ? (
            <p className="task5-model-picker__note">
              {needsReviewCount} {needsReviewCount === 1 ? 'case requires' : 'cases require'} source
              binding restoration and {needsReviewCount === 1 ? 'is' : 'are'} not listed.
            </p>
          ) : null}
        </form>
      </section>
    </div>
  )
}

function ModelTechnicalDetails({
  comparison,
}: {
  readonly comparison: ReturnType<typeof createMockModelComparison>
}) {
  const versions = [
    { label: 'Version A', output: comparison.left },
    { label: 'Version B', output: comparison.right },
  ] as const

  return (
    <details className="task5-model-technical">
      <summary>Technical details</summary>
      <div className="task5-model-technical__content">
        <div className="task5-model-technical__boundary">
          <LockKeyhole aria-hidden="true" />
          <p>
            <strong>Promotion blocked.</strong> These deterministic mock outputs contain no
            human gaze data and cannot support clinical or patient interpretation.
          </p>
        </div>
        <dl className="task5-model-technical__binding">
          <div><dt>Asset SHA-256</dt><dd><code>{comparison.assetSha256}</code></dd></div>
          <div><dt>Full-image source binding</dt><dd><code>{comparison.roi.id} · version {comparison.roi.version}</code></dd></div>
          <div><dt>Binding rule</dt><dd>Same synthetic asset · same full-image source binding</dd></div>
          <div><dt>AOI registration</dt><dd><code>synthetic_template_v1</code></dd></div>
          <div><dt>Summary assignment</dt><dd><code>{comparison.clinicalAoiMethod.assignment}</code></dd></div>
          <div><dt>Radius contribution</dt><dd>Radius ignored · no kernel integration</dd></div>
          <div><dt>AOI role</dt><dd>Post-inference mock summary · simulation unchanged</dd></div>
        </dl>
        <div className="task5-model-technical__versions">
          {versions.map(({ label, output }) => (
            <section key={label}>
              <h3>{label} technical record</h3>
              <dl>
                <div><dt>Simulation profile</dt><dd><code>{output.binding.modelVersion}</code></dd></div>
                <div><dt>Simulation mode</dt><dd><code>{output.binding.modelMode}</code></dd></div>
                <div><dt>Result digest</dt><dd><code aria-label={`Result digest ${output.binding.modelVersion}`}>{output.resultDigest}</code></dd></div>
                <div><dt>Configuration hash</dt><dd><code>{output.binding.configurationHash}</code></dd></div>
                <div><dt>Origin</dt><dd><code>{output.origin}</code></dd></div>
                <div><dt>Engine</dt><dd><code>{output.provenance.engine}</code></dd></div>
                <div><dt>Provenance</dt><dd>Deterministic; canonical synthetic asset; network false; storage false; human gaze false.</dd></div>
              </dl>
            </section>
          ))}
        </div>
      </div>
    </details>
  )
}

function ModelPanel({
  asset,
  roi,
  output,
  label,
}: {
  readonly asset: NonNullable<ReturnType<typeof getWorkbenchAsset>>
  readonly roi: ApprovedRoiAnnotation
  readonly output: MockInferenceOutput
  readonly label: string
}) {
  return (
    <article className="task5-model-panel" data-testid="model-comparison-panel">
      <div className="task5-model-panel__heading">
        <div>
          <span>Same-case simulated field</span>
          <h2>{label}</h2>
          <code>{output.binding.modelVersion}</code>
        </div>
        <GitCompareArrows aria-hidden="true" />
      </div>
      <AttentionMap
        asset={asset}
        output={output}
        roi={roi}
        showHeatmap
        opacity={72}
        showRegion={false}
      />
    </article>
  )
}
