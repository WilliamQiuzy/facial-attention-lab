import { ArrowLeft, ArrowRight, Check, Eye, EyeOff, ScanLine, ShieldAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { AttentionMap } from '../components/AttentionMap'
import { ComparisonSummary } from '../components/ComparisonSummary'
import { MetricCard } from '../components/MetricCard'
import { createAttentionService } from '../model/createAttentionService'
import type { AttentionAnalysis } from '../model/types'

export function AnalysisPage() {
  const [analysis, setAnalysis] = useState<AttentionAnalysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [opacity, setOpacity] = useState(68)
  const [showRegion, setShowRegion] = useState(false)

  useEffect(() => {
    let active = true
    try {
      const service = createAttentionService()
      if (service.mode !== 'demo') {
        setError(
          'Connected research mode is active. The simulated demo does not substitute for a connected analysis.',
        )
      } else {
        service
          .getDemoAnalysis()
          .then((result) => {
            if (active) setAnalysis(result)
          })
          .catch((caught: unknown) => {
            if (active) setError(caught instanceof Error ? caught.message : 'Demo unavailable.')
          })
      }
    } catch (caught: unknown) {
      if (active) setError(caught instanceof Error ? caught.message : 'Demo unavailable.')
    }
    return () => {
      active = false
    }
  }, [])

  if (error) {
    return (
      <section className="page-shell page-shell--narrow" role="alert">
        <p className="eyebrow">Fail-closed state</p>
        <h1>The research demo could not be loaded.</h1>
        <p className="lede">{error}</p>
        <Link className="button button--secondary" to="/cases">
          Return to synthetic cases
        </Link>
      </section>
    )
  }

  if (!analysis) {
    return (
      <div className="page-shell analysis-loading" role="status">
        Loading the synthetic demo…
      </div>
    )
  }

  const metrics = analysis.imageA.metrics
  const secondMetrics = analysis.imageB.metrics

  return (
    <div className="analysis-page">
      <div className="analysis-title page-shell">
        <Link className="back-link" to="/cases">
          <ArrowLeft aria-hidden="true" size={17} /> Synthetic worklist
        </Link>
        <div className="analysis-title__row">
          <div>
            <p className="eyebrow">Case D-001 · Interface fixture</p>
            <h1>Attention pattern demo</h1>
            <p className="lede">
              Explore how a future clinician view could compare visual-attention evidence
              while keeping source, pairing, and quality boundaries visible.
            </p>
          </div>
          <div className="provenance-card" aria-label="Analysis provenance">
            <p>Data origin</p>
            <strong>{analysis.origin}</strong>
            <span>{analysis.capabilityStatus}</span>
            <span>{analysis.imageRelationship}</span>
          </div>
        </div>
      </div>

      <section className="analysis-workbench" aria-labelledby="workbench-title">
        <div className="page-shell">
          <div className="workbench-heading">
            <div>
              <p className="eyebrow">Interactive workbench</p>
              <h2 id="workbench-title">Compare attention layers</h2>
            </div>
            <p className="unpaired-notice">
              <ShieldAlert aria-hidden="true" /> These are different generated identities,
              not a patient pair or a counterfactual result.
            </p>
          </div>

          <div className="analysis-controls" aria-label="Attention map display controls">
            <div className="segmented-control" role="group" aria-label="Image layer">
              <button
                type="button"
                aria-pressed={showHeatmap}
                onClick={() => setShowHeatmap(true)}
              >
                <Eye aria-hidden="true" size={17} /> Attention maps
              </button>
              <button
                type="button"
                aria-pressed={!showHeatmap}
                onClick={() => setShowHeatmap(false)}
              >
                <EyeOff aria-hidden="true" size={17} /> Original images
              </button>
            </div>
            <label className="range-control">
              <span>Heatmap opacity</span>
              <input
                type="range"
                min="10"
                max="100"
                value={opacity}
                disabled={!showHeatmap}
                onChange={(event) => setOpacity(Number(event.currentTarget.value))}
              />
              <output>{opacity}%</output>
            </label>
            <button
              className="roi-toggle"
              type="button"
              aria-label="Show scar region"
              aria-pressed={showRegion}
              onClick={() => setShowRegion((shown) => !shown)}
            >
              <ScanLine aria-hidden="true" size={18} />
              {showRegion ? 'Hide region' : 'Show region'}
            </button>
          </div>

          <div className="attention-grid">
            <AttentionMap
              result={analysis.imageA}
              showHeatmap={showHeatmap}
              opacity={opacity}
              showRegion={showRegion}
              watermark={analysis.watermark}
            />
            <AttentionMap
              result={analysis.imageB}
              showHeatmap={showHeatmap}
              opacity={opacity}
              showRegion={showRegion}
              watermark={analysis.watermark}
            />
          </div>

          <div className="legend" aria-label="Attention map legend">
            <span>Lower simulated density</span>
            <span className="legend__gradient" aria-hidden="true" />
            <span>Higher simulated density</span>
          </div>
        </div>
      </section>

      <section className="page-shell metric-section" aria-labelledby="metric-title">
        <div className="section-heading section-heading--compact">
          <div>
            <p className="eyebrow">Proposal-aligned measures</p>
            <h2 id="metric-title">What a future analysis can report</h2>
          </div>
          <p>
            These four values mirror the proposed eye-tracking measures. Every number below
            is deterministic mock data for interface testing.
          </p>
        </div>
        <div className="metric-grid">
          <MetricCard
            label="Scar-region gaze share"
            imageA={`${metrics.scarGazePercent}%`}
            imageB={`${secondMetrics.scarGazePercent}%`}
            note="Share of gaze time assigned to the defined region."
          />
          <MetricCard
            label="Time to first fixation"
            imageA={`${metrics.timeToFirstFixationMs} ms`}
            imageB={`${secondMetrics.timeToFirstFixationMs} ms`}
            note="Elapsed time before a first fixation enters the region."
          />
          <MetricCard
            label="Fixation duration"
            imageA={`${metrics.fixationDurationMs} ms`}
            imageB={`${secondMetrics.fixationDurationMs} ms`}
            note="Total fixation time accumulated inside the region."
          />
          <MetricCard
            label="Fixation count"
            imageA={metrics.fixationCount.toFixed(1)}
            imageB={secondMetrics.fixationCount.toFixed(1)}
            note="Mean number of region-entering fixations in the fixture."
          />
        </div>
      </section>

      <div className="page-shell">
        <ComparisonSummary comparison={analysis.comparison} />
      </div>

      <section className="quality-section">
        <div className="page-shell quality-section__inner">
          <div>
            <p className="eyebrow">Quality before interpretation</p>
            <h2>Three gates travel with every result.</h2>
          </div>
          <div className="quality-list">
            {analysis.quality.map((gate) => (
              <article key={gate.id}>
                <span className={`gate-icon gate-icon--${gate.status}`}>
                  {gate.status === 'pass' ? <Check aria-hidden="true" /> : '!'}
                </span>
                <div>
                  <div className="quality-list__title">
                    <h3>{gate.label}</h3>
                    <span>{gate.status.replace('_', ' ')}</span>
                  </div>
                  <p>{gate.detail}</p>
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="handoff-cta page-shell">
        <div>
          <p className="eyebrow">Next step</p>
          <h2>Move from analysis language to conversation language.</h2>
          <p>
            The patient view carries the same synthetic case forward while removing
            research jargon and preserving every safety boundary.
          </p>
        </div>
        <Link className="button button--primary" to="/patient-report?case=demo-001">
          Prepare patient explanation <ArrowRight aria-hidden="true" size={18} />
        </Link>
      </section>
    </div>
  )
}
