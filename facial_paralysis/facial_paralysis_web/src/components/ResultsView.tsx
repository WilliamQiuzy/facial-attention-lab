import { AlertTriangle, CheckCircle2, Eye, Info, ScanFace } from 'lucide-react'

import type { DemonstrationResult } from '../model/demonstration'
import type { RegionalSeverity, ResearchInferenceResult } from '../model/inference'

export type DisplayResult = ResearchInferenceResult | DemonstrationResult

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function RegionCard({ title, icon, region, demonstration }: { title: string; icon: React.ReactNode; region: RegionalSeverity; demonstration: boolean }) {
  const markerPosition = region.level * 50
  return (
    <article className="region-card">
      <div className="region-card-header">
        <span className="region-icon">{icon}</span>
        <div><span>{demonstration ? 'Interface preview value' : 'Ordinal region output'}</span><h3>{title}</h3></div>
        <strong className={`severity-chip severity-${region.level}`}>{region.label}</strong>
      </div>
      <div className="severity-scale" aria-label={`${title} ordinal level ${region.level} of 2`}>
        <span className="scale-track"><i style={{ left: `${markerPosition}%` }} /></span>
        <span className="scale-labels"><em>Normal</em><em>Slight</em><em>Strong</em></span>
      </div>
      <div className="threshold-list">
        <div><span>P(level &gt; Normal)</span><strong>{percent(region.pGt[0])}</strong><i><b style={{ width: percent(region.pGt[0]) }} /></i></div>
        <div><span>P(level &gt; Slight)</span><strong>{percent(region.pGt[1])}</strong><i><b style={{ width: percent(region.pGt[1]) }} /></i></div>
      </div>
      <p>{demonstration ? 'Deterministic layout values only; no model processed this video.' : 'Threshold probabilities are ordinal model outputs, not calibrated confidence.'}</p>
    </article>
  )
}

export function ResultsView({ result, onReset }: { result: DisplayResult; onReset: () => void }) {
  const isDemo = result.mode === 'demonstration'
  const scores = result.scores
  return (
    <section className="results-section" aria-labelledby="results-title">
      <div className={isDemo ? 'result-banner is-demo' : 'result-banner is-accepted'} role="status" aria-live="polite" aria-atomic="true">
        {isDemo ? <AlertTriangle aria-hidden="true" size={21} /> : <CheckCircle2 aria-hidden="true" size={21} />}
        <div>
          <strong>{isDemo ? result.provenanceLabel : 'Accepted research inference'}</strong>
          <span>{isDemo ? 'Interface preview values only. No model processed this video.' : 'Schema, segmentation, and checkpoint provenance passed the fail-closed gate.'}</span>
        </div>
      </div>

      <div className="results-heading">
        <div>
          <span className="eyebrow">{isDemo ? 'Interface demonstration' : 'Research output'}</span>
          <h2 id="results-title">Regional movement summary</h2>
          <p>{isDemo ? 'Preview the result layout only. These values were not produced by a model.' : 'Review the source video alongside these model outputs. They do not replace clinician assessment.'}</p>
        </div>
        <button className="button button-secondary" type="button" onClick={onReset}>Start a new session</button>
      </div>

      <div className="probability-card">
        <div className="probability-copy">
          <span className="region-icon"><ScanFace aria-hidden="true" size={24} /></span>
          <div>
            <span>{isDemo ? 'Interface preview value' : 'Binary model output'}</span>
            <h3>{isDemo ? 'Demonstration probability layout' : 'Uncalibrated research probability'}</h3>
            <p>{isDemo ? 'Synthetic display value generated from local file metadata.' : 'Public-data warm-start probability of the model’s “palsy” class.'}</p>
          </div>
        </div>
        <div className="probability-value">
          <strong>{percent(scores.palsyProbability)}</strong>
          <span>{isDemo ? 'not model output' : 'not a diagnosis'}</span>
        </div>
      </div>

      <div className="region-grid">
        <RegionCard title="Eye region" icon={<Eye aria-hidden="true" size={22} />} region={scores.eyes} demonstration={isDemo} />
        <RegionCard title="Mouth region" icon={<ScanFace aria-hidden="true" size={22} />} region={scores.mouth} demonstration={isDemo} />
      </div>

      {!isDemo ? (
        <div className="provenance-card">
          <Info aria-hidden="true" size={20} />
          <div>
            <strong>Validated response provenance</strong>
            <dl>
              <div><dt>Checkpoint</dt><dd>{result.provenance.modelFile}</dd></div>
              <div><dt>SHA-256</dt><dd><code>{result.provenance.modelSha256.slice(0, 12)}…{result.provenance.modelSha256.slice(-8)}</code></dd></div>
              <div><dt>Preprocessing</dt><dd>{result.provenance.preprocessingVersion}</dd></div>
              <div><dt>Segmentation</dt><dd>{result.provenance.segmentationVersion}</dd></div>
            </dl>
          </div>
        </div>
      ) : null}
    </section>
  )
}
