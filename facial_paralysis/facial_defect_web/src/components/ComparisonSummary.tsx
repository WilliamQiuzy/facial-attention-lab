import { Info } from 'lucide-react'
import type { AttentionComparison } from '../model/types'

export function ComparisonSummary({ comparison }: { comparison: AttentionComparison }) {
  return (
    <section className="comparison-summary" aria-labelledby="comparison-summary-title">
      <div className="comparison-summary__delta" aria-hidden="true">
        <span>Layout delta</span>
        <strong>{comparison.scarGazeChangePoints} pts</strong>
      </div>
      <div>
        <p className="eyebrow">How to read this demo</p>
        <h2 id="comparison-summary-title">A difference in two fixtures—not a clinical outcome.</h2>
        <p>{comparison.interpretation}</p>
        <p className="inline-note">
          <Info aria-hidden="true" /> The {comparison.relativeReductionPercent}% relative
          difference is included to test interface layout. It is not evidence of surgical
          change.
        </p>
      </div>
    </section>
  )
}
