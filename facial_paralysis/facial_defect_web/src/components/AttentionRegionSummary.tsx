import type { CSSProperties } from 'react'
import { useId } from 'react'
import type { AttentionPresentation } from '../workbench/attentionPresentation'
import type { InferenceOutput } from '../workbench/types'

type AttentionRegionSummaryProps = {
  presentation: Extract<AttentionPresentation, { ok: true }>
  origin: InferenceOutput['origin']
}

function describeRelativeSignal(level: number): string {
  if (level === 0) return 'No displayed signal'
  if (level < 1 / 3) return 'Lower relative signal'
  if (level < 2 / 3) return 'Moderate relative signal'
  return 'Higher relative signal'
}

export function AttentionRegionSummary({
  presentation,
  origin,
}: AttentionRegionSummaryProps) {
  const titleId = useId()

  return (
    <section className="attention-summary" aria-labelledby={titleId}>
      <div className="attention-result-section__heading">
        <div>
          <h3 id={titleId}>Regional summary</h3>
          <p>The result field divided into nine image-relative areas.</p>
        </div>
      </div>

      <div
        className="attention-summary__grid"
        role="list"
        aria-label="Result field summary"
      >
        {presentation.cells.map((cell) => (
          <div
            className="attention-summary__cell"
            data-dominant={cell.id === presentation.dominantCell?.id || undefined}
            key={cell.id}
            role="listitem"
            style={{ '--signal-level': cell.level } as CSSProperties}
          >
            <span className="attention-summary__position">{cell.label}</span>
            <span className="attention-summary__band">
              {describeRelativeSignal(cell.level)}
            </span>
          </div>
        ))}
      </div>

      <p className="attention-summary__finding">
        {presentation.dominantCell
          ? `The strongest displayed relative signal is in the ${presentation.dominantCell.label.toLowerCase()}.`
          : origin === 'model_prediction'
            ? 'No regional difference is available from this research model result.'
            : 'No regional difference is available from this simulated result.'}
      </p>
    </section>
  )
}
