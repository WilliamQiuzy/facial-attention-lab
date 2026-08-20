import { useState } from 'react'
import { AttentionColorLegend } from '../components/AttentionColorLegend'
import {
  PRESENTATION_BOUNDARY,
  type PresentationTimepoint,
} from '../data/presentationDemoAssets'
import {
  PresentationAttentionStage,
  type PresentationViewMode,
} from '../presentation/PresentationAttentionStage'

type TimepointSelection = PresentationTimepoint | 'both'

const timepointOptions: readonly Readonly<{
  value: TimepointSelection
  label: string
}>[] = [
  { value: 'preoperative', label: 'Pre-operative' },
  { value: 'postoperative', label: 'Post-operative' },
  { value: 'both', label: 'Both' },
]

const viewOptions: readonly Readonly<{
  value: PresentationViewMode
  label: string
}>[] = [
  { value: 'photo', label: 'Photo' },
  { value: 'outline', label: 'Outline' },
]

function SegmentedOption({
  checked,
  label,
  name,
  onChange,
}: {
  readonly checked: boolean
  readonly label: string
  readonly name: string
  readonly onChange: () => void
}) {
  return (
    <label className="presentation-segment">
      <input
        type="radio"
        name={name}
        checked={checked}
        onChange={onChange}
      />
      <span>{label}</span>
    </label>
  )
}

export function PresentationDemoPage() {
  const [timepoint, setTimepoint] =
    useState<TimepointSelection>('both')
  const [viewMode, setViewMode] =
    useState<PresentationViewMode>('photo')
  const [showAttention, setShowAttention] = useState(true)
  const visibleTimepoints: readonly PresentationTimepoint[] =
    timepoint === 'both'
      ? ['preoperative', 'postoperative']
      : [timepoint]

  return (
    <article className="presentation-page page-shell">
      <header className="presentation-hero">
        <div>
          <p className="presentation-hero__kicker">Presentation demo</p>
          <h1>Before and after, at a glance</h1>
          <p className="presentation-hero__lede">
            A synthetic example for discussing how visible attention could be
            compared before and after facial reconstruction.
          </p>
        </div>
        <aside className="presentation-hero__note" aria-label="Demo explanation">
          <strong>What changes in this example</strong>
          <p>
            The cheek lesion is replaced with a small, flat scar-like edit. A
            hand-authored attention signal remains at the site, but is
            deliberately less prominent.
          </p>
        </aside>
      </header>

      <p className="presentation-boundary" role="note">
        {PRESENTATION_BOUNDARY}
      </p>

      <section className="presentation-controls" aria-label="Demo controls">
        <fieldset>
          <legend>Time point</legend>
          <div className="presentation-segments">
            {timepointOptions.map((option) => (
              <SegmentedOption
                key={option.value}
                checked={timepoint === option.value}
                label={option.label}
                name="presentation-timepoint"
                onChange={() => setTimepoint(option.value)}
              />
            ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>Display</legend>
          <div className="presentation-segments">
            {viewOptions.map((option) => (
              <SegmentedOption
                key={option.value}
                checked={viewMode === option.value}
                label={option.label}
                name="presentation-view"
                onChange={() => setViewMode(option.value)}
              />
            ))}
          </div>
        </fieldset>

        <label className="presentation-toggle">
          <input
            type="checkbox"
            checked={showAttention}
            onChange={(event) => setShowAttention(event.target.checked)}
          />
          <span>Show attention layer</span>
        </label>
      </section>

      {showAttention ? (
        <div className="presentation-legend">
          <span>Relative simulated intensity</span>
          <AttentionColorLegend />
        </div>
      ) : null}

      <section
        className={`presentation-comparison${visibleTimepoints.length === 1 ? ' presentation-comparison--single' : ''}`}
        aria-label="Before and after comparison"
      >
        {visibleTimepoints.map((visibleTimepoint) => (
          <PresentationAttentionStage
            key={visibleTimepoint}
            timepoint={visibleTimepoint}
            viewMode={viewMode}
            showAttention={showAttention}
          />
        ))}
      </section>

      <section className="presentation-context" aria-labelledby="presentation-context-title">
        <h2 id="presentation-context-title">How to discuss this demo</h2>
        <p>
          The two images and both attention layers are synthetic presentation
          materials. This interface demonstrates a possible comparison
          workflow; it does not estimate an individual patient's result or
          report measured human gaze.
        </p>
        <dl>
          <div>
            <dt>Photo</dt>
            <dd>Shows the visible synthetic face and site directly.</dd>
          </div>
          <div>
            <dt>Outline</dt>
            <dd>
              Uses 13 MediaPipe-derived contour paths from the exact displayed
              image to reduce self-recognition distractions.
            </dd>
          </div>
        </dl>
      </section>
    </article>
  )
}
