import { useState, type CSSProperties } from 'react'
import {
  PresentationMedia,
  type PresentationViewMode,
} from './PresentationAttentionStage'
import type { PresentationSubjectId } from '../data/presentationDemoAssets'

type PresentationBeforeAfterSliderProps = {
  readonly subjectId: PresentationSubjectId
  readonly viewMode: PresentationViewMode
  readonly showAttention: boolean
}

export function PresentationBeforeAfterSlider({
  subjectId,
  viewMode,
  showAttention,
}: PresentationBeforeAfterSliderProps) {
  const [position, setPosition] = useState(50)

  return (
    <section
      className="presentation-wipe"
      aria-label="Interactive before and after comparison"
      style={
        {
          '--presentation-comparison-position': `${position}%`,
        } as CSSProperties
      }
    >
      <div className="presentation-wipe__labels" aria-hidden="true">
        <span>Pre-op</span>
        <span>Post-op</span>
      </div>
      <p className="presentation-wipe__help">
        Drag the divider to compare the same facial location before and after surgery.
      </p>

      <div className="presentation-wipe__frame">
        <PresentationMedia
          subjectId={subjectId}
          timepoint="postoperative"
          viewMode={viewMode}
          showAttention={showAttention}
        />
        <div className="presentation-wipe__before" aria-hidden="true">
          <PresentationMedia
            subjectId={subjectId}
            timepoint="preoperative"
            viewMode={viewMode}
            showAttention={showAttention}
          />
        </div>
        <span className="presentation-wipe__divider" aria-hidden="true">
          <span>↔</span>
        </span>
      </div>

      <label className="presentation-wipe__control">
        <span>Comparison position</span>
        <input
          aria-label="Comparison position"
          type="range"
          min="0"
          max="100"
          value={position}
          onChange={(event) => setPosition(Number(event.target.value))}
        />
        <output>{position}% pre-op</output>
      </label>
    </section>
  )
}
