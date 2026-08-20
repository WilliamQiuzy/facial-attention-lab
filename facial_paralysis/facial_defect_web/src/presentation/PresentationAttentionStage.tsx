import type { CSSProperties } from 'react'
import { PatientFaceContour } from '../components/PatientFaceContour'
import { attentionColorRgb } from '../components/attentionColorScale'
import {
  PRESENTATION_BOUNDARY,
  presentationDemoAssets,
  type PresentationTimepoint,
} from '../data/presentationDemoAssets'
import { presentationAttentionByTimepoint } from './presentationAttention'
import { registrationByTimepoint } from './presentationFaceRegistration'

export type PresentationViewMode = 'photo' | 'outline'

type PresentationAttentionStageProps = {
  readonly timepoint: PresentationTimepoint
  readonly viewMode: PresentationViewMode
  readonly showAttention: boolean
}

const timepointCopy = {
  preoperative: {
    eyebrow: 'Pre-operative',
    title: 'Visible cheek lesion',
    imageAlt: 'AI-generated synthetic pre-operative facial photograph',
  },
  postoperative: {
    eyebrow: 'Post-operative',
    title: 'Small cheek scar',
    imageAlt: 'AI-generated synthetic post-operative facial photograph',
  },
} as const

function PresentationSignalLayer({
  timepoint,
}: {
  readonly timepoint: PresentationTimepoint
}) {
  const points = presentationAttentionByTimepoint[timepoint]
  const totalSignal = points.reduce(
    (total, point) => total + point.intensity,
    0,
  )

  return (
    <div
      aria-hidden="true"
      className="presentation-signal-layer"
      data-total-signal={totalSignal.toFixed(3)}
    >
      {points.map((point, index) => (
        <span
          className="presentation-signal-point"
          key={`${point.x}-${point.y}-${index}`}
          style={
            {
              '--presentation-point-x': `${point.x * 100}%`,
              '--presentation-point-y': `${point.y * 100}%`,
              '--presentation-point-size': `${point.radius * 200}%`,
              '--presentation-point-intensity': point.intensity,
              '--attention-color-rgb': attentionColorRgb(point.intensity),
            } as CSSProperties
          }
        />
      ))}
    </div>
  )
}

export function PresentationAttentionStage({
  timepoint,
  viewMode,
  showAttention,
}: PresentationAttentionStageProps) {
  const asset = presentationDemoAssets[timepoint]
  const copy = timepointCopy[timepoint]
  const outlineLabel = `Abstract facial outline with simulated ${copy.eyebrow.toLowerCase()} attention`

  return (
    <figure className="presentation-stage">
      <header className="presentation-stage__header">
        <div>
          <p>{copy.eyebrow}</p>
          <h2>{copy.title}</h2>
        </div>
        <span>{viewMode === 'photo' ? 'Photo' : 'De-identified outline'}</span>
      </header>

      <div
        className={`presentation-stage__media presentation-stage__media--${viewMode}`}
        role={viewMode === 'outline' ? 'img' : undefined}
        aria-label={viewMode === 'outline' ? outlineLabel : undefined}
      >
        {viewMode === 'photo' ? (
          <img
            src={asset.url}
            alt={copy.imageAlt}
            width={asset.width}
            height={asset.height}
            loading="eager"
            decoding="async"
          />
        ) : (
          <PatientFaceContour
            registration={registrationByTimepoint[timepoint]}
          />
        )}
        {showAttention ? (
          <PresentationSignalLayer timepoint={timepoint} />
        ) : (
          <span className="presentation-stage__layer-status">
            Attention layer hidden
          </span>
        )}
        <span className="presentation-stage__watermark">
          Hand-authored simulation
        </span>
      </div>

      <figcaption>
        <span>{PRESENTATION_BOUNDARY}</span>
        <small>
          Frontal, non-mirrored synthetic display. Patient left is viewer right.
        </small>
      </figcaption>
    </figure>
  )
}
