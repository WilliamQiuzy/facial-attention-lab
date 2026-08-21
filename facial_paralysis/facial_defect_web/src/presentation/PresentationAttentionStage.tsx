import type { CSSProperties } from 'react'
import { PatientFaceContour } from '../components/PatientFaceContour'
import { attentionColorRgb } from '../components/attentionColorScale'
import {
  presentationDemoAssets,
  presentationSubjectOptions,
  type PresentationSubjectId,
  type PresentationTimepoint,
} from '../data/presentationDemoAssets'
import { presentationAttentionBySubject } from './presentationAttention'
import { registrationBySubject } from './presentationFaceRegistration'

export type PresentationViewMode = 'photo' | 'outline' | 'composite'

type PresentationAttentionStageProps = {
  readonly subjectId: PresentationSubjectId
  readonly timepoint: PresentationTimepoint
  readonly viewMode: PresentationViewMode
  readonly showAttention: boolean
}

const timepointCopy = {
  preoperative: {
    eyebrow: 'Pre-operative',
    title: 'Visible cheek lesion',
    imageAlt: 'Sample pre-operative facial photograph',
  },
  postoperative: {
    eyebrow: 'Post-operative',
    title: 'Healing surgical incision',
  },
} as const

const viewModeLabels: Readonly<Record<PresentationViewMode, string>> = {
  photo: 'Photo',
  outline: 'Outline',
  composite: 'Photo + outline',
}

function PresentationSignalLayer({
  subjectId,
  timepoint,
}: {
  readonly subjectId: PresentationSubjectId
  readonly timepoint: PresentationTimepoint
}) {
  const points = presentationAttentionBySubject[subjectId][timepoint]
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

export function PresentationMedia({
  subjectId,
  timepoint,
  viewMode,
  showAttention,
}: PresentationAttentionStageProps) {
  const asset = presentationDemoAssets[subjectId][timepoint]
  const copy = timepointCopy[timepoint]
  const subjectLabel = presentationSubjectOptions.find(
    (subject) => subject.id === subjectId,
  )?.label
  const outlineLabel = `Abstract facial outline with illustrative ${copy.eyebrow.toLowerCase()} attention`
  const showPhoto = viewMode !== 'outline'
  const showOutline = viewMode !== 'photo'

  return (
    <div
      className={`presentation-stage__media presentation-stage__media--${viewMode}`}
      role={viewMode === 'outline' ? 'img' : undefined}
      aria-label={viewMode === 'outline' ? outlineLabel : undefined}
    >
      {showPhoto ? (
        <img
          src={asset.url}
          alt={`${subjectLabel} sample ${copy.eyebrow.toLowerCase()} facial photograph`}
          width={asset.width}
          height={asset.height}
          loading="eager"
          decoding="async"
        />
      ) : null}
      {showOutline ? (
        <PatientFaceContour registration={registrationBySubject[subjectId][timepoint]} />
      ) : null}
      {showAttention ? (
        <PresentationSignalLayer subjectId={subjectId} timepoint={timepoint} />
      ) : (
        <span className="presentation-stage__layer-status">Attention off</span>
      )}
    </div>
  )
}

export function PresentationAttentionStage({
  subjectId,
  timepoint,
  viewMode,
  showAttention,
}: PresentationAttentionStageProps) {
  const copy = timepointCopy[timepoint]

  return (
    <figure className="presentation-stage">
      <header className="presentation-stage__header">
        <div>
          <p>{copy.eyebrow}</p>
          <h2>{copy.title}</h2>
        </div>
        <span>{viewModeLabels[viewMode]}</span>
      </header>

      <PresentationMedia
        subjectId={subjectId}
        timepoint={timepoint}
        viewMode={viewMode}
        showAttention={showAttention}
      />
    </figure>
  )
}
