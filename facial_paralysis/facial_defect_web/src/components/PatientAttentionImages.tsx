import { useId, type CSSProperties } from 'react'
import type {
  PatientAttentionPoint,
  PatientFaceRegistration,
} from '../patientWorkflow/types'
import { PatientFaceContour } from './PatientFaceContour'

type PatientAttentionImagesProps = {
  readonly previewUrl: string
  readonly width: number
  readonly height: number
  readonly points: readonly PatientAttentionPoint[]
  readonly faceRegistration?: PatientFaceRegistration
}

function AttentionPointLayer({
  points,
  className,
}: {
  readonly points: readonly PatientAttentionPoint[]
  readonly className: string
}) {
  return (
    <div className={className} aria-hidden="true">
      {points.map((point, index) => (
        <span
          className="patient-attention-point"
          key={`${point.x}-${point.y}-${index}`}
          style={
            {
              '--patient-point-x': `${point.x * 100}%`,
              '--patient-point-y': `${point.y * 100}%`,
              '--patient-point-size': `${point.radius * 200}%`,
              '--patient-point-intensity': point.intensity,
            } as CSSProperties
          }
        />
      ))}
    </div>
  )
}

export function PatientAttentionImages({
  previewUrl,
  width,
  height,
  points,
  faceRegistration,
}: PatientAttentionImagesProps) {
  const densityDescriptionId = useId()
  const hasMatchedContour = faceRegistration !== undefined
  const densityDescription = hasMatchedContour
    ? 'Automatically estimated from this photograph for spatial reference. It is not a defect boundary, clinical segmentation, or attention prediction.'
    : 'A face contour could not be matched reliably to this photograph. Retake or upload a centered frontal photograph before using a contour reference.'

  return (
    <section
      className="patient-attention-images"
      aria-label="Patient attention images"
    >
      <div className="patient-attention-images__primary">
        <figure className="patient-attention-images__figure">
          <h3>Original photograph</h3>
          <div className="patient-attention-images__image-plane">
            <img
              src={previewUrl}
              alt="Original frontal photograph"
              width={width}
              height={height}
              loading="eager"
              decoding="async"
            />
          </div>
        </figure>

        <figure className="patient-attention-images__figure">
          <h3>Simulated overlay</h3>
          <div className="patient-attention-images__image-plane">
            <img
              src={previewUrl}
              alt="Frontal photograph with simulated attention overlay"
              width={width}
              height={height}
              loading="eager"
              decoding="async"
            />
            <AttentionPointLayer
              points={points}
              className="patient-attention-images__overlay-layer"
            />
          </div>
        </figure>
      </div>

      <p className="patient-attention-images__orientation">
        Patient right is viewer left; patient left is viewer right.
        Orientation confirmed for this frontal, non-mirrored photograph.
      </p>

      <figure className="patient-attention-images__figure patient-attention-images__density">
        <h3>
          {hasMatchedContour
            ? 'Attention density + matched face contour'
            : 'Attention density'}
        </h3>
        <div
          className="patient-attention-images__density-plane"
          role="img"
          aria-label={
            hasMatchedContour
              ? "Simulated attention density aligned to this photograph's estimated face contour"
              : 'Simulated attention density; face contour unavailable'
          }
          aria-describedby={densityDescriptionId}
          style={{ aspectRatio: `${width} / ${height}` }}
        >
          <AttentionPointLayer
            points={points}
            className="patient-attention-images__density-layer"
          />
          {faceRegistration ? (
            <PatientFaceContour registration={faceRegistration} />
          ) : null}
          <span className="patient-attention-images__watermark">
            Simulated
          </span>
        </div>
        <figcaption id={densityDescriptionId}>
          {densityDescription}
        </figcaption>
      </figure>
    </section>
  )
}
