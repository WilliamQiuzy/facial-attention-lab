import type { CSSProperties } from 'react'
import type {
  PatientAttentionPoint,
} from '../patientWorkflow/types'

type PatientAttentionImagesProps = {
  readonly previewUrl: string
  readonly width: number
  readonly height: number
  readonly points: readonly PatientAttentionPoint[]
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
}: PatientAttentionImagesProps) {
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
        <h3>Attention density</h3>
        <div
          className="patient-attention-images__density-plane"
          role="img"
          aria-label="Simulated attention density without the source photograph"
          style={{ aspectRatio: `${width} / ${height}` }}
        >
          <AttentionPointLayer
            points={points}
            className="patient-attention-images__density-layer"
          />
          <span className="patient-attention-images__watermark">
            Simulated
          </span>
        </div>
        <figcaption>
          Brighter overlap indicates more simulated attention density.
          Same deterministic simulation.
        </figcaption>
      </figure>
    </section>
  )
}
