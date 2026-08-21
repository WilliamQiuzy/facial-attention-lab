import { useId } from 'react'
import type {
  PatientAttentionPoint,
  PatientFaceRegistration,
} from '../patientWorkflow/types'
import {
  deriveClinicalAoiPresentation,
} from '../workbench/clinicalAoiPresentation'

type PatientAoiSummaryProps = {
  readonly points: readonly PatientAttentionPoint[]
  readonly faceRegistration: PatientFaceRegistration
}

function formatShare(share: number): string {
  return `${Math.round(share * 100)}%`
}

export function PatientAoiSummary({
  points,
  faceRegistration,
}: PatientAoiSummaryProps) {
  const titleId = useId()
  const ovalPoints = faceRegistration.paths
    .filter((path) => path.feature === 'face_oval')
    .flatMap((path) => path.points)
  const minimumX = Math.min(...ovalPoints.map((point) => point.x))
  const maximumX = Math.max(...ovalPoints.map((point) => point.x))
  const minimumY = Math.min(...ovalPoints.map((point) => point.y))
  const maximumY = Math.max(...ovalPoints.map((point) => point.y))
  const presentation = deriveClinicalAoiPresentation(points, {
    x: minimumX,
    y: minimumY,
    width: maximumX - minimumX,
    height: maximumY - minimumY,
  })

  return (
    <section
      className="patient-aoi-summary"
      aria-labelledby={titleId}
    >
      <header className="patient-aoi-summary__header">
        <h3 id={titleId}>Attention by facial area</h3>
        <p>
          Face-relative areas summarize this attention density. They do
          not change the analysis.
        </p>
      </header>

      {!presentation.ok ? (
        <p className="patient-aoi-summary__unavailable" role="status">
          A facial-area summary is unavailable for this result.
        </p>
      ) : (
        <>
          <ul className="patient-aoi-summary__areas">
            {presentation.subsites.map((subsite) => (
              <li key={subsite.id}>
                <span>{subsite.label}</span>
                <strong>{formatShare(subsite.share)}</strong>
              </li>
            ))}
            <li>
              <span>Outside four face-relative bands</span>
              <strong>
                {formatShare(presentation.outsideTemplateShare)}
              </strong>
            </li>
          </ul>
          <div
            className="patient-aoi-summary__laterality"
            aria-label="Patient-side summary"
          >
            <p>
              <span>Patient right (viewer left)</span>
              <strong>
                {formatShare(
                  presentation.hemifaces.patientRightShare,
                )}
              </strong>
            </p>
            <p>
              <span>Patient left (viewer right)</span>
              <strong>
                {formatShare(
                  presentation.hemifaces.patientLeftShare,
                )}
              </strong>
            </p>
          </div>
        </>
      )}
      {presentation.ok ? (
        <details className="patient-aoi-summary__method">
          <summary>How percentages are calculated</summary>
          <div>
            <p>
              Percentages use face-relative areas positioned within the
              face contour estimated from this photograph. The contour
              is a spatial reference, not clinical anatomical
              segmentation.
            </p>
            <p>
              Four face-relative bands plus density outside those bands
              total 100%. Patient-right and patient-left shares form a
              separate 100% partition.
            </p>
          </div>
        </details>
      ) : null}
    </section>
  )
}
