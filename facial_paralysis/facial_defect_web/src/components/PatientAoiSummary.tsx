import { useId } from 'react'
import type {
  PatientAttentionPoint,
} from '../patientWorkflow/types'
import {
  deriveClinicalAoiPresentation,
} from '../workbench/clinicalAoiPresentation'

type PatientAoiSummaryProps = {
  readonly points: readonly PatientAttentionPoint[]
}

function formatShare(share: number): string {
  return `${Math.round(share * 100)}% of simulated density`
}

export function PatientAoiSummary({
  points,
}: PatientAoiSummaryProps) {
  const titleId = useId()
  const presentation = deriveClinicalAoiPresentation(points, {
    x: 0,
    y: 0,
    width: 1,
    height: 1,
  })

  return (
    <section
      className="patient-aoi-summary"
      aria-labelledby={titleId}
    >
      <header className="patient-aoi-summary__header">
        <h3 id={titleId}>Attention by facial area</h3>
        <p>
          AOI is a post-inference summary only. It does not crop the
          photograph or alter the simulation.
        </p>
        <p>
          Percentages use a fixed illustrative face template, not detected
          landmarks or patient-specific anatomical registration.
        </p>
        <p>
          Four template facial bands plus outside-template density total
          100%. Patient-right and patient-left shares form a separate 100%
          partition.
        </p>
      </header>

      {!presentation.ok ? (
        <p className="patient-aoi-summary__unavailable" role="status">
          A facial-area summary is unavailable for this simulated result.
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
              <span>Outside four template facial bands</span>
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
    </section>
  )
}
