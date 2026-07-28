import type { CSSProperties } from 'react'
import { useId } from 'react'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'
import type {
  ClinicalAoiPresentation,
} from '../workbench/clinicalAoiPresentation'

type SuccessfulClinicalAoiPresentation = Extract<
  ClinicalAoiPresentation,
  { ok: true }
>

type ClinicalAoiSummaryProps = {
  readonly asset: WorkbenchCatalogEntry
  readonly presentation: SuccessfulClinicalAoiPresentation
}

type ShareRow = Readonly<{
  label: string
  share: number
  displayedPercent: number
}>

function formatShare(share: number): string {
  return `${Math.round(share * 100)}%`
}

function largestRemainderPercentages(shares: readonly number[]): number[] {
  const total = shares.reduce((sum, share) => sum + share, 0)
  if (!(total > 0)) return shares.map(() => 0)

  const quotas = shares.map((share) => (share / total) * 100)
  const percentages = quotas.map(Math.floor)
  const remaining = 100 - percentages.reduce((sum, value) => sum + value, 0)
  const priority = quotas
    .map((quota, index) => ({
      index,
      remainder: quota - percentages[index],
    }))
    .sort(
      (first, second) =>
        second.remainder - first.remainder || first.index - second.index,
    )

  for (let index = 0; index < remaining; index += 1) {
    percentages[priority[index].index] += 1
  }
  return percentages
}

export function ClinicalAoiSummary({
  asset,
  presentation,
}: ClinicalAoiSummaryProps) {
  const titleId = useId()
  const rawTemplateShares = [
    ...presentation.subsites,
    { label: 'Outside template', share: presentation.outsideTemplateShare },
  ]
  const templatePercentages = largestRemainderPercentages(
    rawTemplateShares.map((item) => item.share),
  )
  const templateShares: readonly ShareRow[] = rawTemplateShares.map(
    (item, index) => ({
      ...item,
      displayedPercent: templatePercentages[index],
    }),
  )
  const rawHemifaceShares = [
    {
      label: 'Patient left',
      share: presentation.hemifaces.patientLeftShare,
    },
    {
      label: 'Patient right',
      share: presentation.hemifaces.patientRightShare,
    },
  ]
  const hemifacePercentages = largestRemainderPercentages(
    rawHemifaceShares.map((item) => item.share),
  )
  const hemifaceShares: readonly ShareRow[] = rawHemifaceShares.map(
    (item, index) => ({
      ...item,
      displayedPercent: hemifacePercentages[index],
    }),
  )
  const dominantSubsiteIndex = presentation.dominantSubsite
    ? presentation.subsites.findIndex(
        (subsite) => subsite.id === presentation.dominantSubsite?.id,
      )
    : -1
  const dominantDisplayedPercent =
    dominantSubsiteIndex >= 0
      ? templatePercentages[dominantSubsiteIndex]
      : undefined

  const renderShareRows = (shares: readonly ShareRow[]) =>
    shares.map((item) => (
      <li key={item.label}>
        <span className="clinical-aoi-summary__share-label">{item.label}</span>
        <span
          className="clinical-aoi-summary__bar"
          aria-hidden="true"
          style={
            {
              '--aoi-share': `${item.share * 100}%`,
            } as CSSProperties
          }
        >
          <span />
        </span>
        <strong>{item.displayedPercent}%</strong>
      </li>
    ))

  return (
    <section className="clinical-aoi-summary" aria-labelledby={titleId}>
      <header className="clinical-aoi-summary__header">
        <h3 id={titleId}>Clinical AOI summary</h3>
        <p>
          Shares use simulated point-center intensity weights assigned to the
          fixed template; display radius and boundary overlap are not integrated.
          Not an eye-tracking measurement, severity, or outcome.
        </p>
      </header>

      <div className="clinical-aoi-summary__content">
        <figure className="clinical-aoi-summary__figure">
          <div className="clinical-aoi-summary__image">
            <img
              src={asset.url}
              alt={`${asset.label}: synthetic source face with AOI template`}
              width="1024"
              height="1024"
              loading="eager"
              decoding="async"
            />
            <svg
              className="clinical-aoi-summary__template"
              viewBox="0 0 100 100"
              aria-hidden="true"
              focusable="false"
            >
              <rect x="25" y="16" width="50" height="18" />
              <rect x="25" y="34" width="50" height="14" />
              <rect x="25" y="48" width="50" height="18" />
              <rect x="25" y="66" width="50" height="18" />
              <path d="M 28 34 L 72 34 L 50 84 Z" />
            </svg>
          </div>
          <div
            className="patient-orientation"
            role="group"
            aria-label="Patient orientation"
          >
            <span>Patient right (viewer left)</span>
            <span>Patient left (viewer right)</span>
            <small>Frontal, non-mirrored synthetic display.</small>
          </div>
          <figcaption>
            Fixed anatomical template — simulation; no landmarks detected.
          </figcaption>
        </figure>

        <div className="clinical-aoi-summary__readout">
          <p className="clinical-aoi-summary__registration">
            Registration: Synthetic template v1
          </p>
          <section
            className="clinical-aoi-summary__share-group"
            role="region"
            aria-label="Anatomical template shares"
          >
            <h4>Anatomical template</h4>
            <ul className="clinical-aoi-summary__shares">
              {renderShareRows(templateShares)}
            </ul>
            <p>
              {presentation.totalMass === 0
                ? 'No non-zero point weight to distribute.'
                : 'Subsite template shares + outside template = 100%.'}
            </p>
          </section>
          <section
            className="clinical-aoi-summary__share-group"
            role="region"
            aria-label="Hemiface shares"
          >
            <h4>Hemifaces</h4>
            <ul className="clinical-aoi-summary__shares">
              {renderShareRows(hemifaceShares)}
            </ul>
            <p>
              {presentation.totalMass === 0
                ? 'No non-zero point weight to distribute.'
                : 'Hemifaces = 100%.'}
            </p>
          </section>
          <section
            className="clinical-aoi-summary__overlap"
            role="region"
            aria-label="Central triangle share"
          >
            <div>
              <span>Central triangle</span>
              <strong>{formatShare(presentation.centralTriangleShare)}</strong>
            </div>
            <p>Overlapping AOI · not additive to either group.</p>
          </section>
          <p className="clinical-aoi-summary__finding">
            {presentation.totalMass === 0
              ? 'No dominant anatomical AOI is available from this simulated interface result.'
              : presentation.dominantSubsite &&
                  dominantDisplayedPercent !== undefined
                ? `Largest anatomical AOI share: ${presentation.dominantSubsite.label} (${dominantDisplayedPercent}%).`
                : 'No anatomical subsite contains displayed density in this result.'}
          </p>
        </div>
      </div>

      <p className="clinical-aoi-summary__status">
        AOIs summarize the completed field and do not change the simulation.
      </p>
      <aside className="clinical-aoi-summary__mask-note">
        <strong>Surgical-site mask: not set</strong>
        <span>
          This future, separately versioned contextual annotation is absent. It
          is not the immutable image bound and does not alter the result.
        </span>
      </aside>
    </section>
  )
}
