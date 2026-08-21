import { useState, type CSSProperties } from 'react'
import type {
  PatientAttentionPoint,
  PatientComparisonResultEntry,
  PatientComparisonTimepoint,
} from '../patientWorkflow/types'
import { AttentionColorLegend } from './AttentionColorLegend'
import { PatientFaceContour } from './PatientFaceContour'
import { attentionColorRgb } from './attentionColorScale'

type ComparisonEntry = PatientComparisonResultEntry & {
  readonly previewUrl: string
}

type PatientLongitudinalComparisonProps = {
  readonly pair: Readonly<{
    preoperative: ComparisonEntry
    postoperative: ComparisonEntry
  }>
}

type DisplayMode = 'photo' | 'outline'

const TIMEPOINTS = ['preoperative', 'postoperative'] as const

const TIMEPOINT_LABELS: Readonly<
  Record<PatientComparisonTimepoint, string>
> = {
  preoperative: 'Preoperative',
  postoperative: 'Postoperative',
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00.000Z`))
}

function AttentionLayer({
  points,
}: {
  readonly points: readonly PatientAttentionPoint[]
}) {
  return (
    <div
      className="patient-longitudinal-comparison__attention-layer"
      aria-hidden="true"
    >
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
              '--attention-color-rgb': attentionColorRgb(
                point.intensity,
              ),
            } as CSSProperties
          }
        />
      ))}
    </div>
  )
}

export function PatientLongitudinalComparison({
  pair,
}: PatientLongitudinalComparisonProps) {
  const [displayMode, setDisplayMode] = useState<DisplayMode>('photo')
  const [showAttention, setShowAttention] = useState(true)

  return (
    <section
      className="patient-longitudinal-comparison"
      aria-label="Patient before and after comparison"
    >
      <div className="patient-longitudinal-comparison__header">
        <div>
          <p className="patient-longitudinal-comparison__eyebrow">
            Latest pre- and postoperative visits
          </p>
          <h2>Before and after</h2>
          <p>
            Compare the same patient at both timepoints without leaving
            this record.
          </p>
        </div>

        <div className="patient-longitudinal-comparison__controls">
          <fieldset>
            <legend>Display</legend>
            <div className="patient-longitudinal-comparison__segments">
              {(['photo', 'outline'] as const).map((mode) => (
                <label key={mode}>
                  <input
                    checked={displayMode === mode}
                    name="patient-comparison-display"
                    onChange={() => setDisplayMode(mode)}
                    type="radio"
                    value={mode}
                  />
                  <span>
                    {mode === 'photo' ? 'Photo' : 'Outline'}
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          <label className="patient-longitudinal-comparison__toggle">
            <input
              checked={showAttention}
              onChange={(event) =>
                setShowAttention(event.currentTarget.checked)
              }
              type="checkbox"
            />
            <span>Show attention layer</span>
          </label>
        </div>
      </div>

      <div className="patient-longitudinal-comparison__grid">
        {TIMEPOINTS.map((timepoint) => {
          const entry = pair[timepoint]
          const label = TIMEPOINT_LABELS[timepoint]
          return (
            <article
              className="patient-longitudinal-comparison__card"
              key={timepoint}
            >
              <header>
                <h3>{label}</h3>
                <time dateTime={entry.visit.visitDate}>
                  {formatDate(entry.visit.visitDate)}
                </time>
              </header>
              <div
                className={`patient-longitudinal-comparison__media patient-longitudinal-comparison__media--${displayMode}`}
                style={{
                  aspectRatio: `${entry.capture.width} / ${entry.capture.height}`,
                }}
                {...(displayMode === 'outline'
                  ? {
                      role: 'img',
                      'aria-label': `${label} face outline${
                        showAttention
                          ? ' with illustrative attention'
                          : ''
                      }`,
                    }
                  : {})}
              >
                {displayMode === 'photo' ? (
                  <img
                    src={entry.previewUrl}
                    alt={`${label} patient photograph${
                      showAttention
                        ? ' with illustrative attention'
                        : ''
                    }`}
                    width={entry.capture.width}
                    height={entry.capture.height}
                    decoding="async"
                  />
                ) : (
                  <PatientFaceContour
                    registration={entry.result.faceRegistration}
                  />
                )}
                {showAttention ? (
                  <AttentionLayer points={entry.result.output.points} />
                ) : null}
              </div>
            </article>
          )
        })}
      </div>

      {showAttention ? <AttentionColorLegend /> : null}

      <p className="patient-longitudinal-comparison__disclosure">
        Illustrative workflow output—not measured gaze, a clinical
        measurement, or evidence of treatment effect.
      </p>
    </section>
  )
}
