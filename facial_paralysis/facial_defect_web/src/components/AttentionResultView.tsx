import { useId, useState } from 'react'
import { deriveClinicalAoiPresentation } from '../workbench/clinicalAoiPresentation'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'
import type { InferenceOutput, RoiAnnotation } from '../workbench/types'
import { AttentionMap } from './AttentionMap'
import { AttentionSignalField } from './AttentionSignalField'
import { ClinicalAoiSummary } from './ClinicalAoiSummary'

type ResultViewMode = 'aoi' | 'density' | 'overlay'

type AttentionResultViewProps = {
  readonly asset: WorkbenchCatalogEntry
  readonly output: InferenceOutput
  readonly roi: RoiAnnotation
  readonly layout: 'clinician-stack' | 'patient-compact'
}

const viewChoices = [
  { value: 'aoi', label: 'AOI summary' },
  { value: 'density', label: 'Density field' },
  { value: 'overlay', label: 'Overlay' },
] as const satisfies readonly { value: ResultViewMode; label: string }[]

const failureCopy = {
  EMPTY_FIELD: 'No attention-density points are available.',
  INVALID_BOUNDARY: 'The image boundary could not be verified.',
  INVALID_POINT: 'The attention-density point data could not be verified.',
  POINT_OUTSIDE_BOUNDARY:
    'The attention-density field does not match the verified image boundary.',
} as const

export function AttentionResultView({
  asset,
  output,
  roi,
  layout,
}: AttentionResultViewProps) {
  const [mode, setMode] = useState<ResultViewMode>('aoi')
  const [opacity, setOpacity] = useState(70)
  const radioName = useId()
  const overlayTitleId = useId()
  const aoiUnavailableTitleId = useId()
  const isClinicianStack = layout === 'clinician-stack'
  const connected = output.origin === 'model_prediction'
  const canUseSyntheticTemplate =
    output.origin === 'mock_simulation' &&
    output.attentionSemantics.clinicalAoi.registration ===
      'synthetic_template_v1'
  const presentation = canUseSyntheticTemplate
    ? deriveClinicalAoiPresentation(
        output.heatmap,
        output.binding.roiGeometry,
      )
    : undefined
  const boundaryCopy = connected
    ? 'Research-unvalidated predicted observer-attention density · not observed gaze'
    : 'Simulated attention-density interface structure · not observed or human gaze · not a patient prediction or result'
  const aoiUnavailableView = (
    <section
      className="attention-result-view__unavailable attention-result-view__aoi-unavailable"
      aria-labelledby={aoiUnavailableTitleId}
    >
      <h3 id={aoiUnavailableTitleId}>AOI summary unavailable</h3>
      <p>
        {connected
          ? 'Registration geometry was not supplied with this connected result.'
          : 'The fixed simulation template was not available for this result.'}
      </p>
      <p>
        {connected
          ? 'Connected AOI reporting requires registered landmarks or polygons, explicit orientation metadata, and registration quality control.'
          : 'No anatomical AOI shares were derived.'}
      </p>
    </section>
  )

  const overlayView = (
    <section
      className="attention-result-view__overlay"
      aria-labelledby={isClinicianStack ? overlayTitleId : undefined}
      aria-label={isClinicianStack ? undefined : 'Density overlay'}
    >
      {isClinicianStack ? <h3 id={overlayTitleId}>Density overlay</h3> : null}
      <AttentionMap
        asset={asset}
        output={output}
        roi={roi}
        showHeatmap
        opacity={opacity}
        showRegion={false}
      />
      <details className="attention-result-view__options">
        <summary>Display options</summary>
        <div className="attention-result-view__option-fields">
          <label htmlFor={`${radioName}-opacity`}>
            <span>Overlay opacity</span>
            <input
              id={`${radioName}-opacity`}
              type="range"
              min="20"
              max="100"
              step="5"
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
          </label>
        </div>
      </details>
    </section>
  )

  return (
    <section
      className="attention-result-view"
      data-layout={layout}
      aria-label={
        connected ? 'Research model attention result' : 'Simulated attention result'
      }
    >
      {!isClinicianStack ? (
        <p className="attention-result-view__boundary">{boundaryCopy}</p>
      ) : null}

      {presentation && !presentation.ok ? (
        <div className="attention-result-view__unavailable" role="status">
          <h3>Result view unavailable</h3>
          <p>{failureCopy[presentation.reason]}</p>
        </div>
      ) : isClinicianStack ? (
        <div className="attention-result-story">
          <AttentionSignalField asset={asset} output={output} />
          {overlayView}
          {presentation?.ok ? (
            <ClinicalAoiSummary asset={asset} presentation={presentation} />
          ) : (
            aoiUnavailableView
          )}
        </div>
      ) : (
        <>
          <fieldset className="attention-result-view__choices">
            <legend>Result view</legend>
            <div className="attention-result-view__choice-row">
              {viewChoices.map((choice) => (
                <label key={choice.value}>
                  <input
                    type="radio"
                    name={radioName}
                    value={choice.value}
                    checked={mode === choice.value}
                    onChange={() => setMode(choice.value)}
                  />
                  <span>{choice.label}</span>
                </label>
              ))}
            </div>
          </fieldset>
          {mode === 'aoi' ? (
            presentation?.ok ? (
              <ClinicalAoiSummary asset={asset} presentation={presentation} />
            ) : (
              aoiUnavailableView
            )
          ) : mode === 'density' ? (
            <AttentionSignalField asset={asset} output={output} />
          ) : (
            overlayView
          )}
        </>
      )}

      {isClinicianStack ? (
        <p className="attention-result-view__boundary">{boundaryCopy}</p>
      ) : null}

      <div className="attention-result-view__explanation">
        <p>
          {connected
            ? 'This display shows research-unvalidated predicted observer-attention density. It is not observed gaze, the current facial-paralysis severity model, a clinical result, or evidence of surgical outcome.'
            : 'This display rehearses interface structure with a simulated attention-density field. It is not observed or human gaze, a patient prediction or result, or evidence of surgical outcome.'}
        </p>
        <p>
          Colors do not mean severity, defect location, healing, emotion,
          attractiveness, or surgical success.
        </p>
      </div>
    </section>
  )
}
