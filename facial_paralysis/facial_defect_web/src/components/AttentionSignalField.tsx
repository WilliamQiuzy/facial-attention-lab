import { useId, type CSSProperties } from 'react'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'
import type { InferenceOutput } from '../workbench/types'

type AttentionSignalFieldProps = {
  asset: WorkbenchCatalogEntry
  output: InferenceOutput
}

export function AttentionSignalField({ asset, output }: AttentionSignalFieldProps) {
  const referenceNoteId = useId()
  const connected = output.origin === 'model_prediction'

  return (
    <section className="attention-separate" aria-label="Separated result view">
      <figure className="attention-separate__panel">
        <h3>Source image</h3>
        <div className="attention-separate__image-frame">
          <img
            src={asset.url}
            alt="Synthetic source face"
            width="1024"
            height="1024"
            loading="eager"
            decoding="async"
          />
        </div>
        <div
          className="patient-orientation"
          role="group"
          aria-label={connected ? 'Viewer orientation' : 'Patient orientation'}
        >
          <span>
            {connected ? 'Viewer left' : 'Patient right (viewer left)'}
          </span>
          <span>
            {connected ? 'Viewer right' : 'Patient left (viewer right)'}
          </span>
          {!connected ? (
            <small>Frontal, non-mirrored synthetic display.</small>
          ) : null}
        </div>
        <figcaption>AI-generated synthetic source image</figcaption>
      </figure>

      <figure className="attention-separate__panel">
        <h3>
          {connected
            ? 'Predicted observer-attention density'
            : 'Simulated attention-density field'}
        </h3>
        <div
          className="attention-signal-field"
          role="region"
          aria-describedby={referenceNoteId}
          aria-label={
            connected
              ? 'Predicted observer-attention density field'
              : 'Simulated attention-density field'
          }
        >
          {output.heatmap.map((point, index) => (
            <span
              aria-hidden="true"
              className="attention-signal-field__point"
              key={`${point.x}-${point.y}-${index}`}
              style={
                {
                  '--signal-x': `${point.x * 100}%`,
                  '--signal-y': `${point.y * 100}%`,
                  '--signal-size': `${point.radius * 200}%`,
                  '--signal-strength': point.intensity,
                } as CSSProperties
              }
            />
          ))}
          <span className="attention-signal-field__watermark">
            {connected ? 'MODEL PREDICTION' : 'SIMULATED'}
          </span>
        </div>
        <figcaption className="attention-signal-legend">
          <span>Less relative density</span>
          <span aria-hidden="true" className="attention-signal-legend__scale" />
          <span>More relative density</span>
        </figcaption>
        <p
          className="attention-signal-field__reference-note"
          id={referenceNoteId}
        >
          Face outline unavailable: registered contour or landmarks
          were not supplied with this result.
        </p>
      </figure>
    </section>
  )
}
