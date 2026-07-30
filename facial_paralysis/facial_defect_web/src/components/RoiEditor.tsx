import type { CSSProperties } from 'react'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'
import type { NormalizedRoi } from '../workbench/types'

type RoiCoordinate = keyof NormalizedRoi

type RoiEditorProps = {
  readonly asset: WorkbenchCatalogEntry
  readonly geometry: NormalizedRoi
  readonly disabled: boolean
  readonly onChange: (geometry: NormalizedRoi) => void
}

const CONTROL_LABELS: Readonly<Record<RoiCoordinate, string>> = {
  x: 'ROI x origin',
  y: 'ROI y origin',
  width: 'ROI width',
  height: 'ROI height',
}

function roundNormalized(value: number): number {
  return Number(Math.min(1, Math.max(0, value)).toFixed(2))
}

function boundGeometry(
  geometry: NormalizedRoi,
  coordinate: RoiCoordinate,
  rawValue: number,
): NormalizedRoi {
  const value = roundNormalized(rawValue)
  const next = { ...geometry, [coordinate]: value }

  if (coordinate === 'x') {
    next.x = Math.min(value, 0.99)
    next.width = roundNormalized(Math.max(0.01, Math.min(next.width, 1 - next.x)))
  } else if (coordinate === 'y') {
    next.y = Math.min(value, 0.99)
    next.height = roundNormalized(Math.max(0.01, Math.min(next.height, 1 - next.y)))
  } else if (coordinate === 'width') {
    next.width = roundNormalized(Math.max(0.01, Math.min(value, 1 - next.x)))
  } else {
    next.height = roundNormalized(Math.max(0.01, Math.min(value, 1 - next.y)))
  }

  return next
}

function controlMaximum(geometry: NormalizedRoi, coordinate: RoiCoordinate): number {
  if (coordinate === 'width') return roundNormalized(1 - geometry.x)
  if (coordinate === 'height') return roundNormalized(1 - geometry.y)
  return 0.99
}

export function RoiEditor({ asset, geometry, disabled, onChange }: RoiEditorProps) {
  return (
    <div className="roi-editor">
      <figure className="roi-editor__canvas">
        <img
          src={asset.url}
          alt={`${asset.label}: AI-generated synthetic face`}
          width="1024"
          height="1024"
          loading="eager"
          decoding="async"
          fetchPriority="high"
        />
        <div
          className="roi-editor__box"
          aria-hidden="true"
          style={
            {
              left: `${geometry.x * 100}%`,
              top: `${geometry.y * 100}%`,
              width: `${geometry.width * 100}%`,
              height: `${geometry.height * 100}%`,
            } as CSSProperties
          }
        >
          <span>ROI</span>
        </div>
        <figcaption>
          AI-generated synthetic · independent identity · unpaired
        </figcaption>
      </figure>

      <fieldset className="roi-editor__controls" disabled={disabled}>
        <legend>Normalized ROI geometry</legend>
        {(Object.keys(CONTROL_LABELS) as RoiCoordinate[]).map((coordinate) => {
          const minimum = coordinate === 'width' || coordinate === 'height' ? 0.01 : 0
          return (
            <label key={coordinate}>
              <span>{CONTROL_LABELS[coordinate]}</span>
              <input
                aria-label={CONTROL_LABELS[coordinate]}
                type="range"
                name={`roi-${coordinate}`}
                min={minimum}
                max={controlMaximum(geometry, coordinate)}
                step="0.01"
                value={geometry[coordinate]}
                onChange={(event) =>
                  onChange(
                    boundGeometry(
                      geometry,
                      coordinate,
                      Number(event.currentTarget.value),
                    ),
                  )
                }
              />
              <output>{geometry[coordinate].toFixed(2)}</output>
            </label>
          )
        })}
        <p>
          Values are normalized from 0.00 to 1.00 and always remain contained in
          the selected image.
        </p>
      </fieldset>
    </div>
  )
}
