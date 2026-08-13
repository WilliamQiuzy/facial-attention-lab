import type { CSSProperties } from 'react'
import type { WorkbenchCatalogEntry } from '../workbench/catalog'
import type { InferenceOutput, RoiAnnotation } from '../workbench/types'
import { AttentionColorLegend } from './AttentionColorLegend'
import { attentionColorRgb } from './attentionColorScale'

type SharedAttentionMapProps = {
  showHeatmap: boolean
  opacity: number
  showRegion: boolean
}

type AttentionMapProps = SharedAttentionMapProps & {
  asset: WorkbenchCatalogEntry
  output: InferenceOutput
  roi: RoiAnnotation
}

type RenderableAttentionMap = {
  readonly assetId: string
  readonly label: string
  readonly imageUrl: string
  readonly disclosure: string
  readonly watermark: string
  readonly points: readonly {
    readonly x: number
    readonly y: number
    readonly radius: number
    readonly intensity: number
  }[]
  readonly region: {
    readonly x: number
    readonly y: number
    readonly width: number
    readonly height: number
  }
}

function renderableMap(props: AttentionMapProps): RenderableAttentionMap {
  return {
    assetId: props.asset.id,
    label: props.asset.label,
    imageUrl: props.asset.url,
    disclosure: props.asset.disclosure,
    watermark: props.output.watermark,
    points: props.output.heatmap.map((point) => ({
      x: point.x * 100,
      y: point.y * 100,
      radius: point.radius * 100,
      intensity: point.intensity,
    })),
    region: {
      x: props.roi.geometry.x * 100,
      y: props.roi.geometry.y * 100,
      width: props.roi.geometry.width * 100,
      height: props.roi.geometry.height * 100,
    },
  }
}

export function AttentionMap(props: AttentionMapProps) {
  const { showHeatmap, opacity, showRegion } = props
  const map = renderableMap(props)
  const connected = props.output.origin === 'model_prediction'

  return (
    <figure className="attention-card">
      <div className="attention-card__header">
        <div>
          <p className="attention-card__label">{map.label}</p>
          <p className="attention-card__id">Asset {map.assetId}</p>
        </div>
        <span className="source-chip">AI generated</span>
      </div>
      <div className="attention-frame">
        <img
          src={map.imageUrl}
          alt={`${map.label}: AI-generated synthetic face`}
          width="1024"
          height="1024"
          loading="eager"
          decoding="async"
          fetchPriority="high"
        />
        {showHeatmap ? (
          <div
            className="heatmap-layer"
            style={{ '--heatmap-opacity': opacity / 100 } as CSSProperties}
            aria-hidden="true"
          >
            {map.points.map((point, index) => (
              <span
                className="heatmap-point"
                key={`${point.x}-${point.y}-${index}`}
                style={
                  {
                    '--point-x': `${point.x}%`,
                    '--point-y': `${point.y}%`,
                    '--point-size': `${point.radius * 2}%`,
                    '--point-intensity': point.intensity,
                    '--attention-color-rgb': attentionColorRgb(point.intensity),
                  } as CSSProperties
                }
              />
            ))}
          </div>
        ) : null}
        <span className="heatmap-watermark">{map.watermark}</span>
        {showRegion ? (
          <div
            className="roi-box"
            style={
              {
                left: `${map.region.x}%`,
                top: `${map.region.y}%`,
                width: `${map.region.width}%`,
                height: `${map.region.height}%`,
              } as CSSProperties
            }
          >
            <span>Illustrative region</span>
          </div>
        ) : null}
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
      <AttentionColorLegend />
      <figcaption>{map.disclosure}</figcaption>
    </figure>
  )
}
