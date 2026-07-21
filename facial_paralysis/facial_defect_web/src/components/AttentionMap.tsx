import type { CSSProperties } from 'react'
import type { AttentionResult } from '../model/types'

type AttentionMapProps = {
  result: AttentionResult
  showHeatmap: boolean
  opacity: number
  showRegion: boolean
  watermark: string
}

export function AttentionMap({
  result,
  showHeatmap,
  opacity,
  showRegion,
  watermark,
}: AttentionMapProps) {
  return (
    <figure className="attention-card">
      <div className="attention-card__header">
        <div>
          <p className="attention-card__label">{result.label}</p>
          <p className="attention-card__id">Asset {result.assetId}</p>
        </div>
        <span className="source-chip">AI generated</span>
      </div>
      <div className="attention-frame">
        <img src={result.imageUrl} alt={`${result.label}: AI-generated synthetic face`} />
        {showHeatmap ? (
          <>
            <div
              className="heatmap-layer"
              style={{ '--heatmap-opacity': opacity / 100 } as CSSProperties}
              aria-hidden="true"
            >
              {result.heatmapPoints.map((point, index) => (
                <span
                  className="heatmap-point"
                  key={`${point.x}-${point.y}-${index}`}
                  style={
                    {
                      '--point-x': `${point.x}%`,
                      '--point-y': `${point.y}%`,
                      '--point-size': `${point.radius * 2}%`,
                      '--point-intensity': point.intensity,
                    } as CSSProperties
                  }
                />
              ))}
            </div>
            <span className="heatmap-watermark">{watermark}</span>
          </>
        ) : null}
        {showRegion ? (
          <div
            className="roi-box"
            style={
              {
                left: `${result.regionOfInterest.x}%`,
                top: `${result.regionOfInterest.y}%`,
                width: `${result.regionOfInterest.width}%`,
                height: `${result.regionOfInterest.height}%`,
              } as CSSProperties
            }
          >
            <span>Illustrative region</span>
          </div>
        ) : null}
      </div>
      <figcaption>{result.disclosure}</figcaption>
    </figure>
  )
}
