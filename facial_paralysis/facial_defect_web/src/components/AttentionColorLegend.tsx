import { ATTENTION_COLOR_SCALE_LABEL } from './attentionColorScale'

type AttentionColorLegendProps = {
  readonly element?: 'div' | 'figcaption'
}

export function AttentionColorLegend({
  element: Element = 'div',
}: AttentionColorLegendProps) {
  return (
    <Element className="attention-signal-legend">
      <span>Low</span>
      <span
        aria-label={ATTENTION_COLOR_SCALE_LABEL}
        className="attention-signal-legend__scale"
        role="group"
      />
      <span>Peak</span>
    </Element>
  )
}
