import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const css = readFileSync('src/styles/patient-workflow.css', 'utf8')

function rule(selector: string): string {
  const start = css.indexOf(selector)
  if (start < 0) return ''
  const end = css.indexOf('}', start)
  return css.slice(start, end + 1)
}

describe('patient workflow visual safeguards', () => {
  it('keeps replacement capture actions secondary when a preview exists', () => {
    const previewAction = rule(
      '.capture-panel--has-preview .capture-panel__file-action:first-child',
    )

    expect(previewAction).toMatch(/color:\s*#004f9f/)
    expect(previewAction).toMatch(/background:\s*#fff/)
  })

  it('uses a smooth solid-blue attention field and lets density follow the source ratio', () => {
    const attentionPoint = rule('.patient-attention-point')
    const densityPoint = rule(
      '.patient-attention-images__density-layer .patient-attention-point',
    )
    const densityPlane = rule(
      '.patient-attention-images__density-plane',
    )

    expect(attentionPoint).toMatch(/background:\s*rgb\(/)
    expect(attentionPoint).not.toMatch(/gradient/i)
    expect(attentionPoint).toMatch(/filter:\s*blur\(/)
    expect(rule('.patient-attention-point::after')).toBe('')
    expect(densityPoint).toMatch(/background:\s*rgb\(/)
    expect(densityPoint).toMatch(/filter:\s*blur\(/)
    expect(densityPlane).not.toContain('aspect-ratio')
  })

  it('keeps the photo-matched face contour above density points and below the simulation watermark', () => {
    const contour = rule(
      '.patient-attention-images__density-plane .patient-face-contour',
    )
    expect(contour).toMatch(/z-index:\s*2/)
    expect(contour).toMatch(/pointer-events:\s*none/)
    expect(rule('.patient-face-contour__path')).toMatch(
      /vector-effect:\s*non-scaling-stroke/,
    )
    expect(
      rule('.patient-attention-images__density-layer'),
    ).toMatch(/z-index:\s*1/)
    expect(
      rule('.patient-attention-images__watermark'),
    ).toMatch(/z-index:\s*3/)
  })

  it('styles the unavailable demo note and preserves a 44px attestation target', () => {
    expect(rule('.capture-panel__synthetic-unavailable')).toMatch(
      /color:\s*#435764/,
    )
    expect(rule('.patient-attestation label')).toMatch(
      /min-height:\s*44px/,
    )
  })

  it('stacks the new-visit header at the tablet breakpoint', () => {
    const tabletStart = css.indexOf('@media (max-width: 820px)')
    const phoneStart = css.indexOf('@media (max-width: 599px)')
    const tabletCss = css.slice(tabletStart, phoneStart)

    expect(tabletCss).toMatch(
      /\.patient-visit-create-header\s*\{[^}]*display:\s*grid[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s,
    )
  })

  it('allows long patient and review identifiers to wrap inside their grids', () => {
    expect(css).toMatch(
      /\.patient-list__identity,\s*\.patient-list__visit,\s*\.patient-identity__title,\s*\.clinical-review-queue__patient,\s*\.clinical-review-queue__visit\s*\{[^}]*min-width:\s*0/s,
    )
    expect(css).toMatch(
      /\.patient-list__identity h3,\s*\.patient-list__identity p,\s*\.patient-list__visit p,\s*\.patient-identity__title h1,\s*\.patient-identity__title h2,\s*\.clinical-review-queue__patient strong,\s*\.clinical-review-queue__patient span\s*\{[^}]*overflow-wrap:\s*anywhere/s,
    )
  })

  it('uses restrained loading motion and disables it when reduced motion is requested', () => {
    expect(rule('.patient-loading-icon')).toMatch(
      /animation:\s*patient-loading-spin/,
    )
    expect(css).not.toMatch(/transition:\s*all/)

    const reducedMotionStart = css.indexOf(
      '@media (prefers-reduced-motion: reduce)',
    )
    expect(reducedMotionStart).toBeGreaterThan(-1)
    expect(css.slice(reducedMotionStart)).toMatch(
      /\.patient-loading-icon\s*\{[^}]*animation:\s*none/s,
    )
  })

  it('places photograph and quality checks side by side only when space allows', () => {
    expect(rule('.patient-capture-quality-step')).toMatch(
      /grid-template-columns:\s*minmax\(0,\s*1\.15fr\)\s+minmax\(320px,\s*0\.85fr\)/,
    )

    const tabletStart = css.indexOf('@media (max-width: 820px)')
    const phoneStart = css.indexOf('@media (max-width: 599px)')
    const tabletCss = css.slice(tabletStart, phoneStart)
    expect(tabletCss).toMatch(
      /\.patient-capture-quality-step\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s,
    )
  })

  it('keeps the review textarea and selected decision visibly consistent', () => {
    expect(rule('.patient-result-review__note textarea')).toMatch(
      /border:\s*1px solid #7f909c/,
    )
    expect(
      rule(
        ".patient-result-review__form fieldset label:has(input:checked)",
      ),
    ).toMatch(/background:\s*#eef6fc/)
  })

  it('keeps programmatic workflow focus visible without a full-width dashed outline', () => {
    expect(
      rule(".patient-workflow-page h2[tabindex='-1']"),
    ).toMatch(/width:\s*fit-content/)
    expect(css).toMatch(
      /\.patient-workflow-page h2\[tabindex='-1'\]:focus-visible\s*\{[^}]*outline:\s*0[^}]*box-shadow:\s*0 3px 0 #0057b8/s,
    )
    expect(css).toMatch(
      /\.patient-job-progress\[tabindex='-1'\]:focus-visible\s*\{[^}]*outline:\s*0[^}]*box-shadow:/s,
    )
  })
})
