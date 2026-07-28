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
})
