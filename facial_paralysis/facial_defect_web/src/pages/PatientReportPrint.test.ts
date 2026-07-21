import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'

function printBlock(filePath: string): string {
  const css = readFileSync(filePath, 'utf8')
  return css.slice(css.lastIndexOf('@media print'))
}

describe('patient print safety', () => {
  it('keeps the research warning and interpretation limits in the printout', () => {
    const globalPrintCss = printBlock(path.resolve('src/styles/global.css'))
    const patientPrintCss = printBlock(path.resolve('src/styles/pages.css'))

    expect(globalPrintCss).not.toMatch(/\.research-notice,\s*\.site-header/)
    expect(globalPrintCss).toMatch(/\.research-notice\s*{/)
    expect(patientPrintCss).not.toMatch(/\.patient-toolbar,\s*\.patient-meaning-band/)
    expect(patientPrintCss).toMatch(/\.patient-meaning-band\s*{/)
    expect(patientPrintCss).toMatch(/\.patient-safety-card/)
  })
})
