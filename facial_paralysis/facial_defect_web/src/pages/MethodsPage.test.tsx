import { render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import type { WorkbenchGateway } from '../workbench/WorkbenchGateway'

const pagesCss = readFileSync(resolve('src/styles/pages.css'), 'utf8')

function cssRule(selector: string): string {
  const escapedSelector = selector.replace(
    /[.*+?^${}()|[\]\\]/g,
    '\\$&',
  )
  return (
    pagesCss.match(
      new RegExp(`${escapedSelector}\\s*\\{([^}]+)\\}`),
    )?.[1] ?? ''
  )
}

describe('methods and safeguards', () => {
  it('separates the future attention-density contract from current facial-paralysis evidence', () => {
    render(
      <MemoryRouter initialEntries={['/methods']}>
        <App />
      </MemoryRouter>,
    )

    const page = within(screen.getByRole('main'))
    expect(page.getByRole('heading', { name: /methods, provenance & safeguards/i })).toBeVisible()
    expect(
      page.getByText(
        /no trained facial-defect attention model exists in this repository yet/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(/population-level predicted observer-attention spatial density/i),
    ).toBeVisible()
    expect(
      page.getByText(/fixed-template simulated point-weight shares/i),
    ).toBeVisible()
    expect(
      page.getByText(
        /each simulated point's intensity is assigned by its center.*radius is ignored.*not a raster or density-kernel integral/i,
      ),
    ).toBeVisible()
    expect(page.getByText(/patient left appears on the viewer's right/i)).toBeVisible()
    expect(
      page.getByText(/10 hash-pinned, standalone AI-generated synthetic cases/i),
    ).toBeVisible()
    expect(page.getByText(/single WorkbenchGateway port/i)).toBeVisible()
    expect(page.getByText(/local mock gateway active/i)).toBeVisible()
    expect(page.getByText(/default mode performs no inference network request/i)).toBeVisible()
    expect(
      page.getByText(/palsy probability and eyes\/mouth ordinal outputs/i),
    ).toBeVisible()
    expect(
      page.getByText(
        /landmark-derived left-right asymmetry, eye-closure dynamics, and a Mayo FACES label-free research measurement summary/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /No checked-in checkpoint includes an HB task; the architecture can support one, but Mayo HB calibration has not started/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /not a validated eFACE, Sunnybrook, or HB composite or grade/i,
      ),
    ).toBeVisible()
    expect(page.queryByText(/clinical scorecard/i)).not.toBeInTheDocument()
    expect(
      page.getByText(
        /severity or ordinal payload without spatial points fails closed/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /mock AOIs are automatic post-inference summaries and do not modify the simulation/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /surgical-site mask would be a separate, versioned contextual annotation/i,
      ),
    ).toBeVisible()
    expect(page.getByText(/attention is not emotion, judgment, stigma/i)).toBeVisible()
    expect(page.getByText(/IRB and protocol decisions remain institutional gates/i)).toBeVisible()
    expect(page.queryByText(/two hash-pinned/i)).not.toBeInTheDocument()
    expect(
      page.queryByText(
        /current inference results instead expose four bounded interface metrics/i,
      ),
    ).not.toBeInTheDocument()
    expect(
      page.getByText(
        /connected version 1 is a synthetic spatial contract rehearsal/i,
      ),
    ).toBeVisible()
    expect(page.getByText('registration_geometry_unavailable_v1')).toBeVisible()
    expect(
      page.getByText(
        /does not carry landmarks or polygons, source dimensions, orientation or mirror metadata, or registration quality control/i,
      ),
    ).toBeVisible()
    expect(
      page.getByText(
        /connected AOI reporting remains unavailable and fails closed until the contract is extended/i,
      ),
    ).toBeVisible()
    expect(
      page.queryByText(/model-supplied face registration/i),
    ).not.toBeInTheDocument()
    expect(
      page.getByText(/v2 attention checkpoint uses temporal frame pooling/i),
    ).toBeVisible()
    expect(
      page.getByText((_content, element) => {
        return (
          element?.tagName === 'P' &&
          /HeatmapPoint\[\].*patient-media reference are provisional/i.test(
            element.textContent ?? '',
          )
        )
      }),
    ).toBeVisible()
  })

  it('reports connected mode as an explicit research opt-in without implying fallback', () => {
    const runInference = vi.fn()
    const gateway = {
      mode: 'connected',
      runInference,
    } satisfies WorkbenchGateway
    render(
      <MemoryRouter initialEntries={['/methods']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    const page = within(screen.getByRole('main'))
    const runtime = page.getByRole('status', { name: 'Methods runtime boundary' })
    expect(
      within(runtime).getByText(/synthetic spatial contract rehearsal enabled/i),
    ).toBeVisible()
    expect(within(runtime).getByText(/explicit opt-in/i)).toBeVisible()
    expect(within(runtime).getByText(/never falls back to the mock engine/i)).toBeVisible()
    expect(
      within(runtime).getByText(
        /current facial_paralysis functional-assessment system remains separate and is not connected/i,
      ),
    ).toBeVisible()
    expect(page.getByText('model_prediction')).toBeVisible()
    expect(page.getByText('research_unvalidated')).toBeVisible()
    expect(page.queryByText(/local mock gateway active/i)).not.toBeInTheDocument()
    expect(runInference).not.toHaveBeenCalled()
  })

  it('presents long research references with compact blue-white styling', () => {
    expect(cssRule('.model-hero')).toMatch(/background:\s*#fff/)
    expect(cssRule('.model-hero')).not.toMatch(/gradient/i)
    expect(cssRule('.model-hero h1')).toMatch(
      /font-size:\s*clamp\(2rem,\s*3vw,\s*2\.6rem\)/,
    )
    expect(cssRule('.model-section')).toMatch(
      /padding-block:\s*48px/,
    )
    expect(cssRule('.requirements-section')).toMatch(
      /background:\s*#f7f9fb/,
    )
    expect(cssRule('.methods-hero')).toMatch(
      /padding-block:\s*36px/,
    )
    expect(cssRule('.methods-hero h1')).toMatch(
      /font-size:\s*clamp\(2rem,\s*3vw,\s*2\.6rem\)/,
    )
    expect(cssRule('.methods-layout')).toMatch(
      /padding-block:\s*48px 64px/,
    )
    expect(pagesCss).toMatch(
      /\.model-page h1,\s*\.model-page h2,\s*\.model-page h3,\s*\.methods-page h1,\s*\.methods-page h2,\s*\.methods-page h3\s*\{[^}]*font-family:\s*var\(--research-font-sans\)/s,
    )
  })
})
