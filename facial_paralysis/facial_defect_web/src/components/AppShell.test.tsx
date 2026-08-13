import { fireEvent, render, screen, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'
import type { WorkbenchGateway } from '../workbench/WorkbenchGateway'
import { listWorkbenchAssets } from '../workbench/catalog'

const approvedCaseId = listWorkbenchAssets()[2].id
const globalCss = readFileSync(resolve('src/styles/global.css'), 'utf8')
const workbenchCss = readFileSync(resolve('src/styles/workbench.css'), 'utf8')
const task5Css = readFileSync(resolve('src/styles/task5.css'), 'utf8')
const patientWorkflowCss = readFileSync(
  resolve('src/styles/patient-workflow.css'),
  'utf8',
)
const indexHtml = readFileSync(resolve('index.html'), 'utf8')

const connectedGateway = {
  mode: 'connected',
  runInference: async () => {
    throw new Error('Not invoked while rendering the shell.')
  },
} satisfies WorkbenchGateway

function renderShell(path: string, gateway?: WorkbenchGateway) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App gateway={gateway} />
    </MemoryRouter>,
  )
}

function expectPermanentBoundaries() {
  const environmentNotices = screen.getAllByRole('status', {
    name: 'Workspace environment',
  })
  expect(environmentNotices).toHaveLength(1)

  const environment = environmentNotices[0]
  expect(
    within(environment).getByText(
      'Research prototype · synthetic/test records only · session data resets on refresh · clinical use blocked',
    ),
  ).toBeVisible()
  expect(
    within(environment).getByText(
      'Research prototype · synthetic/test only · clinical use blocked',
    ),
  ).toBeInTheDocument()
  expect(screen.queryByRole('status', { name: /research use status/i })).not.toBeInTheDocument()
}

function getCssRule(source: string, selector: string) {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = source.match(new RegExp(`${escapedSelector}\\s*\\{([^}]+)\\}`))
  expect(match, `Expected CSS rule for ${selector}`).not.toBeNull()
  return match![1]
}

describe('application frame', () => {
  it.each([
    '/',
    '/patients',
    `/analysis?case=${approvedCaseId}`,
    '/research/reviews/review-from-old-session',
  ])('keeps one compact mock environment notice on %s', (path) => {
    renderShell(path)

    expectPermanentBoundaries()
  })

  it.each(['/runs', '/models', '/reviews'])(
    'keeps one compact connected environment notice on %s',
    (path) => {
      renderShell(path, connectedGateway)

      expectPermanentBoundaries()
    },
  )

  it('keeps operational navigation and current location visible', async () => {
    renderShell(`/analysis?case=${approvedCaseId}`)

    const primary = screen.getByRole('navigation', { name: /primary navigation/i })
    expect(within(primary).getAllByRole('link').map((link) => link.textContent)).toEqual([
      'Patients',
      'Reviews',
      'Help',
    ])
    expect(screen.getByRole('main')).toHaveAttribute('id', 'main-content')
    expect(screen.getByRole('main')).toHaveAttribute('tabindex', '-1')
    expect(
      screen.getByRole('heading', { name: 'Simulated observer-attention density' }),
    ).toBeVisible()
  })

  it('does not use an official Mayo logo or trademark lockup', () => {
    const { container } = renderShell('/')

    expect(container.querySelector('img[alt*="Mayo" i]')).not.toBeInTheDocument()
    expect(screen.queryByText(/Mayo-inspired research operations/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/affiliation or endorsement/i)).not.toBeInTheDocument()
    expect(screen.getByText('FR')).toBeVisible()
    expect(screen.getByRole('link', { name: 'Facial Reconstruction Imaging' })).toHaveAttribute(
      'href',
      '/patients',
    )
  })

  it('gives the responsive navigation toggle an explicit changing name and closes after selection', () => {
    const { container } = renderShell('/')

    const menuButton = container.querySelector<HTMLButtonElement>('.menu-button')
    expect(menuButton).not.toBeNull()
    expect(menuButton).toHaveAttribute('aria-label', 'Open navigation menu')
    expect(menuButton).toHaveAttribute('aria-expanded', 'false')

    fireEvent.click(menuButton!)

    expect(menuButton).toHaveAttribute('aria-label', 'Close navigation menu')
    expect(menuButton).toHaveAttribute('aria-expanded', 'true')

    const primary = screen.getByRole('navigation', { name: 'Primary navigation' })
    fireEvent.click(within(primary).getByRole('link', { name: 'Help' }))

    expect(menuButton).toHaveAttribute('aria-label', 'Open navigation menu')
    expect(menuButton).toHaveAttribute('aria-expanded', 'false')
    expect(within(primary).getByRole('link', { name: 'Help' })).toHaveAttribute(
      'aria-current',
      'page',
    )
  })

  it('closes the responsive navigation after any in-app route change', () => {
    const { container } = renderShell('/patients')
    const menuButton = container.querySelector<HTMLButtonElement>('.menu-button')

    fireEvent.click(menuButton!)
    expect(menuButton).toHaveAttribute('aria-expanded', 'true')

    fireEvent.click(screen.getByRole('link', { name: 'New patient' }))

    expect(screen.getByRole('heading', { name: 'New patient' })).toBeVisible()
    expect(menuButton).toHaveAttribute('aria-label', 'Open navigation menu')
    expect(menuButton).toHaveAttribute('aria-expanded', 'false')
  })

  it.each([
    ['/patients/patient-demo-001', 'Patients'],
  ])('keeps %s nested under the active %s workflow', (path, activeLabel) => {
    renderShell(path)

    const primary = screen.getByRole('navigation', { name: 'Primary navigation' })
    const activeLink = within(primary).getByRole('link', { name: activeLabel })
    expect(activeLink).toHaveAttribute('aria-current', 'page')

    for (const link of within(primary).getAllByRole('link')) {
      if (link !== activeLink) {
        expect(link).not.toHaveAttribute('aria-current')
      }
    }
  })

  it('does not mark the clinical Reviews navigation active on a research review route', () => {
    renderShell('/research/reviews/review-from-this-session')

    const primary = screen.getByRole('navigation', { name: 'Primary navigation' })
    for (const link of within(primary).getAllByRole('link')) {
      expect(link).not.toHaveAttribute('aria-current')
    }
  })

  it('keeps a compact footer with one Help route', () => {
    renderShell('/')

    const footer = screen.getByRole('contentinfo')
    expect(
      within(footer).getByText(
        'Facial Reconstruction Imaging · Research prototype',
      ),
    ).toBeVisible()
    expect(
      within(footer).getByText(
        'Synthetic/test records only · Session resets on refresh',
      ),
    ).toBeVisible()

    const resources = within(footer).getByRole('navigation', {
      name: 'Resource navigation',
    })
    expect(within(resources).getAllByRole('link').map((link) => link.textContent)).toEqual([
      'Help',
    ])
    expect(within(footer).queryByText(/permanent boundary/i)).not.toBeInTheDocument()
  })

  it('keeps permanent shell text at a clinician-readable minimum size', () => {
    expect(getCssRule(workbenchCss, '.environment-strip')).toMatch(
      /font-size:\s*(?:0\.875rem|14px)/,
    )
    expect(getCssRule(globalCss, '.site-footer a')).toMatch(
      /font-size:\s*(?:0\.875rem|14px)/,
    )
    expect(getCssRule(globalCss, '.site-footer__brand')).toMatch(
      /font-size:\s*(?:0\.875rem|14px)/,
    )
    expect(getCssRule(globalCss, '.site-footer__boundary')).toMatch(
      /font-size:\s*(?:0\.875rem|14px)/,
    )
  })

  it('keeps three primary links visible on tablets and uses 52px targets only on narrow screens', () => {
    const tabletStart = globalCss.indexOf('@media (max-width: 1199px)')
    const tabletEnd = globalCss.indexOf('@media (max-width: 904px)', tabletStart)
    const tabletCss = globalCss.slice(tabletStart, tabletEnd)
    const mobileStart = globalCss.indexOf('@media (max-width: 599px)')
    const mobileEnd = globalCss.indexOf('@media print', mobileStart)
    const mobileCss = globalCss.slice(mobileStart, mobileEnd)
    expect(tabletCss).not.toMatch(/\.menu-button\s*\{[^}]*display:\s*inline-flex/)
    expect(tabletCss).not.toMatch(/\.primary-nav\s*\{[^}]*display:\s*none/)
    expect(getCssRule(mobileCss, '.menu-button')).toMatch(/display:\s*inline-flex/)
    expect(getCssRule(mobileCss, '.primary-nav')).toMatch(/display:\s*none/)
    expect(getCssRule(mobileCss, '.primary-nav a')).toMatch(/min-height:\s*52px/)
    expect(getCssRule(mobileCss, '.primary-nav a')).toMatch(/padding:\s*0/)
    expect(workbenchCss).not.toMatch(/\.workspace-header\s+\.primary-nav\s+a\s*\{/)
  })

  it('keeps the 320px header compact without dropping the research boundary', () => {
    const mobileStart = globalCss.indexOf('@media (max-width: 599px)')
    const mobileEnd = globalCss.indexOf('@media print', mobileStart)
    const mobileCss = globalCss.slice(mobileStart, mobileEnd)
    const narrowMobileStart = globalCss.indexOf('@media (max-width: 359px)')
    const narrowMobileCss = globalCss.slice(narrowMobileStart)
    const workbenchMobileStart = workbenchCss.indexOf('@media (max-width: 599px)')
    const workbenchMobileEnd = workbenchCss.indexOf(
      '@media (max-width: 599px)',
      workbenchMobileStart + 1,
    )
    const workbenchMobileCss = workbenchCss.slice(
      workbenchMobileStart,
      workbenchMobileEnd,
    )

    expect(getCssRule(mobileCss, '.site-header__inner')).toMatch(/gap:\s*8px/)
    expect(getCssRule(narrowMobileCss, '.brand__name--full')).toMatch(
      /display:\s*none/,
    )
    expect(getCssRule(narrowMobileCss, '.brand__name--compact')).toMatch(
      /display:\s*block/,
    )
    expect(getCssRule(workbenchMobileCss, '.environment-strip__copy--full')).toMatch(
      /display:\s*none/,
    )
    expect(getCssRule(workbenchMobileCss, '.environment-strip__copy--compact')).toMatch(
      /display:\s*inline/,
    )
  })

  it('removes the obsolete notice CSS and prints the environment strip intentionally', () => {
    expect(globalCss).not.toContain('.research-notice')

    const printStart = workbenchCss.lastIndexOf('@media print')
    const printCss = workbenchCss.slice(printStart)
    const printedEnvironment = getCssRule(printCss, '.environment-strip')

    expect(printedEnvironment).toMatch(/background:\s*#fff/)
    expect(printedEnvironment).toMatch(/border-(?:top|bottom):/)
  })

  it('uses a plain header and removes the unused marketing decoration', () => {
    expect(globalCss).not.toContain('backdrop-filter')
    expect(globalCss).not.toContain('.hero__orb')
    expect(globalCss).not.toContain('.path-card')
    expect(globalCss).not.toContain('.readiness-band')
  })

  it('allows the page body to shrink within a 320px viewport with a scrollbar', () => {
    const bodyRule = getCssRule(globalCss, 'body')

    expect(bodyRule).toMatch(/min-width:\s*0/)
    expect(bodyRule).not.toMatch(/min-width:\s*320px/)
  })

  it('uses expected pointer affordances and a white browser chrome color', () => {
    expect(getCssRule(workbenchCss, '.workspace-button')).toMatch(
      /cursor:\s*pointer/,
    )
    expect(getCssRule(workbenchCss, '.workspace-button')).toMatch(
      /touch-action:\s*manipulation/,
    )
    expect(workbenchCss).toMatch(
      /\.workspace-button:not\(:disabled\):active\s*\{[^}]*transform:\s*translateY\(1px\)/s,
    )
    expect(getCssRule(workbenchCss, '.status-badge')).toMatch(
      /font-size:\s*0\.875rem/,
    )
    expect(patientWorkflowCss).toMatch(
      /\.patient-primary-action,\s*\.patient-secondary-action,\s*\.patient-link-action\s*\{[^}]*touch-action:\s*manipulation/s,
    )
    expect(patientWorkflowCss).toMatch(
      /\.patient-(?:primary|secondary|link)-action:not\(:disabled\):active/s,
    )
    expect(task5Css).toMatch(
      /\.task5-case-checklist__row > a\s*\{[^}]*display:\s*inline-flex[^}]*min-height:\s*44px[^}]*align-items:\s*center/s,
    )
    expect(task5Css).toMatch(
      /\.task5-exclusions a\s*\{[^}]*display:\s*inline-flex[^}]*min-height:\s*44px[^}]*align-items:\s*center/s,
    )
    expect(patientWorkflowCss).toMatch(
      /\.patient-workflow-page h2\[tabindex='-1'\],[^{]+\.patient-job-progress\[tabindex='-1'\]\s*\{[^}]*scroll-margin-top:\s*96px/s,
    )
    expect(patientWorkflowCss).toMatch(
      /@media \(max-width:\s*599px\)[\s\S]*\.capture-panel__preview img\s*\{[^}]*max-height:\s*320px/s,
    )
    expect(indexHtml).toMatch(
      /<meta\s+name="theme-color"\s+content="#ffffff"\s*\/>/,
    )
    expect(workbenchCss).toMatch(
      /\.workspace-page__header h1,[^{]+\.workspace-empty h3\s*\{[^}]*font-family:\s*var\(--research-font-sans\)/s,
    )
    expect(workbenchCss).toMatch(
      /\.attention-result-section__heading h3,[^{]+\.attention-result-view__unavailable h3\s*\{[^}]*font-family:\s*var\(--research-font-sans\)/s,
    )
    expect(workbenchCss).toMatch(
      /\.inference-visual > \.workspace-panel__heading h2\[tabindex='-1'\]:focus\s*\{[^}]*outline:\s*0[^}]*box-shadow:/s,
    )
    expect(workbenchCss).toMatch(
      /\.case-filters > summary::after\s*\{[^}]*content:\s*''/s,
    )
    expect(patientWorkflowCss).toMatch(
      /\.patient-aoi-summary__method summary::after\s*\{[^}]*content:\s*''/s,
    )
    expect(task5Css).toMatch(
      /\.task5-page h1,\s*\.task5-page h2,\s*\.task5-page h3\s*\{[^}]*font-family:\s*var\(--research-font-sans\)/s,
    )
    expect(workbenchCss).toMatch(
      /@media \(prefers-reduced-motion: reduce\)\s*\{[^}]*\.workspace-loading-icon\s*\{[^}]*animation:\s*none/s,
    )
  })
})
