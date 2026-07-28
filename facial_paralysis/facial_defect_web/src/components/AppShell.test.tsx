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
})
