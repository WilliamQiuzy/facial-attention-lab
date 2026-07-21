import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Link, Route, Routes } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ScrollToTop } from './ScrollToTop'

afterEach(() => {
  vi.restoreAllMocks()
  document.title = ''
})

describe('ScrollToTop', () => {
  it('returns the viewport to the top after in-app navigation', async () => {
    const scrollSpy = vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/cases']}>
        <ScrollToTop />
        <Link to="/analysis">Open analysis</Link>
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('link', { name: /open analysis/i }))
    expect(scrollSpy).toHaveBeenLastCalledWith({ top: 0, left: 0, behavior: 'instant' })
  })

  it('focuses and names the new page after route navigation', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/cases']}>
        <ScrollToTop />
        <Link to="/analysis">Open analysis</Link>
        <main id="main-content" tabIndex={-1}>
          <Routes>
            <Route path="/cases" element={<h1>Synthetic cases</h1>} />
            <Route path="/analysis" element={<h1>Attention pattern demo</h1>} />
          </Routes>
        </main>
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('link', { name: /open analysis/i }))

    const heading = screen.getByRole('heading', { name: /attention pattern demo/i })
    expect(heading).toHaveAttribute('tabindex', '-1')
    expect(heading).toHaveFocus()
    expect(document.title).toBe('Attention pattern demo | Facial Attention Lab')
  })

  it('waits for an asynchronously loaded route heading before announcing it', async () => {
    function AsyncHeading() {
      const [ready, setReady] = useState(false)
      useEffect(() => {
        void Promise.resolve().then(() => setReady(true))
      }, [])
      return ready ? <h1>Loaded analysis</h1> : <div role="status">Loading…</div>
    }

    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/cases']}>
        <ScrollToTop />
        <Link to="/analysis?case=demo-001">Open loaded analysis</Link>
        <main id="main-content" tabIndex={-1}>
          <Routes>
            <Route path="/cases" element={<h1>Synthetic cases</h1>} />
            <Route path="/analysis" element={<AsyncHeading />} />
          </Routes>
        </main>
      </MemoryRouter>,
    )

    await user.click(screen.getByRole('link', { name: /open loaded analysis/i }))
    const heading = await screen.findByRole('heading', { name: /loaded analysis/i })

    await waitFor(() => expect(heading).toHaveFocus())
    expect(document.title).toBe('Loaded analysis | Facial Attention Lab')
  })
})
