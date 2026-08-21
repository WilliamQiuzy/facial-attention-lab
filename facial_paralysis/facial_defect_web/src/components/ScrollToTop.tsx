import { useEffect } from 'react'
import { useLocation } from 'react-router-dom'

export function ScrollToTop() {
  const { pathname } = useLocation()

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })

    const main = document.getElementById('main-content')
    const announceHeading = () => {
      const heading = main?.querySelector<HTMLElement>('h1')
      if (!heading) return false

      heading.tabIndex = -1
      const headingText = heading.textContent?.trim()
      if (headingText) {
        document.title = `${headingText} | FaceAI`
      }
      heading.focus({ preventScroll: true })
      return true
    }

    if (announceHeading()) return undefined

    main?.focus({ preventScroll: true })
    if (!main) return undefined

    const observer = new MutationObserver(() => {
      if (announceHeading()) observer.disconnect()
    })
    observer.observe(main, { childList: true, subtree: true })

    return () => observer.disconnect()
  }, [pathname])

  return null
}
