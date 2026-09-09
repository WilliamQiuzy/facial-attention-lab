import { Menu, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'

const navigation = [
  { to: '/patients', label: 'Patients' },
  { to: '/reviews', label: 'Reviews' },
  { to: '/about', label: 'Help' },
] as const

export function AppHeader() {
  const { pathname } = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)
  const closeMenu = () => setMenuOpen(false)

  useEffect(() => {
    setMenuOpen(false)
  }, [pathname])

  return (
    <header className="site-header workspace-header">
      <div className="site-header__inner">
        <NavLink
          className="brand"
          to="/patients"
          aria-label="Facial Reconstruction Imaging"
          onClick={closeMenu}
        >
          <span className="brand__mark" aria-hidden="true">FR</span>
          <span className="brand__name brand__name--full" aria-hidden="true">
            Facial Reconstruction Imaging
          </span>
          <span className="brand__name brand__name--compact" aria-hidden="true">
            Facial Imaging
          </span>
        </NavLink>

        <button
          className="menu-button"
          type="button"
          aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={menuOpen}
          aria-controls="primary-navigation"
          onClick={() => setMenuOpen((open) => !open)}
        >
          {menuOpen ? <X aria-hidden="true" /> : <Menu aria-hidden="true" />}
          <span>{menuOpen ? 'Close' : 'Menu'}</span>
        </button>

        <nav
          id="primary-navigation"
          className={`primary-nav${menuOpen ? ' primary-nav--open' : ''}`}
          aria-label="Primary navigation"
        >
          {navigation.map((item) => (
            <NavLink key={item.to} to={item.to} onClick={closeMenu}>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
