import { Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const navigation = [
  { to: '/patients', label: 'Patients' },
  { to: '/reviews', label: 'Reviews' },
  { to: '/about', label: 'Help' },
] as const

export function AppHeader() {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <header className="site-header workspace-header">
      <div className="site-header__inner">
        <NavLink className="brand" to="/patients" onClick={() => setMenuOpen(false)}>
          <span className="brand__mark" aria-hidden="true">FR</span>
          <span className="brand__name">Facial Reconstruction Imaging</span>
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
            <NavLink key={item.to} to={item.to} onClick={() => setMenuOpen(false)}>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
