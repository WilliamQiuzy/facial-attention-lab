import { Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'

const navigation = [
  { to: '/', label: 'Overview', end: true },
  { to: '/cases', label: 'Synthetic cases' },
  { to: '/analysis', label: 'Attention demo' },
  { to: '/patient-report', label: 'Patient explanation' },
  { to: '/model', label: 'Model & data' },
  { to: '/methods', label: 'Methods' },
]

export function AppHeader() {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <header className="site-header">
      <div className="site-header__inner">
        <NavLink className="brand" to="/" onClick={() => setMenuOpen(false)}>
          <span className="brand__mark" aria-hidden="true">
            FA
          </span>
          <span>
            <span className="brand__name">Facial Attention Lab</span>
            <span className="brand__context">
              Independent research prototype · Mayo-inspired visual system
            </span>
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
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
