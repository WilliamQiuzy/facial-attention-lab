import type { ReactNode } from 'react'

type StatusBadgeProps = {
  readonly children: ReactNode
  readonly tone?: 'neutral' | 'info' | 'success' | 'warning' | 'blocked'
}

export function StatusBadge({ children, tone = 'neutral' }: StatusBadgeProps) {
  return <span className={`status-badge status-badge--${tone}`}>{children}</span>
}
