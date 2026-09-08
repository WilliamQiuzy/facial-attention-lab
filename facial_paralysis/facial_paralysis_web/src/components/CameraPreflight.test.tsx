import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useCameraPreflight } from '../hooks/useCameraPreflight'
import { CameraPreflight } from './CameraPreflight'

vi.mock('../hooks/useCameraPreflight', () => ({ useCameraPreflight: vi.fn(() => 'checking') }))

describe('optional camera framing guidance', () => {
  const videoRef = { current: null }
  beforeEach(() => { vi.mocked(useCameraPreflight).mockReturnValue('checking') })

  it('renders nothing when inactive and passes the parent video reference unchanged', () => {
    const { container } = render(<CameraPreflight videoRef={videoRef} active={false} />)
    expect(container).toBeEmptyDOMElement()
    expect(useCameraPreflight).toHaveBeenCalledWith(videoRef, false)
  })

  it.each([
    ['closer', 'Come a little closer'],
    ['center', 'Center your face'],
    ['light', 'The lighting looks dim'],
    ['no-face', 'Bring your face into view'],
    ['one-face', 'Keep one face in view'],
    ['ready', 'Framing looks good'],
    ['unavailable', 'Camera hints unavailable'],
  ] as const)('announces only the current %s hint without any recording gate', (hint, title) => {
    vi.mocked(useCameraPreflight).mockReturnValue(hint)
    render(<CameraPreflight videoRef={videoRef} active />)
    expect(screen.getAllByRole('status')).toHaveLength(1)
    expect(screen.getByRole('status')).toHaveTextContent(title)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
    expect(screen.getByText(/you can start anytime/i)).toBeInTheDocument()
  })
})
