import { describe, expect, it } from 'vitest'

import { getSessionFilename } from './sessionIdentity'

describe('anonymous session filenames', () => {
  const metadata = { type: 'video/webm;codecs=vp9,opus', lastModified: Date.UTC(2026, 8, 8, 14, 22, 33, 123) }

  it('links report and recording with a deterministic timestamp and anonymous suffix', () => {
    const recording = new File(['video'], 'Patient-Name-123.webm', metadata)
    const pdf = getSessionFilename(recording, 'report')
    const video = getSessionFilename(recording, 'recording')
    expect(pdf).toMatch(/^faces-research-20260908T142233123Z-[a-z0-9]+-report\.pdf$/)
    expect(video.replace('-recording.webm', '')).toBe(pdf.replace('-report.pdf', ''))
    expect(getSessionFilename(recording, 'report')).toBe(pdf)
    expect(pdf).not.toMatch(/patient|123\.webm/i)
  })

  it('does not derive identity from a potentially identifiable source filename', () => {
    const first = new File(['video'], 'Alice.webm', metadata)
    const renamed = new File(['video'], 'Bob.webm', metadata)
    const changed = new File(['other video'], 'Alice.webm', metadata)
    expect(getSessionFilename(first, 'report')).toBe(getSessionFilename(renamed, 'report'))
    expect(getSessionFilename(first, 'report')).not.toBe(getSessionFilename(changed, 'report'))
  })

  it.each([
    ['video/mp4', 'name.webm', 'mp4'],
    ['video/quicktime', 'name', 'mov'],
    ['video/webm;codecs=vp8,opus', 'name', 'webm'],
    ['', 'MRN-123.MOV', 'mov'],
    ['', 'MRN-123.ogg', 'ogg'],
    ['', 'MRN-123.html', 'video'],
  ])('preserves a safe container extension for %s', (type, name, extension) => {
    expect(getSessionFilename(new File(['video'], name, { ...metadata, type }), 'recording')).toMatch(new RegExp(`-recording\\.${extension}$`))
  })
})
