import { describe, expect, it } from 'vitest'

import { createDemonstrationResult } from './demonstration'

describe('createDemonstrationResult', () => {
  it('permanently labels its values as demonstration rather than model output', () => {
    const file = new File(['synthetic'], 'example.webm', { type: 'video/webm' })
    const result = createDemonstrationResult(file)

    expect(result.mode).toBe('demonstration')
    expect(result.provenanceLabel).toBe('DEMONSTRATION - NOT MODEL OUTPUT')
    expect(result.modelSha256).toBeNull()
  })

  it('is deterministic for the same local file metadata', () => {
    const file = new File(['synthetic'], 'example.webm', {
      type: 'video/webm',
      lastModified: 100,
    })
    expect(createDemonstrationResult(file)).toEqual(createDemonstrationResult(file))
  })
})
