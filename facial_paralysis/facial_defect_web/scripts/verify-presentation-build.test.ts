// @vitest-environment node
import { createHash } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { PRESENTATION_BOUNDARY } from '../src/data/presentationDemoManifest'
// @ts-expect-error The verifier is intentionally executable as plain Node ESM.
import { verifyPresentationHtml } from './verify-presentation-build.mjs'

const pre = Buffer.from('pre-image')
const post = Buffer.from('post-image')
const expectedImageHashes = [pre, post].map((bytes) =>
  createHash('sha256').update(bytes).digest('hex'),
)

function fixture(extra = '') {
  return `<!doctype html><html><body>${PRESENTATION_BOUNDARY}<img src="data:image/png;base64,${pre.toString('base64')}"><img src="data:image/png;base64,${post.toString('base64')}">${extra}</body></html>`
}

describe('offline presentation build verifier', () => {
  it('accepts one self-contained HTML with the exact disclosure and image bytes', () => {
    expect(() =>
      verifyPresentationHtml({
        html: fixture(),
        directoryEntries: ['presentation.html'],
        expectedImageHashes,
        boundary: PRESENTATION_BOUNDARY,
      }),
    ).not.toThrow()
  })

  it.each([
    ['network script', '<script src="https://example.test/app.js"></script>'],
    ['MediaPipe model', '<span>face_landmarker.task</span>'],
    ['WASM payload', '<span>detector.wasm</span>'],
    ['camera permission', '<script>navigator.mediaDevices.getUserMedia()</script>'],
    ['runtime fetch', '<script>fetch("/asset")</script>'],
  ])('rejects a %s dependency', (_name, unsafeMarkup) => {
    expect(() =>
      verifyPresentationHtml({
        html: fixture(unsafeMarkup),
        directoryEntries: ['presentation.html'],
        expectedImageHashes,
        boundary: PRESENTATION_BOUNDARY,
      }),
    ).toThrow(/offline|dependency|network|camera|model|wasm/i)
  })

  it('rejects missing or substituted source image bytes', () => {
    expect(() =>
      verifyPresentationHtml({
        html: fixture().replace(post.toString('base64'), Buffer.from('wrong').toString('base64')),
        directoryEntries: ['presentation.html'],
        expectedImageHashes,
        boundary: PRESENTATION_BOUNDARY,
      }),
    ).toThrow(/image|sha/i)
  })

  it('rejects sibling build files and a weakened disclosure', () => {
    expect(() =>
      verifyPresentationHtml({
        html: fixture().replace(PRESENTATION_BOUNDARY, 'Simulation only'),
        directoryEntries: ['presentation.html', 'assets'],
        expectedImageHashes,
        boundary: PRESENTATION_BOUNDARY,
      }),
    ).toThrow(/single|disclosure/i)
  })
})
