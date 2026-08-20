import { createHash } from 'node:crypto'
import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const EXACT_BOUNDARY =
  'HAND-AUTHORED SIMULATION — NOT HUMAN GAZE — NOT A PREDICTED SURGICAL OUTCOME — CLINICAL USE BLOCKED'

export const EXACT_IMAGE_HASHES = [
  '1c43951ed068dc7b88a28e1a0e68d724f1f8b649067b81323ceb30fe2cc5eb30',
  '72d0ee02fa6313b9ddb3c6b4ccf3c1f8c277c98b51a3b4453152948f21e7a58b',
]

const forbiddenRuntimePatterns = [
  {
    pattern:
      /<(?:script|link|img|source)\b[^>]+(?:src|href)=["'](?!data:|#)[^"']+["']/i,
    label: 'network or sibling-file dependency',
  },
  { pattern: /face_landmarker\.task/i, label: 'MediaPipe model dependency' },
  { pattern: /\.wasm(?:\b|[?#"'])/i, label: 'WASM dependency' },
  {
    pattern: /(?:getUserMedia|navigator\.mediaDevices)/i,
    label: 'camera dependency',
  },
  { pattern: /\bfetch\s*\(/i, label: 'runtime network dependency' },
  { pattern: /XMLHttpRequest/i, label: 'runtime network dependency' },
]

function hash(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

export function verifyPresentationHtml({
  html,
  directoryEntries,
  expectedImageHashes,
  boundary,
}) {
  if (
    directoryEntries.length !== 1 ||
    directoryEntries[0] !== 'presentation.html'
  ) {
    throw new Error(
      'The offline package must contain the single presentation.html file only.',
    )
  }
  if (!html.includes(boundary)) {
    throw new Error('The exact presentation disclosure is missing.')
  }

  for (const { pattern, label } of forbiddenRuntimePatterns) {
    if (pattern.test(html)) {
      throw new Error(`Offline presentation contains a ${label}.`)
    }
  }

  const embeddedPngHashes = [
    ...html.matchAll(/data:image\/png;base64,([a-z0-9+/=]+)/gi),
  ].map((match) => hash(Buffer.from(match[1], 'base64')))
  const uniqueHashes = new Set(embeddedPngHashes)

  if (
    expectedImageHashes.length !== 2 ||
    uniqueHashes.size !== 2 ||
    expectedImageHashes.some((expected) => !uniqueHashes.has(expected))
  ) {
    throw new Error(
      'The offline presentation does not contain both exact SHA-bound image payloads.',
    )
  }
}

async function verifyBuildDirectory(directory) {
  const entries = (await readdir(directory)).sort()
  const html = await readFile(path.join(directory, 'presentation.html'), 'utf8')
  verifyPresentationHtml({
    html,
    directoryEntries: entries,
    expectedImageHashes: EXACT_IMAGE_HASHES,
    boundary: EXACT_BOUNDARY,
  })
  console.log(
    `Verified offline presentation: ${path.join(directory, 'presentation.html')}`,
  )
}

const isCli = process.argv[1]
  ? fileURLToPath(import.meta.url) === path.resolve(process.argv[1])
  : false

if (isCli) {
  await verifyBuildDirectory(
    path.resolve(process.cwd(), process.argv[2] ?? 'presentation-dist'),
  )
}
