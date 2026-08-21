import { createHash } from 'node:crypto'
import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

export const EXACT_BOUNDARY =
  'SYNTHETIC DEMO — ILLUSTRATIVE ATTENTION, NOT MEASURED GAZE'

export const EXACT_IMAGE_HASHES = [
  '1c43951ed068dc7b88a28e1a0e68d724f1f8b649067b81323ceb30fe2cc5eb30',
  '4e62df4478f6852d788adf58a77056e208a72f5936b0960af8d6ed17af6d95e5',
  '14909f75c0ba2ae4ab5e4ac1cf976f261d6d61eff09ae5804d865e2e6229d374',
  'ecbb751e8c13b37465f899fd912a4dc4f713088388ca7a3ae94e925e55743b8b',
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
  const disclosureCount = html.split(boundary).length - 1
  if (disclosureCount !== 1) {
    throw new Error(
      'The exact presentation disclosure must appear once and only once.',
    )
  }
  if (/clinical use blocked/i.test(html)) {
    throw new Error(
      'The presentation contains duplicated clinical-boundary copy.',
    )
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
    expectedImageHashes.length !== 4 ||
    uniqueHashes.size !== 4 ||
    expectedImageHashes.some((expected) => !uniqueHashes.has(expected))
  ) {
    throw new Error(
      'The offline presentation does not contain all four exact SHA-bound image payloads.',
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
