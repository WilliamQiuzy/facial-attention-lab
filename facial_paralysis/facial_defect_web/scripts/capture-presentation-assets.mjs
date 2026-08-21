import { createHash } from 'node:crypto'
import {
  copyFile,
  mkdir,
  readFile,
  readdir,
  unlink,
  writeFile,
} from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { chromium } from 'playwright'
import {
  EXACT_BOUNDARY,
  EXACT_IMAGE_HASHES,
  verifyPresentationHtml,
} from './verify-presentation-build.mjs'

const outputRoot = path.resolve('presentation-assets')
const screenshotRoot = path.join(outputRoot, 'screenshots')
const builtHtmlPath = path.resolve('presentation-dist/presentation.html')
const shareableHtmlPath = path.join(
  outputRoot,
  'FaceAI-Demo.html',
)

await mkdir(screenshotRoot, { recursive: true })
await Promise.all(
  (await readdir(screenshotRoot))
    .filter((entry) => entry.endsWith('.png'))
    .map((entry) => unlink(path.join(screenshotRoot, entry))),
)
await copyFile(builtHtmlPath, shareableHtmlPath)

const html = await readFile(shareableHtmlPath, 'utf8')
verifyPresentationHtml({
  html,
  directoryEntries: ['presentation.html'],
  expectedImageHashes: EXACT_IMAGE_HASHES,
  boundary: EXACT_BOUNDARY,
})

const browser = await chromium.launch({ headless: true })
const capturedFiles = []

async function capture(locator, filename, options = {}) {
  const filePath = path.join(screenshotRoot, filename)
  await locator.screenshot({ path: filePath, ...options })
  capturedFiles.push(filePath)
}

function watchOfflinePage(page, errors, remoteRequests) {
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text())
  })
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('request', (request) => {
    if (/^https?:/i.test(request.url())) remoteRequests.push(request.url())
  })
}

try {
  const context = await browser.newContext({
    offline: true,
    viewport: { width: 1440, height: 1000 },
    deviceScaleFactor: 1,
  })
  const page = await context.newPage()
  const errors = []
  const remoteRequests = []
  watchOfflinePage(page, errors, remoteRequests)

  await page.goto(pathToFileURL(shareableHtmlPath).href, {
    waitUntil: 'load',
  })
  await page.waitForFunction(() =>
    [...document.images].every((image) => image.complete),
  )

  await capture(
    page.locator('.presentation-comparison'),
    'subject-a-before-after-photo-attention.png',
  )

  await page.getByRole('checkbox', { name: 'Show attention layer' }).uncheck()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-a-before-after-photo-clean.png',
  )
  await page.getByRole('checkbox', { name: 'Show attention layer' }).check()

  await page.getByRole('radio', { name: 'Outline', exact: true }).click()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-a-before-after-outline-attention.png',
  )

  await page.getByRole('radio', { name: 'Photo + outline', exact: true }).click()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-a-before-after-photo-outline-attention.png',
  )

  await page.getByRole('radio', { name: 'Drag slider', exact: true }).click()
  await capture(
    page.locator('.presentation-wipe'),
    'subject-a-before-after-drag-comparison.png',
  )

  await page.getByRole('radio', { name: 'Post-operative', exact: true }).click()
  await page.getByRole('radio', { name: 'Photo', exact: true }).click()
  await page.getByRole('checkbox', { name: 'Show attention layer' }).uncheck()
  await capture(
    page.locator('.presentation-stage'),
    'subject-a-postoperative-wound-clean.png',
  )

  await page.getByRole('radio', { name: 'Subject B', exact: true }).click()
  await page.getByRole('radio', { name: 'Both', exact: true }).click()
  await page.getByRole('radio', { name: 'Side by side', exact: true }).click()
  await page.getByRole('checkbox', { name: 'Show attention layer' }).check()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-b-before-after-photo-attention.png',
  )

  await page.getByRole('checkbox', { name: 'Show attention layer' }).uncheck()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-b-before-after-photo-clean.png',
  )
  await page.getByRole('checkbox', { name: 'Show attention layer' }).check()

  await page.getByRole('radio', { name: 'Outline', exact: true }).click()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-b-before-after-outline-attention.png',
  )

  await page.getByRole('radio', { name: 'Photo + outline', exact: true }).click()
  await capture(
    page.locator('.presentation-comparison'),
    'subject-b-before-after-photo-outline-attention.png',
  )

  await page.getByRole('radio', { name: 'Drag slider', exact: true }).click()
  await capture(
    page.locator('.presentation-wipe'),
    'subject-b-before-after-drag-comparison.png',
  )

  await page.getByRole('radio', { name: 'Post-operative', exact: true }).click()
  await page.getByRole('radio', { name: 'Photo', exact: true }).click()
  await page.getByRole('checkbox', { name: 'Show attention layer' }).uncheck()
  await capture(
    page.locator('.presentation-stage'),
    'subject-b-postoperative-wound-clean.png',
  )

  await context.close()

  const mobileContext = await browser.newContext({
    offline: true,
    viewport: { width: 390, height: 844 },
    deviceScaleFactor: 1,
  })
  const mobilePage = await mobileContext.newPage()
  watchOfflinePage(mobilePage, errors, remoteRequests)
  await mobilePage.goto(pathToFileURL(shareableHtmlPath).href, {
    waitUntil: 'load',
  })
  await mobilePage.waitForFunction(() =>
    [...document.images].every((image) => image.complete),
  )
  await mobilePage.screenshot({
    path: path.join(screenshotRoot, 'subject-a-demo-mobile.png'),
  })
  capturedFiles.push(path.join(screenshotRoot, 'subject-a-demo-mobile.png'))
  await mobilePage.getByRole('radio', { name: 'Subject B', exact: true }).click()
  await mobilePage.screenshot({
    path: path.join(screenshotRoot, 'subject-b-demo-mobile.png'),
  })
  capturedFiles.push(path.join(screenshotRoot, 'subject-b-demo-mobile.png'))
  if (errors.length > 0) {
    throw new Error(`Browser errors: ${errors.join(' | ')}`)
  }
  if (remoteRequests.length > 0) {
    throw new Error(
      `Offline presentation attempted remote requests: ${remoteRequests.join(', ')}`,
    )
  }

  await mobileContext.close()
} finally {
  await browser.close()
}

const hashFile = async (filePath) =>
  createHash('sha256').update(await readFile(filePath)).digest('hex')

const manifest = {
  schema: 'facial-attention-presentation-assets/1',
  generatedFrom: 'presentation-dist/presentation.html',
  disclosure: EXACT_BOUNDARY,
  offlineHtml: {
    path: path.relative(outputRoot, shareableHtmlPath),
    sha256: await hashFile(shareableHtmlPath),
  },
  screenshots: await Promise.all(
    capturedFiles.map(async (filePath) => ({
      path: path.relative(outputRoot, filePath),
      sha256: await hashFile(filePath),
    })),
  ),
  sourceImageSha256: EXACT_IMAGE_HASHES,
}

await writeFile(
  path.join(outputRoot, 'manifest.json'),
  `${JSON.stringify(manifest, null, 2)}\n`,
)

console.log(`Wrote ${shareableHtmlPath}`)
console.log(`Captured ${capturedFiles.length} presentation screenshots.`)
