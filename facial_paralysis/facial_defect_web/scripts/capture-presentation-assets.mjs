import { createHash } from 'node:crypto'
import {
  copyFile,
  mkdir,
  readFile,
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
  'facial-attention-presentation-demo.html',
)

await mkdir(screenshotRoot, { recursive: true })
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
    'before-after-photo-attention.png',
  )

  await page.getByRole('radio', { name: 'Outline' }).click()
  await capture(
    page.locator('.presentation-comparison'),
    'before-after-outline-attention.png',
  )

  await page.getByRole('radio', { name: 'Post-operative' }).click()
  await page.getByRole('radio', { name: 'Photo' }).click()
  await page.getByRole('checkbox', { name: 'Show attention layer' }).uncheck()
  await capture(
    page.locator('.presentation-stage'),
    'postoperative-scar-clean.png',
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
    path: path.join(screenshotRoot, 'demo-mobile.png'),
  })
  capturedFiles.push(path.join(screenshotRoot, 'demo-mobile.png'))
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
