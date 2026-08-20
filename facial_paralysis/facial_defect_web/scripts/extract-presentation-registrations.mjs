import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'

const browser = await chromium.launch({ headless: true })

try {
  const page = await browser.newPage()
  await page.goto('http://127.0.0.1:5173/registration-extract.html', {
    waitUntil: 'networkidle',
  })
  const output = page.locator('[data-testid="presentation-registrations"]')
  await output.waitFor({ state: 'visible' })
  await page.waitForFunction(() => {
    const element = document.querySelector(
      '[data-testid="presentation-registrations"]',
    )
    return element?.getAttribute('data-status') !== 'loading'
  })

  const status = await output.getAttribute('data-status')
  const text = await output.textContent()
  if (status !== 'ready' || !text) {
    throw new Error(text || 'Presentation registration extraction failed.')
  }

  await mkdir('tmp', { recursive: true })
  await writeFile(
    'tmp/presentation-registrations.json',
    `${JSON.stringify(JSON.parse(text), null, 2)}\n`,
  )
  console.log('Wrote tmp/presentation-registrations.json')
} finally {
  await browser.close()
}
