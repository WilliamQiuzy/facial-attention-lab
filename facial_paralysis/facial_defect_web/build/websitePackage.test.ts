// @vitest-environment node

import { existsSync } from 'node:fs'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

describe('packaged FaceAI website server', () => {
  it('serves static assets and falls back to the app for patient routes', async () => {
    const serverPath = path.resolve('website-package/server.mjs')
    const temporaryRoot = await mkdtemp(path.join(tmpdir(), 'faceai-package-'))
    const siteRoot = path.join(temporaryRoot, 'site')
    await mkdir(path.join(siteRoot, 'assets'), { recursive: true })
    await writeFile(
      path.join(siteRoot, 'index.html'),
      '<!doctype html><title>FaceAI</title><main>FaceAI website</main>',
    )
    await writeFile(path.join(siteRoot, 'assets', 'app.js'), 'window.faceAi = true')

    expect(existsSync(serverPath)).toBe(true)

    const { createFaceAiServer } = await import(pathToFileURL(serverPath).href)
    const server = createFaceAiServer({ siteRoot })

    try {
      await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
      const address = server.address()
      if (!address || typeof address === 'string') {
        throw new Error('FaceAI test server did not bind to a TCP port.')
      }
      const origin = `http://127.0.0.1:${address.port}`

      const home = await fetch(`${origin}/`)
      expect(home.status).toBe(200)
      expect(home.headers.get('content-type')).toContain('text/html')
      expect(await home.text()).toContain('FaceAI website')

      const patientRoute = await fetch(`${origin}/patients/sample-record`)
      expect(patientRoute.status).toBe(200)
      expect(await patientRoute.text()).toContain('FaceAI website')

      const asset = await fetch(`${origin}/assets/app.js`)
      expect(asset.status).toBe(200)
      expect(asset.headers.get('content-type')).toContain('text/javascript')
      expect(await asset.text()).toContain('window.faceAi')

      const post = await fetch(`${origin}/`, { method: 'POST' })
      expect(post.status).toBe(405)
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((error: Error | undefined) => {
          if (error) reject(error)
          else resolve()
        })
      }).catch(() => undefined)
      await rm(temporaryRoot, { recursive: true, force: true })
    }
  })
})
