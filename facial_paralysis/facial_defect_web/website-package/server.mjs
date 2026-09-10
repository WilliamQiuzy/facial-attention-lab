import { createReadStream, realpathSync } from 'node:fs'
import { stat } from 'node:fs/promises'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'

const MIME_TYPES = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
  ['.task', 'application/octet-stream'],
  ['.wasm', 'application/wasm'],
  ['.webp', 'image/webp'],
])

function resolveInside(root, requestPath) {
  let decodedPath
  try {
    decodedPath = decodeURIComponent(requestPath)
  } catch {
    return null
  }
  if (decodedPath.includes('\0')) return null

  const normalizedRoot = path.resolve(root)
  const normalizedRequest = path.posix
    .normalize(decodedPath)
    .replace(/^\/+/, '')
  const candidate = path.resolve(normalizedRoot, normalizedRequest)
  if (
    candidate !== normalizedRoot &&
    !candidate.startsWith(`${normalizedRoot}${path.sep}`)
  ) {
    return null
  }
  return candidate
}

async function existingFile(candidate) {
  try {
    const fileStat = await stat(candidate)
    if (fileStat.isFile()) return candidate
    if (fileStat.isDirectory()) {
      const indexPath = path.join(candidate, 'index.html')
      const indexStat = await stat(indexPath)
      return indexStat.isFile() ? indexPath : null
    }
  } catch {
    return null
  }
  return null
}

function sendText(response, statusCode, message) {
  response.writeHead(statusCode, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  })
  response.end(message)
}

export function createFaceAiServer({ siteRoot }) {
  const absoluteSiteRoot = path.resolve(siteRoot)

  return http.createServer(async (request, response) => {
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.setHeader('Allow', 'GET, HEAD')
      sendText(response, 405, 'Method not allowed')
      return
    }

    const requestUrl = new URL(request.url ?? '/', 'http://localhost')
    const candidate = resolveInside(absoluteSiteRoot, requestUrl.pathname)
    if (!candidate) {
      sendText(response, 400, 'Invalid request path')
      return
    }

    let filePath = await existingFile(candidate)
    if (!filePath && !path.extname(requestUrl.pathname)) {
      filePath = await existingFile(path.join(absoluteSiteRoot, 'index.html'))
    }
    if (!filePath) {
      sendText(response, 404, 'File not found')
      return
    }

    const extension = path.extname(filePath).toLowerCase()
    response.writeHead(200, {
      'Content-Type': MIME_TYPES.get(extension) ?? 'application/octet-stream',
      'Cache-Control': extension === '.html'
        ? 'no-store'
        : 'public, max-age=31536000, immutable',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
    })
    if (request.method === 'HEAD') {
      response.end()
      return
    }

    const stream = createReadStream(filePath)
    stream.on('error', () => {
      if (!response.headersSent) sendText(response, 500, 'Unable to read file')
      else response.destroy()
    })
    stream.pipe(response)
  })
}

function openBrowser(url) {
  const command = process.platform === 'darwin'
    ? ['open', [url]]
    : process.platform === 'win32'
      ? ['cmd', ['/c', 'start', '', url]]
      : ['xdg-open', [url]]
  const child = spawn(command[0], command[1], {
    detached: true,
    stdio: 'ignore',
  })
  child.unref()
}

function readArgument(name) {
  const index = process.argv.indexOf(name)
  return index >= 0 ? process.argv[index + 1] : undefined
}

function isDirectExecution(argumentPath) {
  if (!argumentPath) return false
  try {
    return realpathSync(argumentPath) === realpathSync(fileURLToPath(import.meta.url))
  } catch {
    return false
  }
}

const launchedDirectly = isDirectExecution(process.argv[1])

if (launchedDirectly) {
  const packageRoot = path.dirname(fileURLToPath(import.meta.url))
  const siteRoot = readArgument('--site') ?? path.join(packageRoot, 'site')
  const requestedPort = Number.parseInt(readArgument('--port') ?? '0', 10)
  const port = Number.isInteger(requestedPort) && requestedPort >= 0
    ? requestedPort
    : 0
  const shouldOpen = !process.argv.includes('--no-open')
  const server = createFaceAiServer({ siteRoot })

  server.listen(port, '127.0.0.1', () => {
    const address = server.address()
    if (!address || typeof address === 'string') {
      throw new Error('FaceAI could not determine the local server port.')
    }
    const url = `http://localhost:${address.port}/`
    console.log(`FaceAI is running at ${url}`)
    console.log('Keep this window open. Press Ctrl+C to stop FaceAI.')
    if (shouldOpen) openBrowser(url)
  })

  server.on('error', (error) => {
    console.error(`FaceAI could not start: ${error.message}`)
    process.exitCode = 1
  })

  const stop = () => server.close(() => process.exit(0))
  process.once('SIGINT', stop)
  process.once('SIGTERM', stop)
}
