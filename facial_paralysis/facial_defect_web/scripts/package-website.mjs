import { chmod, copyFile, cp, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const releaseRoot = path.join(projectRoot, 'release')
const outputPath = path.resolve(
  process.argv[2] ?? path.join(releaseRoot, 'FaceAI-Website.zip'),
)
const packageName = 'FaceAI-Website'
const temporaryRoot = await mkdtemp(path.join(tmpdir(), 'faceai-website-package-'))
const packageRoot = path.join(temporaryRoot, packageName)

try {
  await mkdir(packageRoot, { recursive: true })
  await cp(path.join(projectRoot, 'dist'), path.join(packageRoot, 'site'), {
    recursive: true,
  })

  for (const fileName of [
    'server.mjs',
    'START-FACEAI-MAC.command',
    'START-FACEAI-WINDOWS.bat',
    'README-START-HERE.txt',
  ]) {
    await copyFile(
      path.join(projectRoot, 'website-package', fileName),
      path.join(packageRoot, fileName),
    )
  }
  await chmod(path.join(packageRoot, 'START-FACEAI-MAC.command'), 0o755)

  const revision = spawnSync('git', ['rev-parse', '--short=12', 'HEAD'], {
    cwd: projectRoot,
    encoding: 'utf8',
  })
  const revisionText = revision.status === 0
    ? revision.stdout.trim()
    : 'unavailable'
  await writeFile(
    path.join(packageRoot, 'BUILD-INFO.txt'),
    `FaceAI full website\nRevision: ${revisionText}\nGenerated: ${new Date().toISOString()}\n`,
  )

  const temporaryZip = path.join(temporaryRoot, 'FaceAI-Website.zip')
  const zip = spawnSync('zip', ['-r', '-q', temporaryZip, packageName], {
    cwd: temporaryRoot,
    encoding: 'utf8',
  })
  if (zip.status !== 0) {
    throw new Error(zip.stderr || 'The zip command failed.')
  }

  await mkdir(path.dirname(outputPath), { recursive: true })
  await copyFile(temporaryZip, outputPath)
  console.log(`Created ${outputPath}`)
} finally {
  await rm(temporaryRoot, { recursive: true, force: true })
}
