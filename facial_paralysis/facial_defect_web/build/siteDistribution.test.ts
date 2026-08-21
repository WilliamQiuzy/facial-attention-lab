import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'

describe('FaceAI website distribution', () => {
  const projectRoot = process.cwd()
  const packageJson = JSON.parse(
    readFileSync(path.join(projectRoot, 'package.json'), 'utf8'),
  ) as {
    name: string
    scripts: Record<string, string>
    devDependencies: Record<string, string>
  }

  it('ships as the FaceAI website without a standalone demo build', () => {
    expect(packageJson.name).toBe('faceai')
    expect(packageJson.scripts).not.toHaveProperty('build:presentation')
    expect(packageJson.scripts).not.toHaveProperty('capture:presentation')
    expect(packageJson.devDependencies).not.toHaveProperty('vite-plugin-singlefile')
    expect(packageJson.scripts.dev).toContain('--host localhost')
    expect(packageJson.scripts.preview).toContain('--host localhost')

    for (const retiredPath of [
      'presentation.html',
      'vite.presentation.config.ts',
      'presentation-assets',
      'src/presentationEntry.tsx',
      'src/pages/PresentationDemoPage.tsx',
    ]) {
      expect(existsSync(path.join(projectRoot, retiredPath))).toBe(false)
    }
  })

  it('documents the complete website as the doctor-facing run target', () => {
    const readme = readFileSync(path.join(projectRoot, 'README.md'), 'utf8')

    expect(readme).toContain('Run the complete FaceAI website')
    expect(readme).not.toMatch(
      /FaceAI-Demo|single-file|one-file offline|build:presentation|capture:presentation/i,
    )
  })
})
