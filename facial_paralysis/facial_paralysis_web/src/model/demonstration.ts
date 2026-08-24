export type SeverityLabel = 'Normal' | 'Slight' | 'Strong'

export interface RegionalSeverity {
  readonly level: 0 | 1 | 2
  readonly expected: number
  readonly pGt: readonly [number, number]
  readonly label: SeverityLabel
}

export interface DemonstrationResult {
  readonly mode: 'demonstration'
  readonly provenanceLabel: 'DEMONSTRATION - NOT MODEL OUTPUT'
  readonly modelSha256: null
  readonly scores: {
    readonly palsyProbability: number
    readonly eyes: RegionalSeverity
    readonly mouth: RegionalSeverity
  }
}

function checksum(input: string): number {
  let value = 2166136261
  for (let index = 0; index < input.length; index += 1) {
    value ^= input.charCodeAt(index)
    value = Math.imul(value, 16777619)
  }
  return value >>> 0
}

function region(level: 0 | 1 | 2, offset: number): RegionalSeverity {
  const labels = ['Normal', 'Slight', 'Strong'] as const
  const first = Math.min(0.94, 0.44 + level * 0.2 + offset)
  const second = Math.min(first, Math.max(0.08, 0.14 + level * 0.28 - offset / 2))
  return {
    level,
    expected: Number((first + second).toFixed(2)),
    pGt: [Number(first.toFixed(2)), Number(second.toFixed(2))],
    label: labels[level],
  }
}

export function createDemonstrationResult(file: File): DemonstrationResult {
  const seed = checksum(`${file.name}:${file.size}:${file.lastModified}`)
  const eyesLevel = (seed % 3) as 0 | 1 | 2
  const mouthLevel = ((seed >>> 4) % 3) as 0 | 1 | 2
  const probability = Number((0.34 + (seed % 51) / 100).toFixed(2))
  const offset = ((seed % 9) - 4) / 100

  return {
    mode: 'demonstration',
    provenanceLabel: 'DEMONSTRATION - NOT MODEL OUTPUT',
    modelSha256: null,
    scores: {
      palsyProbability: probability,
      eyes: region(eyesLevel, offset),
      mouth: region(mouthLevel, -offset),
    },
  }
}
