import { demoAnalysis } from '../data/demoCase'
import type { AttentionAnalysis, DemoAttentionService } from './types'

export class MockAttentionService implements DemoAttentionService {
  readonly mode = 'demo' as const

  async getDemoAnalysis(): Promise<AttentionAnalysis> {
    return demoAnalysis
  }
}
