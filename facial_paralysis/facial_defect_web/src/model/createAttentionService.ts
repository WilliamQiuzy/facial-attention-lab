import { HttpAttentionService } from './httpAttentionService'
import { MockAttentionService } from './mockAttentionService'
import type { AttentionService } from './types'

export type AttentionServiceConfig = {
  enableConnectedMode?: boolean
  apiUrl?: string
}

function environmentConfig(): Required<Pick<AttentionServiceConfig, 'enableConnectedMode'>> &
  Pick<AttentionServiceConfig, 'apiUrl'> {
  return {
    enableConnectedMode: import.meta.env.VITE_ENABLE_CONNECTED_MODE === 'true',
    apiUrl: import.meta.env.VITE_ATTENTION_API_URL,
  }
}

export function createAttentionService(
  config: AttentionServiceConfig = environmentConfig(),
): AttentionService {
  if (!config.enableConnectedMode) return new MockAttentionService()

  if (!config.apiUrl?.trim()) {
    throw new Error('Connected mode requires an explicit API URL.')
  }

  return new HttpAttentionService(config.apiUrl)
}
