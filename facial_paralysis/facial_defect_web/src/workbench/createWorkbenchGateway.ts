import {
  DEFAULT_HTTP_WORKBENCH_TIMEOUT_MS,
  HttpWorkbenchGateway,
  normalizeHttpWorkbenchApiUrl,
  resolveHttpWorkbenchTimeoutMs,
} from './HttpWorkbenchGateway'
import { MockWorkbenchGateway } from './MockWorkbenchGateway'
import type {
  ConnectedWorkbenchGateway,
  WorkbenchGateway,
} from './WorkbenchGateway'
import { WorkbenchError } from './types'

export type WorkbenchGatewayConfig = {
  readonly enabled?: boolean
  readonly apiUrl?: string
  readonly timeoutMs?: number
}

export type ConnectedWorkbenchGatewayFactory = (
  apiUrl: string,
  timeoutMs: number,
) => ConnectedWorkbenchGateway

export type CreateWorkbenchGatewayDependencies = {
  readonly connectedFactory?: ConnectedWorkbenchGatewayFactory
}

function environmentConfig(): WorkbenchGatewayConfig {
  return {
    enabled: import.meta.env.VITE_ENABLE_CONNECTED_MODE === 'true',
    apiUrl: import.meta.env.VITE_ATTENTION_API_URL,
  }
}

const defaultConnectedFactory: ConnectedWorkbenchGatewayFactory = (
  apiUrl,
  timeoutMs,
) =>
  new HttpWorkbenchGateway(apiUrl, {
    timeoutMs,
  })

export function createWorkbenchGateway(
  config: WorkbenchGatewayConfig = environmentConfig(),
  dependencies: CreateWorkbenchGatewayDependencies = {},
): WorkbenchGateway {
  if (!config.enabled) return new MockWorkbenchGateway()

  const apiUrl = config.apiUrl?.trim()
  if (!apiUrl) {
    throw new WorkbenchError({
      reason: 'CONNECTED_MODE_NOT_CONFIGURED',
      message: 'Connected workbench mode requires an explicit API URL.',
      field: 'apiUrl',
    })
  }

  const normalizedApiUrl = normalizeHttpWorkbenchApiUrl(apiUrl)
  const timeoutMs = resolveHttpWorkbenchTimeoutMs(
    config.timeoutMs ?? DEFAULT_HTTP_WORKBENCH_TIMEOUT_MS,
  )

  return (dependencies.connectedFactory ?? defaultConnectedFactory)(
    normalizedApiUrl,
    timeoutMs,
  )
}
