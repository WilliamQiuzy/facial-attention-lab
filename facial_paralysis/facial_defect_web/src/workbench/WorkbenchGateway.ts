import type { InferenceBinding, InferenceOutput } from './types'

export type WorkbenchGatewayMode = 'mock' | 'connected'

export type RunInferenceOptions = {
  readonly signal?: AbortSignal
}

export interface WorkbenchGateway {
  readonly mode: WorkbenchGatewayMode

  runInference(
    binding: InferenceBinding,
    options?: RunInferenceOptions,
  ): Promise<InferenceOutput>
}

export type ConnectedWorkbenchGateway = WorkbenchGateway & {
  readonly mode: 'connected'
}
