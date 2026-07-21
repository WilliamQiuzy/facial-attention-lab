import type {
  RunInferenceOptions,
  WorkbenchGateway,
} from './WorkbenchGateway'
import { runMockEngine } from './mockEngine'
import {
  WorkbenchError,
  type InferenceBinding,
  type MockInferenceOutput,
} from './types'

export type WorkbenchScheduler = () => Promise<void>

const immediateScheduler: WorkbenchScheduler = () => Promise.resolve()

function abortedRequest(): WorkbenchError {
  return new WorkbenchError({
    reason: 'REQUEST_ABORTED',
    message: 'The inference request was aborted.',
  })
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw abortedRequest()
}

function waitForSchedule(
  scheduled: Promise<void>,
  signal: AbortSignal | undefined,
): Promise<void> {
  if (!signal) return scheduled
  if (signal.aborted) return Promise.reject(abortedRequest())

  return new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener('abort', onAbort)
      reject(abortedRequest())
    }
    const settle = (operation: () => void) => {
      signal.removeEventListener('abort', onAbort)
      operation()
    }

    signal.addEventListener('abort', onAbort, { once: true })
    scheduled.then(
      () => settle(resolve),
      (error: unknown) => settle(() => reject(error)),
    )
  })
}

export class MockWorkbenchGateway implements WorkbenchGateway {
  readonly mode = 'mock' as const

  constructor(private readonly scheduler: WorkbenchScheduler = immediateScheduler) {}

  async runInference(
    binding: InferenceBinding,
    options: RunInferenceOptions = {},
  ): Promise<MockInferenceOutput> {
    throwIfAborted(options.signal)
    await waitForSchedule(this.scheduler(), options.signal)
    throwIfAborted(options.signal)
    return runMockEngine(binding)
  }
}
