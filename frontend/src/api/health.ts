import type { HealthResponse } from '../types/health'

export async function getBackendHealth(): Promise<HealthResponse> {
  const response = await fetch('/v1/health')

  if (!response.ok) {
    throw new Error(`Health request failed with status ${String(response.status)}`)
  }

  const data: unknown = await response.json()

  if (!isHealthResponse(data)) {
    throw new Error('Health endpoint returned an invalid response.')
  }

  return data
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (!isRecord(value)) {
    return false
  }

  return value.status === 'ok' && typeof value.service === 'string'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
