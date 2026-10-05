import type {
  CatalogResponse,
  HardwareCatalogEntry,
  ModelCatalogEntry,
} from '../types/catalog'

export async function getCatalog(): Promise<CatalogResponse> {
  const response = await fetch('/v1/catalog')

  if (!response.ok) {
    throw new Error(`Catalog request failed with status ${String(response.status)}.`)
  }

  const data: unknown = await response.json()

  if (!isCatalogResponse(data)) {
    throw new Error('Catalog endpoint returned an invalid response.')
  }

  return data
}

function isCatalogResponse(value: unknown): value is CatalogResponse {
  if (!isRecord(value)) {
    return false
  }

  return (
    Array.isArray(value.models) &&
    value.models.every(isModelCatalogEntry) &&
    Array.isArray(value.hardware) &&
    value.hardware.every(isHardwareCatalogEntry)
  )
}

function isModelCatalogEntry(value: unknown): value is ModelCatalogEntry {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.name === 'string' &&
    typeof value.n_layer === 'number' &&
    typeof value.d_model === 'number' &&
    typeof value.n_head === 'number' &&
    typeof value.n_kv_head === 'number' &&
    typeof value.d_head === 'number' &&
    typeof value.d_ff === 'number' &&
    typeof value.vocab_size === 'number' &&
    typeof value.tied_embeddings === 'boolean' &&
    typeof value.norm_has_bias === 'boolean'
  )
}

function isHardwareCatalogEntry(value: unknown): value is HardwareCatalogEntry {
  if (!isRecord(value)) {
    return false
  }

  return (
    typeof value.name === 'string' &&
    typeof value.peak_bf16_tflops === 'number' &&
    typeof value.memory_bandwidth_tb_s === 'number' &&
    typeof value.memory_capacity_gib === 'number'
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
