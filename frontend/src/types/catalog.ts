export interface ModelCatalogEntry {
  name: string
  n_layer: number
  d_model: number
  n_head: number
  n_kv_head: number
  d_head: number
  d_ff: number
  vocab_size: number
  tied_embeddings: boolean
  norm_has_bias: boolean
}

export interface HardwareCatalogEntry {
  name: string
  peak_bf16_tflops: number
  memory_bandwidth_tb_s: number
  memory_capacity_gib: number
}

export interface CatalogResponse {
  models: ModelCatalogEntry[]
  hardware: HardwareCatalogEntry[]
}
