import { useEffect, useState } from 'react'

import { getCatalog } from '../api/catalog'
import type { CatalogResponse } from '../types/catalog'

import './ConfigurationSelector.css'

export function ConfigurationSelector() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null)
  const [selectedModelName, setSelectedModelName] = useState('')
  const [selectedHardwareName, setSelectedHardwareName] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    async function loadCatalog(): Promise<void> {
      try {
        const nextCatalog = await getCatalog()

        if (!isMounted) {
          return
        }

        setCatalog(nextCatalog)
        setSelectedModelName(nextCatalog.models[0]?.name ?? '')
        setSelectedHardwareName(nextCatalog.hardware[0]?.name ?? '')
      } catch (caughtError: unknown) {
        if (!isMounted) {
          return
        }

        setError(
          caughtError instanceof Error
            ? caughtError.message
            : 'Unable to load the model and hardware catalog.',
        )
      }
    }

    void loadCatalog()

    return () => {
      isMounted = false
    }
  }, [])

  const selectedModel =
    catalog?.models.find((model) => model.name === selectedModelName) ?? null

  const selectedHardware =
    catalog?.hardware.find((hardware) => hardware.name === selectedHardwareName) ?? null

  if (error !== null) {
    return (
      <section className="configuration-selector" role="alert">
        <p className="configuration-selector__error">{error}</p>
      </section>
    )
  }

  if (catalog === null) {
    return (
      <section className="configuration-selector" aria-busy="true">
        <p className="configuration-selector__loading">
          Loading configuration catalog...
        </p>
      </section>
    )
  }

  return (
    <section className="configuration-selector" aria-labelledby="configuration-title">
      <div className="configuration-selector__header">
        <div>
          <p className="configuration-selector__eyebrow">Configuration</p>
          <h2 id="configuration-title">Target model and hardware</h2>
        </div>

        <p className="configuration-selector__description">
          Select catalog specifications for the target inference configuration.
        </p>
      </div>

      <div className="configuration-selector__controls">
        <label className="configuration-selector__field">
          <span>Model</span>

          <select
            value={selectedModelName}
            onChange={(event) => {
              setSelectedModelName(event.target.value)
            }}
          >
            {catalog.models.map((model) => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>
        </label>

        <label className="configuration-selector__field">
          <span>Hardware</span>

          <select
            value={selectedHardwareName}
            onChange={(event) => {
              setSelectedHardwareName(event.target.value)
            }}
          >
            {catalog.hardware.map((hardware) => (
              <option key={hardware.name} value={hardware.name}>
                {hardware.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="configuration-selector__details">
        {selectedModel !== null && (
          <div className="configuration-selector__card">
            <span className="configuration-selector__card-label">Model</span>
            <strong>{selectedModel.name}</strong>

            <dl>
              <div>
                <dt>Layers</dt>
                <dd>{selectedModel.n_layer}</dd>
              </div>

              <div>
                <dt>Hidden size</dt>
                <dd>{selectedModel.d_model.toLocaleString()}</dd>
              </div>

              <div>
                <dt>Attention heads</dt>
                <dd>{selectedModel.n_head}</dd>
              </div>

              <div>
                <dt>KV heads</dt>
                <dd>{selectedModel.n_kv_head}</dd>
              </div>
            </dl>
          </div>
        )}

        {selectedHardware !== null && (
          <div className="configuration-selector__card">
            <span className="configuration-selector__card-label">Hardware</span>
            <strong>{selectedHardware.name}</strong>

            <dl>
              <div>
                <dt>BF16</dt>
                <dd>{selectedHardware.peak_bf16_tflops} TFLOPS</dd>
              </div>

              <div>
                <dt>Bandwidth</dt>
                <dd>{selectedHardware.memory_bandwidth_tb_s} TB/s</dd>
              </div>

              <div>
                <dt>Memory</dt>
                <dd>{selectedHardware.memory_capacity_gib} GiB</dd>
              </div>
            </dl>
          </div>
        )}
      </div>
    </section>
  )
}
