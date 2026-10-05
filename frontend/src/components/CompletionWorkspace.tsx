import { CompletionPanel } from './CompletionPanel'

import './CompletionWorkspace.css'

export function CompletionWorkspace() {
  return (
    <section className="completion-workspace" aria-labelledby="inference-engine-title">
      <header className="completion-workspace__header">
        <div>
          <p className="completion-workspace__eyebrow">Inference console</p>

          <h1 id="inference-engine-title" className="completion-workspace__title">
            LLM Inference Engine
          </h1>
        </div>

        <p className="completion-workspace__description">
          Submit independent streaming requests against the shared inference engine.
        </p>
      </header>

      <div className="completion-workspace__grid">
        <CompletionPanel title="Request A" />
        <CompletionPanel title="Request B" />
      </div>
    </section>
  )
}
