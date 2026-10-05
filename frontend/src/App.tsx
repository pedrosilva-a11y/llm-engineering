import { useRef, useState } from 'react'
import type { SubmitEvent } from 'react'

import { streamCompletion } from './api/completions'

import './App.css'

const MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
const MAX_TOKENS = 128

function App() {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault()

    if (prompt.trim() === '') {
      return
    }

    void generateCompletion()
  }

  function handleStop() {
    abortControllerRef.current?.abort()
  }

  async function generateCompletion(): Promise<void> {
    setOutput('')
    setError(null)
    setIsGenerating(true)

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      await streamCompletion(
        {
          model: MODEL_NAME,
          prompt,
          max_tokens: MAX_TOKENS,
          stream: true,
        },
        (chunk) => {
          const cumulativeText = chunk.choices[0]?.cumulative_text

          if (cumulativeText !== undefined) {
            setOutput(cumulativeText)
          }
        },
        controller.signal,
      )
    } catch (caughtError: unknown) {
      if (caughtError instanceof DOMException && caughtError.name === 'AbortError') {
        return
      }

      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'An unexpected error occurred.',
      )
    } finally {
      abortControllerRef.current = null
      setIsGenerating(false)
    }
  }

  return (
    <main className="app">
      <header className="app__header">
        <h1 className="app__title">LLM Inference Engine</h1>
      </header>

      <form className="completion-form" onSubmit={handleSubmit}>
        <textarea
          className="completion-form__textarea"
          value={prompt}
          onChange={(event) => {
            setPrompt(event.target.value)
          }}
          placeholder="Enter a prompt"
          rows={6}
        />

        <div className="completion-form__actions">
          <button
            className="completion-form__button"
            type="submit"
            disabled={isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate'}
          </button>

          {isGenerating && (
            <button
              className="completion-form__button completion-form__button--stop"
              type="button"
              onClick={handleStop}
            >
              Stop
            </button>
          )}
        </div>
      </form>

      {error !== null && <p className="app__error">{error}</p>}

      <pre className="completion-output">{output}</pre>
    </main>
  )
}

export default App
