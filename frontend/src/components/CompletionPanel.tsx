import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, SubmitEvent } from 'react'

import { streamCompletion } from '../api/completions'

import './CompletionPanel.css'

const MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
const MAX_TOKENS = 128

export function CompletionPanel() {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)

  const abortControllerRef = useRef<AbortController | null>(null)
  const outputRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!isGenerating || outputRef.current === null) {
      return
    }

    outputRef.current.scrollTop = outputRef.current.scrollHeight
  }, [output, isGenerating])

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault()

    if (prompt.trim() === '') {
      return
    }

    void generateCompletion()
  }

  function handlePromptKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !isGenerating) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
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
    <section className="completion-panel">
      <header className="completion-panel__header">
        <h1 className="completion-panel__title">LLM Inference Engine</h1>
      </header>

      <form className="completion-form" onSubmit={handleSubmit}>
        <textarea
          className="completion-form__textarea"
          value={prompt}
          onChange={(event) => {
            setPrompt(event.target.value)
          }}
          onKeyDown={handlePromptKeyDown}
          placeholder="Enter a prompt"
          aria-label="Completion prompt"
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

      {error !== null && (
        <p className="completion-panel__error" role="alert">
          {error}
        </p>
      )}

      <div
        ref={outputRef}
        className="completion-output"
        aria-live="polite"
        aria-busy={isGenerating}
      >
        {output !== '' ? (
          <span>{output}</span>
        ) : (
          <span className="completion-output__placeholder">
            {isGenerating
              ? 'Waiting for first token...'
              : 'Generated text will stream here.'}
          </span>
        )}

        {isGenerating && (
          <span className="completion-output__cursor" aria-hidden="true" />
        )}
      </div>
    </section>
  )
}
