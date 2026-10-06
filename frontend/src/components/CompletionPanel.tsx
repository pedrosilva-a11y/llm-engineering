import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, SubmitEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { streamCompletion } from '../api/completions'

import './CompletionPanel.css'

const MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
const MAX_TOKENS = 128

type RequestState =
  'idle' | 'waiting' | 'generating' | 'completed' | 'cancelled' | 'error'

interface CompletionMetrics {
  tokenCount: number
  clientObservedTtftMs: number | null
  durationMs: number | null
  decodeTokensPerSecond: number | null
}

interface CompletionPanelProps {
  title: string
}

const INITIAL_METRICS: CompletionMetrics = {
  tokenCount: 0,
  clientObservedTtftMs: null,
  durationMs: null,
  decodeTokensPerSecond: null,
}

const REQUEST_STATE_LABELS: Record<RequestState, string> = {
  idle: 'Ready',
  waiting: 'Waiting for first token',
  generating: 'Generating',
  completed: 'Completed',
  cancelled: 'Cancelled',
  error: 'Error',
}

export function CompletionPanel({ title }: CompletionPanelProps) {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [requestState, setRequestState] = useState<RequestState>('idle')
  const [metrics, setMetrics] = useState<CompletionMetrics>(INITIAL_METRICS)

  const requestStartedAtRef = useRef<number | null>(null)
  const firstTokenAtRef = useRef<number | null>(null)
  const lastTokenAtRef = useRef<number | null>(null)
  const tokenCountRef = useRef(0)
  const abortControllerRef = useRef<AbortController | null>(null)
  const outputRef = useRef<HTMLDivElement | null>(null)

  const isGenerating = requestState === 'waiting' || requestState === 'generating'

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

  function finishRequestMetrics(): void {
    const startedAt = requestStartedAtRef.current

    if (startedAt === null) {
      return
    }

    const finishedAt = performance.now()
    const durationMs = finishedAt - startedAt
    const firstTokenAt = firstTokenAtRef.current
    const lastTokenAt = lastTokenAtRef.current
    const tokenCount = tokenCountRef.current

    const decodeTokensPerSecond =
      firstTokenAt !== null &&
      lastTokenAt !== null &&
      tokenCount > 1 &&
      lastTokenAt > firstTokenAt
        ? (tokenCount - 1) / ((lastTokenAt - firstTokenAt) / 1_000)
        : null

    setMetrics((current) => ({
      ...current,
      durationMs,
      decodeTokensPerSecond,
    }))
  }

  async function generateCompletion(): Promise<void> {
    setOutput('')
    setError(null)
    setMetrics(INITIAL_METRICS)
    setRequestState('waiting')

    const controller = new AbortController()

    abortControllerRef.current = controller
    requestStartedAtRef.current = performance.now()
    firstTokenAtRef.current = null
    lastTokenAtRef.current = null
    tokenCountRef.current = 0

    try {
      await streamCompletion(
        {
          model: MODEL_NAME,
          prompt,
          max_tokens: MAX_TOKENS,
          stream: true,
        },
        (chunk) => {
          const now = performance.now()
          const cumulativeText = chunk.choices[0]?.cumulative_text

          lastTokenAtRef.current = now

          if (firstTokenAtRef.current === null) {
            firstTokenAtRef.current = now
            setRequestState('generating')
          }

          tokenCountRef.current += 1

          const startedAt = requestStartedAtRef.current
          const firstTokenAt = firstTokenAtRef.current

          setMetrics((current) => ({
            ...current,
            tokenCount: tokenCountRef.current,
            clientObservedTtftMs:
              current.clientObservedTtftMs ??
              (startedAt !== null ? firstTokenAt - startedAt : null),
          }))

          if (cumulativeText !== undefined) {
            setOutput(cumulativeText)
          }
        },
        controller.signal,
      )

      finishRequestMetrics()
      setRequestState('completed')
    } catch (caughtError: unknown) {
      finishRequestMetrics()

      if (caughtError instanceof DOMException && caughtError.name === 'AbortError') {
        setRequestState('cancelled')
        return
      }

      setRequestState('error')
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'An unexpected error occurred.',
      )
    } finally {
      abortControllerRef.current = null
    }
  }

  return (
    <section className="completion-panel">
      <header className="completion-panel__header">
        <h2 className="completion-panel__title">{title}</h2>

        <span className="completion-panel__stream-label">Independent stream</span>
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

      <div className="completion-telemetry">
        <span
          className={`completion-telemetry__status completion-telemetry__status--${requestState}`}
        >
          {REQUEST_STATE_LABELS[requestState]}
        </span>

        <dl className="completion-telemetry__metrics">
          <div>
            <dt>Tokens</dt>
            <dd>{metrics.tokenCount}</dd>
          </div>

          <div>
            <dt>Client-observed TTFT</dt>
            <dd>{formatMilliseconds(metrics.clientObservedTtftMs)}</dd>
          </div>

          <div>
            <dt>Duration</dt>
            <dd>{formatMilliseconds(metrics.durationMs)}</dd>
          </div>

          <div>
            <dt>Decode tokens/s</dt>
            <dd>
              {metrics.decodeTokensPerSecond === null
                ? '—'
                : metrics.decodeTokensPerSecond.toFixed(1)}
            </dd>
          </div>
        </dl>
      </div>

      <div
        ref={outputRef}
        className="completion-output"
        aria-live="polite"
        aria-busy={isGenerating}
      >
        {output !== '' ? (
          <div className="completion-output__markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{output}</ReactMarkdown>
          </div>
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

function formatMilliseconds(value: number | null): string {
  if (value === null) {
    return '—'
  }

  return `${value.toFixed(value < 100 ? 1 : 0)} ms`
}
