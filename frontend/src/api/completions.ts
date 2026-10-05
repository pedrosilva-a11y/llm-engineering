import type { CompletionChunk, CompletionRequest } from '../types/completion'

type CompletionChunkHandler = (chunk: CompletionChunk) => void

export async function streamCompletion(
  request: CompletionRequest,
  onChunk: CompletionChunkHandler,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch('/v1/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
    signal: signal ?? null,
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(
      `Completion request failed with status ${String(response.status)}: ${message}`,
    )
  }

  if (response.body === null) {
    throw new Error('Completion response does not contain a body.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()

  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()

      if (done) {
        break
      }

      buffer += decoder.decode(value, { stream: true })
      buffer = buffer.replace(/\r\n/g, '\n')

      let separatorIndex = buffer.indexOf('\n\n')

      while (separatorIndex !== -1) {
        const event = buffer.slice(0, separatorIndex)
        buffer = buffer.slice(separatorIndex + 2)

        const data = extractEventData(event)

        if (data === '[DONE]') {
          return
        }

        if (data !== null) {
          const chunk = parseCompletionChunk(data)
          onChunk(chunk)
        }

        separatorIndex = buffer.indexOf('\n\n')
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function extractEventData(event: string): string | null {
  const dataLines = event
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice('data:'.length).trimStart())

  if (dataLines.length === 0) {
    return null
  }

  return dataLines.join('\n')
}

function parseCompletionChunk(data: string): CompletionChunk {
  let parsed: unknown

  try {
    parsed = JSON.parse(data)
  } catch {
    throw new Error('Completion stream returned invalid JSON.')
  }

  if (!isCompletionChunk(parsed)) {
    throw new Error('Completion stream returned an invalid chunk.')
  }

  return parsed
}

function isCompletionChunk(value: unknown): value is CompletionChunk {
  if (!isRecord(value)) {
    return false
  }

  if (
    typeof value.id !== 'string' ||
    typeof value.object !== 'string' ||
    typeof value.model !== 'string' ||
    !Array.isArray(value.choices) ||
    value.choices.length === 0
  ) {
    return false
  }

  return value.choices.every((choice) => {
    if (!isRecord(choice)) {
      return false
    }

    return (
      typeof choice.index === 'number' &&
      typeof choice.cumulative_text === 'string' &&
      typeof choice.token_id === 'number' &&
      (typeof choice.finish_reason === 'string' || choice.finish_reason === null)
    )
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}
