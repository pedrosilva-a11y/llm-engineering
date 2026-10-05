import { useState } from 'react'
import type { SubmitEvent } from 'react'

import { streamCompletion } from './api/completions'

const MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
const MAX_TOKENS = 128

function App() {
  const [prompt, setPrompt] = useState('')
  const [output, setOutput] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)

  function handleSubmit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault()

    if (prompt.trim() === '') {
      return
    }

    void generateCompletion()
  }

  async function generateCompletion(): Promise<void> {
    setOutput('')
    setError(null)
    setIsGenerating(true)

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
      )
    } catch (caughtError: unknown) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'An unexpected error occurred.',
      )
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <main>
      <h1>LLM Inference Engine</h1>

      <form onSubmit={handleSubmit}>
        <textarea
          value={prompt}
          onChange={(event) => {
            setPrompt(event.target.value)
          }}
          placeholder="Enter a prompt"
          rows={6}
        />

        <button type="submit" disabled={isGenerating}>
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
      </form>

      {error !== null && <p>{error}</p>}

      <pre>{output}</pre>
    </main>
  )
}

export default App
