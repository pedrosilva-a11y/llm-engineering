export interface CompletionRequest {
  model: string
  prompt: string
  max_tokens: number
  stream: true
}

export interface CompletionChoice {
  index: number
  cumulative_text: string
  token_id: number
  finish_reason: string | null
}

export interface CompletionChunk {
  id: string
  object: string
  model: string
  choices: CompletionChoice[]
}
