# converge

[![npm](https://img.shields.io/npm/v/@jeremysnr/converge)](https://www.npmjs.com/package/@jeremysnr/converge)
[![MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Every LLM provider uses a different message format. This converts between them. No API calls, no auth, no network. JSON in, JSON out.

```
npm install @jeremysnr/converge
```

## Usage

```ts
import { fromOpenAI, toAnthropic, fromAnthropic, toOpenAI, fromGemini, toGemini } from '@jeremysnr/converge'

// OpenAI → Anthropic
const canonical = fromOpenAI(openaiMessages)
const { system, messages } = toAnthropic(canonical)

// Anthropic → OpenAI
const canonical = fromAnthropic({ system, messages })
const openaiMessages = toOpenAI(canonical)

// OpenAI → Gemini
const { system_instruction, contents } = toGemini(fromOpenAI(openaiMessages))

// Gemini → Anthropic
const { system, messages } = toAnthropic(fromGemini({ system_instruction, contents }))
```

All conversions go through a canonical `Message[]` in the middle. You can inspect or modify it between steps if you need to.

## Supported formats

| Format    | In              | Out             |
|-----------|-----------------|-----------------|
| OpenAI    | `fromOpenAI()`  | `toOpenAI()`    |
| Anthropic | `fromAnthropic()` | `toAnthropic()` |
| Gemini    | `fromGemini()`  | `toGemini()`    |

## Canonical format

```ts
interface Message {
  role:          'system' | 'user' | 'assistant' | 'tool'
  content:       Part[]
  name?:         string        // participant name, or function name on tool messages
  tool_call_id?: string        // present when role === 'tool'
  tool_calls?:   ToolCall[]    // present on assistant messages
  is_error?:     boolean       // present on tool messages
}

type Part =
  | { type: 'text';  text: string }
  | { type: 'image'; mime_type: string; data: string; encoding: 'base64' | 'url' }

interface ToolCall {
  id:     string
  name:   string
  args:   Record<string, unknown>   // always a parsed object, never a JSON string
  index?: number
}
```

## API

### `fromOpenAI(messages: unknown[]): Message[]`

Converts an OpenAI `messages` array to canonical form. Handles the `developer` role (maps to `system`), the deprecated `function` / `function_call` fields, data URI splitting for `image_url`, JSON string parsing for `tool_calls[].function.arguments`, and backwards resolution of tool message names from the preceding assistant turn.

### `toOpenAI(messages: Message[]): unknown[]`

Converts canonical messages to OpenAI format. Re-serialises `tool_calls[].function.arguments` to a JSON string, sets `content: null` on assistant messages that have tool calls, and reconstructs base64 images as data URIs.

### `fromAnthropic(input: AnthropicPayload | unknown[]): Message[]`

Accepts `{ system?, messages }` or a bare messages array. Extracts `tool_result` blocks from user messages into canonical `tool` role messages, maps `tool_use` blocks to `tool_calls`, and preserves `is_error`.

### `toAnthropic(messages: Message[]): AnthropicPayload`

Returns `{ system?, messages }`. Concatenates `system` role messages into the top-level `system` string. Folds consecutive `tool` messages into a single user message with `tool_result` blocks, merging any following user message into the same turn to avoid consecutive user messages, which the Anthropic API rejects.

### `fromGemini(input: GeminiPayload | unknown[]): Message[]`

Accepts `{ system_instruction?, contents }` or a bare contents array. Maps the `model` role to `assistant`, converts `function_call` parts to `tool_calls`, and converts `function_response` parts to `tool` role messages.

### `toGemini(messages: Message[]): GeminiPayload`

Returns `{ system_instruction?, contents }`. Merges consecutive same-role contents (Gemini rejects them), folds consecutive `tool` messages into a single user content with `function_response` parts, and injects a blank text part where needed (Gemini requires at least one part per content).

## Known lossy conversions

| What | What is lost |
|---|---|
| OpenAI `input_audio` parts | Dropped, no audio equivalent in the other formats |
| OpenAI `image_url.detail` | Dropped |
| OpenAI `image_url` with a plain URL | `mime_type` degrades to `image/*`, cannot be derived from a URL |
| OpenAI `refusal` on assistant messages | Dropped |
| OpenAI `name` field when converting to Gemini | Dropped, Gemini has no participant name field |
| Anthropic `cache_control` on blocks | Dropped |
| Anthropic `document` blocks | Dropped |
| Gemini `video_metadata`, `media_resolution` | Dropped |
| Gemini `function_call` IDs | Synthesised on read, original IDs are not preserved |
| Multiple `system` messages in OpenAI | Concatenated into one when converting to Anthropic or Gemini |

## Requirements

Node 18+, Deno, Bun, or any ES2020-capable runtime. No runtime dependencies.
