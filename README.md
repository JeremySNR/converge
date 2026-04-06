# converge

[![npm](https://img.shields.io/npm/v/@jeremysnr/converge)](https://www.npmjs.com/package/@jeremysnr/converge)
[![MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Zero-dependency LLM message format conversion. Converts message arrays between OpenAI, Anthropic, and Google Gemini formats. Pure data transformation — no API calls, no authentication, no network.

```
npm install @jeremysnr/converge
```

## Usage

```ts
import { fromOpenAI, toAnthropic, fromAnthropic, toOpenAI, fromGemini, toGemini } from 'converge'

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

All conversions pass through a canonical `Message[]` representation. You can inspect or modify it between conversions.

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
  | { type: 'text';     text: string }
  | { type: 'image';    mime_type: string; data: string; encoding: 'base64' | 'url' }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }

interface ToolCall {
  id:     string
  name:   string
  args:   Record<string, unknown>   // always a parsed object
  index?: number
}
```

## API

### `fromOpenAI(messages: unknown[]): Message[]`

Converts an OpenAI `messages` array to canonical form. Handles:
- `developer` role (normalised to `system`)
- Deprecated `function` / `function_call` role and field
- `image_url` with data URIs (`data:image/png;base64,...`) — split into `mime_type` + `data`
- `tool_calls[].function.arguments` as a JSON string — parsed to an object
- Tool message `name` resolved by scanning backwards to the matching `tool_calls` entry

### `toOpenAI(messages: Message[]): unknown[]`

Converts canonical messages to OpenAI format.
- `tool_calls[].function.arguments` serialised back to a JSON string
- Assistant messages with `tool_calls` get `content: null`
- Image parts reconstructed as `data:mime;base64,...` data URIs

### `fromAnthropic(input: AnthropicPayload | unknown[]): Message[]`

Accepts either `{ system?, messages }` or a bare messages array.
- `tool_result` blocks inside `user` messages are extracted as canonical `tool` role messages
- `tool_use` blocks become `tool_calls` on the `assistant` message
- `is_error` on `tool_result` preserved

### `toAnthropic(messages: Message[]): AnthropicPayload`

Returns `{ system?, messages }`.
- `system` role messages concatenated into the top-level `system` string
- Consecutive `tool` role messages folded into one `user` message with `tool_result` blocks
- `tool_calls` serialised as `tool_use` blocks

### `fromGemini(input: GeminiPayload | unknown[]): Message[]`

Accepts either `{ system_instruction?, contents }` or a bare contents array.
- `model` role normalised to `assistant`
- `function_call` parts become `tool_calls`; `function_response` parts become `tool` role messages

### `toGemini(messages: Message[]): GeminiPayload`

Returns `{ system_instruction?, contents }`.
- `system` role extracted to `system_instruction`
- Consecutive same-role contents merged (Gemini forbids consecutive same-role messages)
- Consecutive `tool` messages folded into one `user` content with `function_response` parts
- Empty content gets a `{ text: '' }` placeholder (Gemini requires at least one part)

## Known lossy conversions

| Conversion | What is lost |
|---|---|
| OpenAI `input_audio` parts | Dropped (no audio equivalent in Anthropic/Gemini canonical) |
| OpenAI `image_url.detail` | Dropped |
| Anthropic `cache_control` on blocks | Dropped |
| Anthropic `document` blocks | Dropped |
| Gemini `video_metadata`, `media_resolution` | Dropped |
| OpenAI `refusal` field on assistant | Dropped |
| OpenAI `name` on participant when converting to Gemini | Dropped (Gemini has no `name` field) |
| Gemini `function_call` IDs | Synthesised as `name_index`; original IDs not preserved |
| Mid-conversation `system` messages (OpenAI) | Multiple system messages concatenated into one when converting to Anthropic or Gemini |

## Runtime requirements

Node 18+, Deno, Bun, or any edge runtime with ES2020 support.

## Zero runtime dependencies

The package has no runtime dependencies. `devDependencies` are used only for building and testing.
