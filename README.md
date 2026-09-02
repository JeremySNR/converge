# converge

[![npm](https://img.shields.io/npm/v/@jeremysnr/converge)](https://www.npmjs.com/package/@jeremysnr/converge)
[![CI](https://github.com/JeremySNR/converge/actions/workflows/ci.yml/badge.svg)](https://github.com/JeremySNR/converge/actions/workflows/ci.yml)
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

// OpenAI → Gemini (REST shape, snake_case)
const { system_instruction, contents } = toGemini(fromOpenAI(openaiMessages))

// OpenAI → Gemini (@google/genai SDK shape, camelCase)
const { systemInstruction, contents } = toGemini(fromOpenAI(openaiMessages), { casing: 'camel' })

// Gemini → Anthropic (either casing is accepted on input)
const { system, messages } = toAnthropic(fromGemini({ systemInstruction, contents }))
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

Converts an OpenAI Chat Completions `messages` array to canonical form. Handles the `developer` role (maps to `system`), the deprecated `function` / `function_call` fields, data URI splitting for `image_url`, JSON string parsing for `tool_calls[].function.arguments`, and backwards resolution of tool message names from the preceding assistant turn.

### `toOpenAI(messages: Message[]): unknown[]`

Converts canonical messages to OpenAI Chat Completions format. Re-serialises `tool_calls[].function.arguments` to a JSON string, sets `content: null` on assistant messages that have tool calls, and reconstructs base64 images as data URIs.

### `fromAnthropic(input: AnthropicPayload | unknown[]): Message[]`

Accepts `{ system?, messages }` or a bare messages array. Extracts `tool_result` blocks from user messages into canonical `tool` role messages, maps `tool_use` blocks to `tool_calls`, and preserves `is_error`.

### `toAnthropic(messages: Message[]): AnthropicPayload`

Returns `{ system?, messages }`. Concatenates `system` role messages into the top-level `system` string. Folds consecutive `tool` messages into a single user message with `tool_result` blocks, merging any following user message into the same turn to avoid consecutive user messages, which the Anthropic API rejects.

### `fromGemini(input: GeminiPayload | unknown[]): Message[]`

Accepts `{ system_instruction?, contents }`, `{ systemInstruction?, contents }`, or a bare contents array. Maps the `model` role to `assistant`, converts `function_call` parts to `tool_calls`, and converts `function_response` parts to `tool` role messages.

Both Gemini spellings are accepted on input, and can be mixed within one payload:

| REST API (snake_case)          | @google/genai SDK (camelCase) |
|--------------------------------|-------------------------------|
| `system_instruction`           | `systemInstruction`           |
| `inline_data` / `mime_type`    | `inlineData` / `mimeType`     |
| `file_data` / `file_uri`       | `fileData` / `fileUri`        |
| `function_call`                | `functionCall`                |
| `function_response`            | `functionResponse`            |

### `toGemini(messages: Message[], options?: { casing?: 'snake' | 'camel' }): GeminiPayload`

Returns `{ system_instruction?, contents }` by default, or `{ systemInstruction?, contents }` with `{ casing: 'camel' }`. Use `'snake'` (the default) for the REST API and `'camel'` for the `@google/genai` JavaScript SDK. Merges consecutive same-role contents (Gemini rejects them), folds consecutive `tool` messages into a single user content with `function_response` parts, and injects a blank text part where needed (Gemini requires at least one part per content).

### Gemini function call ids

Gemini 2.x supports an optional `id` on `function_call` and `function_response` parts. `fromGemini` uses that `id` as the canonical `ToolCall.id` and `tool_call_id`, falling back to the function name when the payload has no `id`. `toGemini` writes the canonical id back as `id` on both parts whenever one is present. This means two parallel calls to the same function keep distinct ids across an OpenAI → Gemini → OpenAI round-trip, and each tool result is matched to the right call. If you are targeting a Gemini model that predates the `id` field, strip it from the output before sending.

## Known lossy conversions

| What | What is lost |
|---|---|
| OpenAI `input_audio` parts | Dropped, no audio equivalent in the other formats |
| OpenAI `image_url.detail` | Dropped |
| OpenAI `image_url` with a plain URL | `mime_type` degrades to `image/*`, cannot be derived from a URL |
| OpenAI `refusal` on assistant messages | Dropped |
| OpenAI `name` field when converting to Gemini | Dropped, Gemini has no participant name field |
| OpenAI Responses API format | Not supported. Only the Chat Completions `messages` shape is handled; `input` / `output` item arrays are not recognised |
| Anthropic `cache_control` on blocks | Dropped |
| Anthropic `document` blocks | Dropped |
| Anthropic `thinking` / `redacted_thinking` blocks | Dropped. An extended-thinking conversation that includes tool use will be rejected by the Anthropic API on replay, because the thinking block that preceded the `tool_use` is no longer present |
| Gemini `video_metadata`, `media_resolution` | Dropped |
| Gemini `function_call` ids | Preserved when the input carries `id` (Gemini 2.x). Payloads without an `id` fall back to the function name, so parallel calls to the same function cannot be told apart |
| Multiple `system` messages in OpenAI | Concatenated into one when converting to Anthropic or Gemini |

## Requirements

Node 18+, Deno, Bun, or any ES2020-capable runtime. No runtime dependencies.
