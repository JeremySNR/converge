// converge — zero-dependency LLM message format conversion
// OpenAI ↔ Anthropic ↔ Gemini

// ─── Canonical types ─────────────────────────────────────────────────────────

export type Role = 'system' | 'user' | 'assistant' | 'tool'

export interface TextPart     { type: 'text';     text: string }
export interface ImagePart    { type: 'image';    mime_type: string; data: string; encoding: 'base64' | 'url' }
export type Part = TextPart | ImagePart

export interface ToolCall { id: string; name: string; args: Record<string, unknown>; index?: number }

/** Canonical message representation. All conversions pass through this. */
export interface Message {
  role:          Role
  content:       Part[]
  name?:         string        // participant name (user/assistant) or function name (tool)
  tool_call_id?: string        // present when role === 'tool'
  tool_calls?:   ToolCall[]    // present on assistant messages
  is_error?:     boolean       // present on tool messages
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

function parseDataUri(url: string): ImagePart | null {
  const m = url.match(/^data:([^;]+);base64,(.+)$/)
  return m ? { type: 'image', mime_type: m[1], data: m[2], encoding: 'base64' } : null
}

function parseArgs(s: string): Record<string, unknown> {
  try { return JSON.parse(s) } catch { return { _raw: s } }
}

function partsToText(parts: Part[]): string {
  return parts.filter((p): p is TextPart => p.type === 'text').map(p => p.text).join('\n')
}

function resolveToolNames(msgs: Message[]): void {
  for (let i = 0; i < msgs.length; i++) {
    if (msgs[i].role !== 'tool' || msgs[i].name || !msgs[i].tool_call_id) continue
    const id = msgs[i].tool_call_id!
    for (let j = i - 1; j >= 0; j--) {
      const match = msgs[j].tool_calls?.find(t => t.id === id)
      if (match) { msgs[i].name = match.name; break }
    }
  }
}

// ─── OpenAI ──────────────────────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Obj = Record<string, any>

function oaiContentToParts(c: unknown): Part[] {
  if (!c) return []
  if (typeof c === 'string') return c ? [{ type: 'text', text: c }] : []
  if (!Array.isArray(c)) return []
  return (c as Obj[]).flatMap((p): Part[] => {
    if (p.type === 'text') return [{ type: 'text', text: p.text as string }]
    if (p.type === 'image_url') {
      const url: string = (p.image_url as Obj).url
      return [parseDataUri(url) ?? { type: 'image', mime_type: 'image/*', data: url, encoding: 'url' }]
    }
    return []
  })
}

export function fromOpenAI(messages: unknown[]): Message[] {
  const out: Message[] = []
  for (const raw of messages as Obj[]) {
    const role: string = raw.role
    // Normalise deprecated roles
    if (role === 'developer') { out.push({ role: 'system', content: oaiContentToParts(raw.content) }); continue }
    if (role === 'function') {
      out.push({ role: 'tool', name: raw.name, tool_call_id: raw.name ?? '', content: raw.content ? [{ type: 'text', text: String(raw.content) }] : [] })
      continue
    }
    const content = oaiContentToParts(raw.content)
    const tool_calls: ToolCall[] = []
    for (const tc of (raw.tool_calls ?? []) as Obj[]) {
      const fn: Obj = tc.function
      tool_calls.push({ id: tc.id, name: fn.name, args: parseArgs(fn.arguments) })
    }
    if (raw.function_call) {
      tool_calls.push({ id: 'legacy', name: raw.function_call.name, args: parseArgs(raw.function_call.arguments) })
    }
    const msg: Message = { role: role as Role, content }
    if (raw.name)         msg.name         = raw.name
    if (raw.tool_call_id) msg.tool_call_id = raw.tool_call_id
    if (tool_calls.length) msg.tool_calls  = tool_calls
    out.push(msg)
  }
  resolveToolNames(out)
  return out
}

function partToOAI(p: Part): unknown {
  if (p.type === 'text')  return { type: 'text', text: p.text }
  return { type: 'image_url', image_url: { url: p.encoding === 'base64' ? `data:${p.mime_type};base64,${p.data}` : p.data } }
}

export function toOpenAI(messages: Message[]): unknown[] {
  return messages.map(m => {
    if (m.role === 'tool') return { role: 'tool', tool_call_id: m.tool_call_id ?? '', content: partsToText(m.content) }
    if (m.role === 'assistant') {
      const out: Obj = { role: 'assistant', content: m.tool_calls?.length ? null : (partsToText(m.content) || null) }
      if (m.tool_calls?.length) out.tool_calls = m.tool_calls.map(tc => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.args) } }))
      if (m.name) out.name = m.name
      return out
    }
    const out: Obj = { role: m.role, content: m.content.length === 1 && m.content[0].type === 'text' ? m.content[0].text : m.content.length === 0 ? '' : m.content.map(partToOAI) }
    if (m.name) out.name = m.name
    return out
  })
}

// ─── Anthropic ───────────────────────────────────────────────────────────────

export interface AnthropicPayload { system?: string; messages: unknown[] }

function anthBlockToPart(b: Obj): Part | null {
  if (b.type === 'text') return { type: 'text', text: b.text }
  if (b.type === 'image') {
    const s: Obj = b.source
    return s.type === 'base64'
      ? { type: 'image', mime_type: s.media_type, data: s.data, encoding: 'base64' }
      : { type: 'image', mime_type: 'image/*',    data: s.url,  encoding: 'url' }
  }
  return null
}

export function fromAnthropic(input: AnthropicPayload | unknown[]): Message[] {
  const msgs = Array.isArray(input) ? input : input.messages
  const sys  = Array.isArray(input) ? undefined : input.system
  const out: Message[] = []
  if (sys) {
    const text = Array.isArray(sys) ? (sys as Obj[]).map(b => b.text as string).join('\n') : String(sys)
    out.push({ role: 'system', content: [{ type: 'text', text }] })
  }
  for (const raw of msgs as Obj[]) {
    const blocks: Obj[] = Array.isArray(raw.content) ? raw.content : [{ type: 'text', text: raw.content }]
    const parts: Part[] = []; const toolCalls: ToolCall[] = []; const toolResults: Message[] = []
    for (const b of blocks) {
      if (b.type === 'tool_result') {
        const content: Part[] = Array.isArray(b.content)
          ? (b.content as Obj[]).flatMap((c): Part[] => {
              const p = anthBlockToPart(c); return p ? [p] : []
            })
          : [{ type: 'text', text: String(b.content ?? '') }]
        toolResults.push({ role: 'tool', tool_call_id: b.tool_use_id, content, ...(b.is_error ? { is_error: true } : {}) })
      } else if (b.type === 'tool_use') {
        toolCalls.push({ id: b.id, name: b.name, args: b.input as Record<string, unknown> })
      } else {
        const p = anthBlockToPart(b); if (p) parts.push(p)
      }
    }
    // tool_results come before the accompanying text in canonical
    out.push(...toolResults)
    if (parts.length || toolCalls.length || !toolResults.length) {
      const msg: Message = { role: raw.role as Role, content: parts }
      if (toolCalls.length) msg.tool_calls = toolCalls
      out.push(msg)
    }
  }
  resolveToolNames(out)
  return out
}

function partToAnthBlock(p: Part): unknown {
  if (p.type === 'text')  return { type: 'text', text: p.text }
  return { type: 'image', source: p.encoding === 'base64' ? { type: 'base64', media_type: p.mime_type, data: p.data } : { type: 'url', url: p.data } }
}

export function toAnthropic(messages: Message[]): AnthropicPayload {
  let system: string | undefined
  const out: unknown[] = []
  let i = 0
  while (i < messages.length) {
    const m = messages[i]
    if (m.role === 'system') { system = (system !== undefined ? system + '\n' : '') + partsToText(m.content); i++; continue }
    if (m.role === 'tool') {
      const blocks: unknown[] = []
      while (i < messages.length && messages[i].role === 'tool') {
        const t = messages[i++]
        const tc = t.content.map(partToAnthBlock)
        blocks.push({ type: 'tool_result', tool_use_id: t.tool_call_id ?? '', content: tc.length === 1 && (tc[0] as Obj).type === 'text' ? (tc[0] as Obj).text : tc, ...(t.is_error ? { is_error: true } : {}) })
      }
      // If the next message is also user, merge it in — Anthropic forbids consecutive user messages
      if (i < messages.length && messages[i].role === 'user') {
        const next = messages[i++]
        blocks.push(...next.content.map(partToAnthBlock))
      }
      out.push({ role: 'user', content: blocks }); continue
    }
    if (m.role === 'assistant' && m.tool_calls?.length) {
      const blocks: unknown[] = []
      const text = partsToText(m.content); if (text) blocks.push({ type: 'text', text })
      for (const tc of m.tool_calls) blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.args })
      out.push({ role: 'assistant', content: blocks }); i++; continue
    }
    const blocks = m.content.map(partToAnthBlock)
    out.push({ role: m.role, content: blocks.length === 1 && (blocks[0] as Obj).type === 'text' ? (blocks[0] as Obj).text : blocks })
    i++
  }
  return system !== undefined ? { system, messages: out } : { messages: out }
}

// ─── Gemini ──────────────────────────────────────────────────────────────────

export interface GeminiPayload { system_instruction?: { parts: unknown[] }; contents: unknown[] }

export function fromGemini(input: GeminiPayload | unknown[]): Message[] {
  const contents = Array.isArray(input) ? input : input.contents
  const sys      = Array.isArray(input) ? undefined : input.system_instruction
  const out: Message[] = []
  if (sys) {
    const text = (sys.parts as Obj[]).map(p => p.text as string).join('\n')
    out.push({ role: 'system', content: [{ type: 'text', text }] })
  }
  for (const c of contents as Obj[]) {
    const role: Role = c.role === 'model' ? 'assistant' : 'user'
    const parts: Part[] = []; const toolCalls: ToolCall[] = []; const toolResults: Message[] = []
    for (const p of c.parts as Obj[]) {
      if ('text' in p && p.text !== '')          parts.push({ type: 'text', text: p.text })
      else if ('inline_data' in p)               parts.push({ type: 'image', mime_type: p.inline_data.mime_type, data: p.inline_data.data, encoding: 'base64' })
      else if ('file_data' in p)                 parts.push({ type: 'image', mime_type: p.file_data.mime_type, data: p.file_data.file_uri, encoding: 'url' })
      else if ('function_call' in p) {
        const fc: Obj = p.function_call
        // Gemini may append thought-signature suffix to IDs — strip it
        const id = String(fc.name).replace(/_\d+$/, '')
        toolCalls.push({ id, name: fc.name, args: fc.args })
      }
      else if ('function_response' in p) {
        const fr: Obj = p.function_response
        // If the response was written by toGemini it has an `output` string key; otherwise serialise the whole object
        const text = typeof fr.response?.output === 'string' ? fr.response.output : JSON.stringify(fr.response)
        toolResults.push({ role: 'tool', name: fr.name, tool_call_id: fr.name, content: [{ type: 'text', text }] })
      }
    }
    out.push(...toolResults)
    if (parts.length || toolCalls.length) {
      const msg: Message = { role, content: parts }
      if (toolCalls.length) msg.tool_calls = toolCalls
      out.push(msg)
    }
  }
  return out
}

export function toGemini(messages: Message[]): GeminiPayload {
  let system_instruction: { parts: unknown[] } | undefined
  const contents: Obj[] = []
  let i = 0
  while (i < messages.length) {
    const m = messages[i]
    if (m.role === 'system') {
      const text = partsToText(m.content)
      system_instruction = system_instruction
        ? { parts: [{ text: (system_instruction.parts[0] as { text: string }).text + '\n' + text }] }
        : { parts: [{ text }] }
      i++; continue
    }
    if (m.role === 'tool') {
      const parts: unknown[] = []
      while (i < messages.length && messages[i].role === 'tool') {
        const t = messages[i++]
        parts.push({ function_response: { name: t.name ?? t.tool_call_id ?? '', response: { output: partsToText(t.content) } } })
      }
      // Merge into previous user content if possible, otherwise push new
      const last = contents[contents.length - 1]
      if (last?.role === 'user') last.parts.push(...parts)
      else contents.push({ role: 'user', parts })
      continue
    }
    const gRole = m.role === 'assistant' ? 'model' : 'user'
    const parts: unknown[] = m.content.map(p => {
      if (p.type === 'text')  return { text: p.text }
      return p.encoding === 'base64' ? { inline_data: { mime_type: p.mime_type, data: p.data } } : { file_data: { mime_type: p.mime_type, file_uri: p.data } }
    })
    if (m.tool_calls?.length) for (const tc of m.tool_calls) parts.push({ function_call: { name: tc.name, args: tc.args } })
    if (parts.length === 0) parts.push({ text: '' }) // Gemini requires at least one part
    // Merge consecutive same-role contents (Gemini forbids consecutive same role)
    const last = contents[contents.length - 1]
    if (last?.role === gRole) last.parts.push(...parts)
    else contents.push({ role: gRole, parts })
    i++
  }
  return system_instruction ? { system_instruction, contents } : { contents }
}
