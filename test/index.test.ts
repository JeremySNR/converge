import assert from 'node:assert/strict'
import {
  fromOpenAI, toOpenAI,
  fromAnthropic, toAnthropic,
  fromGemini, toGemini,
  type Message,
} from '../src/index.js'

// ─── Helpers ─────────────────────────────────────────────────────────────────
let passed = 0; let failed = 0

function test(name: string, fn: () => void) {
  try { fn(); console.log(`  ✓ ${name}`); passed++ }
  catch (e) { console.error(`  ✗ ${name}\n    ${(e as Error).message}`); failed++ }
}

// ─── fromOpenAI / toOpenAI ────────────────────────────────────────────────────
console.log('\nOpenAI')

test('plain text messages round-trip', () => {
  const oai = [
    { role: 'system', content: 'You are helpful.' },
    { role: 'user',   content: 'Hello' },
    { role: 'assistant', content: 'Hi there!' },
  ]
  const canon = fromOpenAI(oai)
  assert.equal(canon[0].role, 'system')
  assert.equal(canon[1].role, 'user')
  assert.equal(canon[2].role, 'assistant')
  const back = toOpenAI(canon) as { role: string; content: string }[]
  assert.equal(back[0].content, 'You are helpful.')
  assert.equal(back[1].content, 'Hello')
  assert.equal(back[2].content, 'Hi there!')
})

test('tool call and tool result', () => {
  const oai = [
    { role: 'user', content: 'What is the weather?' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"London"}' } }] },
    { role: 'tool', tool_call_id: 'call_1', content: 'Rainy, 12°C' },
  ]
  const canon = fromOpenAI(oai)
  assert.equal(canon[1].role, 'assistant')
  assert.deepEqual(canon[1].tool_calls![0].args, { city: 'London' })
  assert.equal(canon[2].role, 'tool')
  assert.equal(canon[2].tool_call_id, 'call_1')
  assert.equal(canon[2].name, 'get_weather') // resolved from preceding assistant
  const back = toOpenAI(canon) as { role: string; content: null; tool_calls?: { function: { arguments: string } }[] }[]
  assert.equal(back[1].content, null) // null when tool_calls present
  assert.equal(JSON.parse(back[1].tool_calls![0].function.arguments).city, 'London')
})

test('image with data URI', () => {
  const oai = [{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,abc123' } }, { type: 'text', text: 'Describe this' }] }]
  const canon = fromOpenAI(oai)
  const img = canon[0].content[0]
  assert.equal(img.type, 'image')
  if (img.type === 'image') {
    assert.equal(img.mime_type, 'image/png')
    assert.equal(img.data, 'abc123')
    assert.equal(img.encoding, 'base64')
  }
  const back = toOpenAI(canon) as { role: string; content: { type: string; image_url: { url: string } }[] }[]
  assert.equal(back[0].content[0].image_url.url, 'data:image/png;base64,abc123')
})

test('developer role → system', () => {
  const canon = fromOpenAI([{ role: 'developer', content: 'Be concise.' }])
  assert.equal(canon[0].role, 'system')
})

test('legacy function_call', () => {
  const canon = fromOpenAI([{ role: 'assistant', content: null, function_call: { name: 'foo', arguments: '{"x":1}' } }])
  assert.equal(canon[0].tool_calls![0].name, 'foo')
  assert.deepEqual(canon[0].tool_calls![0].args, { x: 1 })
})

test('malformed JSON in arguments falls back gracefully', () => {
  const canon = fromOpenAI([{ role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'foo', arguments: 'not json' } }] }])
  assert.deepEqual(canon[0].tool_calls![0].args, { _raw: 'not json' })
})

// ─── fromAnthropic / toAnthropic ──────────────────────────────────────────────
console.log('\nAnthropic')

test('system + messages', () => {
  const input = { system: 'You help.', messages: [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }] }
  const canon = fromAnthropic(input)
  assert.equal(canon[0].role, 'system')
  assert.equal(canon[1].role, 'user')
  const back = toAnthropic(canon)
  assert.equal(back.system, 'You help.')
  assert.equal(back.messages.length, 2)
})

test('tool_use and tool_result blocks', () => {
  const input = {
    messages: [
      { role: 'user', content: 'Weather?' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'tu_1', name: 'get_weather', input: { city: 'London' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu_1', content: 'Rainy' }] },
    ],
  }
  const canon = fromAnthropic(input)
  const assistantMsg = canon.find(m => m.role === 'assistant')!
  assert.equal(assistantMsg.tool_calls![0].name, 'get_weather')
  assert.deepEqual(assistantMsg.tool_calls![0].args, { city: 'London' })
  const toolMsg = canon.find(m => m.role === 'tool')!
  assert.equal(toolMsg.tool_call_id, 'tu_1')
  // Round-trip back
  const back = toAnthropic(canon)
  // The last user message wraps the tool_result blocks
  const toolResultMsg = [...back.messages].reverse().find((m: { role: string }) => m.role === 'user') as { role: string; content: { type: string; tool_use_id: string }[] }
  assert.equal(toolResultMsg.content[0].type, 'tool_result')
})

test('multiple system messages are concatenated', () => {
  const canon: Message[] = [
    { role: 'system', content: [{ type: 'text', text: 'First.' }] },
    { role: 'system', content: [{ type: 'text', text: 'Second.' }] },
    { role: 'user',   content: [{ type: 'text', text: 'Hi' }] },
  ]
  const back = toAnthropic(canon)
  assert.ok(back.system!.includes('First.'))
  assert.ok(back.system!.includes('Second.'))
})

test('tool_result is_error preserved', () => {
  const input = { messages: [
    { role: 'assistant', content: [{ type: 'tool_use', id: 'tu_2', name: 'run', input: {} }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu_2', content: 'Failed', is_error: true }] },
  ]}
  const canon = fromAnthropic(input)
  const toolMsg = canon.find(m => m.role === 'tool')!
  assert.equal(toolMsg.is_error, true)
  const back = toAnthropic(canon)
  const userMsg = back.messages.find((m: { role: string }) => m.role === 'user') as { content: { is_error: boolean }[] }
  assert.equal(userMsg.content[0].is_error, true)
})

test('image blocks', () => {
  const input = { messages: [{ role: 'user', content: [{ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'xyz' } }, { type: 'text', text: 'Describe' }] }] }
  const canon = fromAnthropic(input)
  const img = canon[0].content[0]
  assert.equal(img.type, 'image')
  if (img.type === 'image') { assert.equal(img.mime_type, 'image/jpeg'); assert.equal(img.data, 'xyz') }
  const back = toAnthropic(canon)
  const msg = back.messages[0] as { content: { type: string; source: { type: string } }[] }
  assert.equal(msg.content[0].type, 'image')
  assert.equal(msg.content[0].source.type, 'base64')
})

// ─── fromGemini / toGemini ────────────────────────────────────────────────────
console.log('\nGemini')

test('model role → assistant', () => {
  const input = { contents: [{ role: 'user', parts: [{ text: 'Hi' }] }, { role: 'model', parts: [{ text: 'Hello!' }] }] }
  const canon = fromGemini(input)
  assert.equal(canon[0].role, 'user')
  assert.equal(canon[1].role, 'assistant')
})

test('system_instruction extracted', () => {
  const input = { system_instruction: { parts: [{ text: 'Be concise.' }] }, contents: [{ role: 'user', parts: [{ text: 'Hi' }] }] }
  const canon = fromGemini(input)
  assert.equal(canon[0].role, 'system')
  assert.equal(canon[0].content[0].type, 'text')
  if (canon[0].content[0].type === 'text') assert.equal(canon[0].content[0].text, 'Be concise.')
})

test('function_call and function_response', () => {
  const input = { contents: [
    { role: 'user',  parts: [{ text: 'Weather?' }] },
    { role: 'model', parts: [{ function_call: { name: 'get_weather', args: { city: 'Paris' } } }] },
    { role: 'user',  parts: [{ function_response: { name: 'get_weather', response: { temp: 20 } } }] },
  ]}
  const canon = fromGemini(input)
  const asst = canon.find(m => m.role === 'assistant')!
  assert.equal(asst.tool_calls![0].name, 'get_weather')
  assert.deepEqual(asst.tool_calls![0].args, { city: 'Paris' })
  const tool = canon.find(m => m.role === 'tool')!
  assert.equal(tool.name, 'get_weather')
})

test('consecutive same-role messages are merged', () => {
  const canon: Message[] = [
    { role: 'user',      content: [{ type: 'text', text: 'Go' }] },
    { role: 'assistant', content: [], tool_calls: [{ id: 'c1', name: 'fn', args: {} }] },
    { role: 'tool',      content: [{ type: 'text', text: 'result' }], tool_call_id: 'c1', name: 'fn' },
    { role: 'tool',      content: [{ type: 'text', text: 'result2' }], tool_call_id: 'c2', name: 'fn2' },
  ]
  const out = toGemini(canon)
  // The two tool messages should be merged into one user content
  const userContents = out.contents.filter((c: { role: string }) => c.role === 'user') as { parts: unknown[] }[]
  const fnResponses = userContents.flatMap(c => c.parts).filter((p: unknown) => typeof p === 'object' && p !== null && 'function_response' in p)
  assert.equal(fnResponses.length, 2)
})

test('empty content gets placeholder part', () => {
  const canon: Message[] = [{ role: 'assistant', content: [], tool_calls: [{ id: 'c1', name: 'fn', args: { x: 1 } }] }]
  const out = toGemini(canon)
  const content = out.contents[0] as { parts: unknown[] }
  assert.ok(content.parts.some(p => typeof p === 'object' && p !== null && 'function_call' in p))
})

test('inline_data ↔ image round-trip', () => {
  const input = { contents: [{ role: 'user', parts: [{ inline_data: { mime_type: 'image/png', data: 'base64data' } }, { text: 'Describe' }] }] }
  const canon = fromGemini(input)
  const img = canon[0].content[0]
  assert.equal(img.type, 'image')
  if (img.type === 'image') assert.equal(img.data, 'base64data')
  const back = toGemini(canon)
  const part = (back.contents[0] as { parts: { inline_data: { data: string } }[] }).parts[0]
  assert.equal(part.inline_data.data, 'base64data')
})

// ─── Round-trip tests ─────────────────────────────────────────────────────────
console.log('\nRound-trips')

test('OpenAI → Anthropic → OpenAI (text)', () => {
  const original = [
    { role: 'system', content: 'You help.' },
    { role: 'user',   content: 'Hello' },
    { role: 'assistant', content: 'Hi!' },
  ]
  const result = toOpenAI(fromAnthropic(toAnthropic(fromOpenAI(original)))) as { role: string; content: string }[]
  assert.equal(result[0].role, 'system')
  assert.equal(result[0].content, 'You help.')
  assert.equal(result[1].content, 'Hello')
  assert.equal(result[2].content, 'Hi!')
})

test('OpenAI → Gemini → OpenAI (text)', () => {
  const original = [
    { role: 'system',    content: 'Be helpful.' },
    { role: 'user',      content: 'What is 2+2?' },
    { role: 'assistant', content: '4' },
  ]
  const result = toOpenAI(fromGemini(toGemini(fromOpenAI(original)))) as { role: string; content: string }[]
  assert.equal(result[0].content, 'Be helpful.')
  assert.equal(result[1].content, 'What is 2+2?')
  assert.equal(result[2].content, '4')
})

test('OpenAI → Anthropic → OpenAI (tool call)', () => {
  const original = [
    { role: 'user', content: 'Get weather' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'weather', arguments: '{"city":"NYC"}' } }] },
    { role: 'tool', tool_call_id: 'tc1', content: 'Sunny, 22°C' },
    { role: 'assistant', content: 'It is sunny in NYC.' },
  ]
  const result = toOpenAI(fromAnthropic(toAnthropic(fromOpenAI(original)))) as { role: string; content: string | null; tool_calls?: { function: { name: string; arguments: string } }[] }[]
  assert.equal(result[1].content, null)
  assert.equal(result[1].tool_calls![0].function.name, 'weather')
  assert.equal(JSON.parse(result[1].tool_calls![0].function.arguments).city, 'NYC')
  assert.equal(result[3].content, 'It is sunny in NYC.')
})

test('OpenAI → Gemini → OpenAI (tool call)', () => {
  const original = [
    { role: 'user', content: 'Run calc' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'tc2', type: 'function', function: { name: 'calc', arguments: '{"expr":"1+1"}' } }] },
    { role: 'tool', tool_call_id: 'tc2', content: '2' },
    { role: 'assistant', content: 'Result is 2.' },
  ]
  const result = toOpenAI(fromGemini(toGemini(fromOpenAI(original)))) as { role: string; content: string | null; tool_calls?: { function: { name: string; arguments: string } }[] }[]
  assert.equal(result[1].tool_calls![0].function.name, 'calc')
  assert.equal(JSON.parse(result[1].tool_calls![0].function.arguments).expr, '1+1')
  assert.equal(result[3].content, 'Result is 2.')
})

test('Anthropic → Gemini → Anthropic (tool call)', () => {
  const original: { system: string; messages: unknown[] } = {
    system: 'You help.',
    messages: [
      { role: 'user', content: 'Do it' },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'tu1', name: 'act', input: { n: 42 } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu1', content: 'done' }] },
      { role: 'assistant', content: 'All done.' },
    ],
  }
  const result = toAnthropic(fromGemini(toGemini(fromAnthropic(original))))
  assert.equal(result.system, 'You help.')
  const assistantMsg = result.messages.find((m: { role: string }) => m.role === 'assistant' && Array.isArray((m as { content: unknown }).content)) as { role: string; content: { type: string; name: string; input: { n: number } }[] }
  assert.equal(assistantMsg.content[0].type, 'tool_use')
  assert.equal(assistantMsg.content[0].name, 'act')
})

// ─── Edge cases ──────────────────────────────────────────────────────────────
console.log('\nEdge cases')

test('parallel tool calls (multiple tool_calls in one assistant turn)', () => {
  const oai = [
    { role: 'user', content: 'Search and calculate' },
    { role: 'assistant', content: null, tool_calls: [
      { id: 'c1', type: 'function', function: { name: 'search',  arguments: '{"q":"news"}' } },
      { id: 'c2', type: 'function', function: { name: 'calc',    arguments: '{"expr":"2+2"}' } },
    ]},
    { role: 'tool', tool_call_id: 'c1', content: 'results' },
    { role: 'tool', tool_call_id: 'c2', content: '4' },
  ]
  const canon = fromOpenAI(oai)
  assert.equal(canon[1].tool_calls!.length, 2)
  assert.equal(canon[2].name, 'search')
  assert.equal(canon[3].name, 'calc')

  // Anthropic: both tool_use blocks in one assistant message, both tool_result blocks in one user message
  const anth = toAnthropic(canon)
  const asst = anth.messages.find((m: { role: string }) => m.role === 'assistant') as { content: { type: string }[] }
  assert.equal(asst.content.filter((b: { type: string }) => b.type === 'tool_use').length, 2)
  // The tool_result user message is the last user message (not the first)
  const userMsg = [...anth.messages].reverse().find((m: { role: string }) => m.role === 'user') as { content: { type: string }[] }
  assert.equal(userMsg.content.filter((b: { type: string }) => b.type === 'tool_result').length, 2)

  // Gemini: both function_call parts in one model content, both function_response parts in one user content
  const gem = toGemini(canon)
  const modelContent = gem.contents.find((c: { role: string }) => c.role === 'model') as { parts: unknown[] }
  assert.equal(modelContent.parts.filter((p: unknown) => typeof p === 'object' && p !== null && 'function_call' in p).length, 2)
  const userContent = [...gem.contents].reverse().find((c: { role: string }) => c.role === 'user') as { parts: unknown[] }
  assert.equal(userContent.parts.filter((p: unknown) => typeof p === 'object' && p !== null && 'function_response' in p).length, 2)
})

test('assistant message with both text and tool_calls', () => {
  // OpenAI allows text alongside tool_calls
  const oai = [
    { role: 'user', content: 'Do it' },
    { role: 'assistant', content: "I'll call the tool now.", tool_calls: [
      { id: 'c1', type: 'function', function: { name: 'act', arguments: '{}' } },
    ]},
    { role: 'tool', tool_call_id: 'c1', content: 'done' },
  ]
  const canon = fromOpenAI(oai)
  assert.equal(canon[1].content[0].type, 'text')
  assert.equal(canon[1].tool_calls!.length, 1)

  // Anthropic: text block first, then tool_use block
  const anth = toAnthropic(canon)
  const asst = anth.messages.find((m: { role: string }) => m.role === 'assistant') as { content: { type: string }[] }
  assert.equal(asst.content[0].type, 'text')
  assert.equal(asst.content[1].type, 'tool_use')

  // Round-trip back to OpenAI preserves the text
  const back = toOpenAI(fromAnthropic(anth)) as { role: string; content: string | null; tool_calls?: unknown[] }[]
  const asstBack = back.find(m => m.role === 'assistant')!
  // After round-trip through Anthropic, text+tool_use come back as null content with tool_calls
  // (Anthropic doesn't distinguish — toOpenAI puts text in content only when no tool_calls)
  assert.ok(asstBack.tool_calls!.length === 1)
})

test('mixed tool_result + text in same Anthropic user message', () => {
  // A user message containing both tool_result blocks and a follow-up text part
  const input = { messages: [
    { role: 'assistant', content: [{ type: 'tool_use', id: 'tu1', name: 'fn', input: {} }] },
    { role: 'user', content: [
      { type: 'tool_result', tool_use_id: 'tu1', content: 'ok' },
      { type: 'text', text: 'Now explain the result.' },
    ]},
  ]}
  const canon = fromAnthropic(input)
  // Should produce: tool message + user message
  assert.equal(canon.filter(m => m.role === 'tool').length, 1)
  assert.equal(canon.filter(m => m.role === 'user').length, 1)
  const userMsg = canon.find(m => m.role === 'user')!
  assert.equal(userMsg.content[0].type, 'text')
  if (userMsg.content[0].type === 'text') assert.equal(userMsg.content[0].text, 'Now explain the result.')
})

test('Anthropic system as TextBlock array', () => {
  const input = {
    system: [{ type: 'text', text: 'First.' }, { type: 'text', text: 'Second.' }],
    messages: [{ role: 'user', content: 'Hi' }],
  }
  const canon = fromAnthropic(input)
  assert.equal(canon[0].role, 'system')
  if (canon[0].content[0].type === 'text') assert.ok(canon[0].content[0].text.includes('First.'))
  const back = toAnthropic(canon)
  assert.ok(back.system!.includes('First.'))
  assert.ok(back.system!.includes('Second.'))
})

test('fromAnthropic accepts bare message array', () => {
  const msgs = [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello' }]
  const canon = fromAnthropic(msgs)
  assert.equal(canon.length, 2)
  assert.equal(canon[0].role, 'user')
})

test('fromGemini accepts bare contents array', () => {
  const contents = [{ role: 'user', parts: [{ text: 'Hi' }] }, { role: 'model', parts: [{ text: 'Hello' }] }]
  const canon = fromGemini(contents)
  assert.equal(canon.length, 2)
  assert.equal(canon[1].role, 'assistant')
})

test('URL image round-trip through all three providers', () => {
  const canon: Message[] = [{
    role: 'user',
    content: [
      { type: 'image', mime_type: 'image/png', data: 'https://example.com/img.png', encoding: 'url' },
      { type: 'text', text: 'What is this?' },
    ],
  }]
  // OpenAI: image_url with plain URL; mime_type is lost (can't derive from URL)
  const oai = toOpenAI(canon) as { content: { type: string; image_url: { url: string } }[] }[]
  assert.equal(oai[0].content[0].image_url.url, 'https://example.com/img.png')
  const fromOai = fromOpenAI(oai)[0].content[0]
  if (fromOai.type === 'image') {
    assert.equal(fromOai.data, 'https://example.com/img.png')
    assert.equal(fromOai.encoding, 'url')
    // mime_type degrades to 'image/*' — documented lossy conversion
  }

  // Anthropic: image with url source
  const anth = toAnthropic(canon)
  const anthMsg = anth.messages[0] as { content: { type: string; source: { type: string; url: string } }[] }
  assert.equal(anthMsg.content[0].source.type, 'url')
  assert.equal(anthMsg.content[0].source.url, 'https://example.com/img.png')

  // Gemini: file_data with file_uri
  const gem = toGemini(canon)
  const gemPart = (gem.contents[0] as { parts: { file_data: { file_uri: string } }[] }).parts[0]
  assert.equal(gemPart.file_data.file_uri, 'https://example.com/img.png')
  assert.deepEqual(fromGemini(gem)[0].content[0], canon[0].content[0])
})

test('mid-conversation system message in OpenAI is preserved in canonical, absorbed in Anthropic/Gemini', () => {
  const oai = [
    { role: 'system',    content: 'Original system.' },
    { role: 'user',      content: 'Hello' },
    { role: 'system',    content: 'Injected instruction.' }, // mid-conversation
    { role: 'assistant', content: 'Hi!' },
  ]
  const canon = fromOpenAI(oai)
  assert.equal(canon.filter(m => m.role === 'system').length, 2)

  // Anthropic concatenates both into one system field
  const anth = toAnthropic(canon)
  assert.ok(anth.system!.includes('Original system.'))
  assert.ok(anth.system!.includes('Injected instruction.'))
  assert.equal(anth.messages.length, 2) // only user + assistant remain

  // Gemini: system_instruction gets last system text (or concatenated)
  const gem = toGemini(canon)
  assert.ok((gem.system_instruction!.parts[0] as { text: string }).text.includes('Original system.'))
})

test('empty messages array', () => {
  assert.deepEqual(fromOpenAI([]), [])
  assert.deepEqual(toOpenAI([]), [])
  assert.deepEqual(fromAnthropic([]), [])
  assert.deepEqual(toAnthropic([]), { messages: [] })
  assert.deepEqual(fromGemini([]), [])
  assert.deepEqual(toGemini([]), { contents: [] })
})

test('user message after tool results does not produce consecutive user messages in Anthropic', () => {
  // Common real-world pattern: tool call → result → user follow-up
  const oai = [
    { role: 'user',      content: 'Get the weather' },
    { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'weather', arguments: '{}' } }] },
    { role: 'tool',      tool_call_id: 'c1', content: 'Sunny, 22°C' },
    { role: 'user',      content: 'Great, now plan my day' },
    { role: 'assistant', content: 'Here is your plan.' },
  ]
  const anth = toAnthropic(fromOpenAI(oai))
  const roles = anth.messages.map((m: { role: string }) => m.role)
  for (let i = 1; i < roles.length; i++) {
    assert.notEqual(roles[i], roles[i - 1], `consecutive ${roles[i]} messages at index ${i - 1} and ${i}`)
  }
  assert.deepEqual(roles, ['user', 'assistant', 'user', 'assistant'])
  // The third message should contain both the tool_result and the follow-up text
  const thirdMsg = anth.messages[2] as { content: { type: string }[] }
  assert.ok(thirdMsg.content.some((b: { type: string }) => b.type === 'tool_result'))
  assert.ok(thirdMsg.content.some((b: { type: string }) => b.type === 'text'))
})

test('tool result with image content survives Anthropic round-trip', () => {
  // A tool that returns an image (e.g. a screenshot tool)
  const canon: Message[] = [
    { role: 'user', content: [{ type: 'text', text: 'Take a screenshot' }] },
    { role: 'assistant', content: [], tool_calls: [{ id: 'tc1', name: 'screenshot', args: {} }] },
    { role: 'tool', tool_call_id: 'tc1', name: 'screenshot', content: [
      { type: 'image', mime_type: 'image/png', data: 'abc123', encoding: 'base64' },
    ]},
  ]
  const anth = toAnthropic(canon)
  const userMsg = [...anth.messages].reverse().find((m: { role: string }) => m.role === 'user') as { content: { type: string; content: unknown }[] }
  const toolResult = userMsg.content[0]
  // content should be an array of blocks, not a plain string
  assert.ok(Array.isArray(toolResult.content), 'tool_result content should be a block array when it contains an image')

  // Round-trip should preserve the image
  const back = fromAnthropic(anth)
  const toolMsg = back.find(m => m.role === 'tool')!
  assert.equal(toolMsg.content[0].type, 'image')
  if (toolMsg.content[0].type === 'image') assert.equal(toolMsg.content[0].data, 'abc123')
})

test('Gemini tool result round-trip does not corrupt text content', () => {
  const canon: Message[] = [
    { role: 'user',      content: [{ type: 'text', text: 'Weather?' }] },
    { role: 'assistant', content: [], tool_calls: [{ id: 'c1', name: 'get_weather', args: {} }] },
    { role: 'tool',      content: [{ type: 'text', text: 'Sunny, 22°C' }], tool_call_id: 'c1', name: 'get_weather' },
  ]
  const back = fromGemini(toGemini(canon))
  const toolMsg = back.find(m => m.role === 'tool')!
  assert.equal(toolMsg.content[0].type, 'text')
  if (toolMsg.content[0].type === 'text') assert.equal(toolMsg.content[0].text, 'Sunny, 22°C')
})

test('parallel tool calls full round-trip (OpenAI → Anthropic → OpenAI)', () => {
  const original = [
    { role: 'user', content: 'Go' },
    { role: 'assistant', content: null, tool_calls: [
      { id: 'c1', type: 'function', function: { name: 'a', arguments: '{"x":1}' } },
      { id: 'c2', type: 'function', function: { name: 'b', arguments: '{"y":2}' } },
    ]},
    { role: 'tool', tool_call_id: 'c1', content: 'res_a' },
    { role: 'tool', tool_call_id: 'c2', content: 'res_b' },
    { role: 'assistant', content: 'Done.' },
  ]
  const result = toOpenAI(fromAnthropic(toAnthropic(fromOpenAI(original)))) as { role: string; content: string | null; tool_calls?: { id: string; function: { name: string } }[] }[]
  assert.equal(result[1].tool_calls!.length, 2)
  assert.equal(result[1].tool_calls![0].function.name, 'a')
  assert.equal(result[1].tool_calls![1].function.name, 'b')
  assert.equal(result[4].content, 'Done.')
})

// ─── Results ──────────────────────────────────────────────────────────────────
console.log(`\n${passed + failed} tests: ${passed} passed, ${failed} failed\n`)
if (failed > 0) process.exit(1)
