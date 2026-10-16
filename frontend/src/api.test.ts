import { describe, expect, it } from 'vitest'
import { consumeSSE, partialAnswer, validateBase } from './api'
import type { StreamEvent } from './types'
describe('SSE transport', () => {
  it('handles byte fragmentation, Chinese UTF-8, CRLF and final answer events', async () => {
    const bytes = new TextEncoder().encode(': heartbeat\r\n\r\ndata: {"event":"token","text":"你好"}\r\n\r\ndata: {"event":"done"}\n\n')
    const stream = new ReadableStream<Uint8Array>({ start(controller) { for (const byte of bytes) controller.enqueue(new Uint8Array([byte])); controller.close() } })
    const events: StreamEvent[] = []; await consumeSSE(stream, event => events.push(event))
    expect(events).toEqual([{ event: 'token', text: '你好' }, { event: 'done' }])
  })
  it('rejects a truncated event instead of treating it as completed', async () => {
    const stream = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode('data: {"event":')); c.close() } })
    await expect(consumeSSE(stream, () => {})).rejects.toThrow('响应流不完整')
  })
})
describe('streaming answer envelope', () => {
  it('shows only answer content, supports escapes and hides incomplete unicode', () => {
    expect(partialAnswer('{"answer": "你好\\n\\u4e16\\u754c", "refused":false}')).toBe('你好\n世界')
    expect(partialAnswer('{"answer": "你好\\u4e')).toBe('你好')
    expect(partialAnswer('{"answer": "引号\\"内容')).toBe('引号"内容')
    expect(partialAnswer('{"citation_chunk_ids":[]')).toBe('')
  })
})
describe('API settings', () => {
  it('accepts the proxy and http endpoints, rejects unsafe or malformed URLs', () => {
    expect(validateBase('/api/v1')).toBe(''); expect(validateBase('http://localhost:8000/api/v1')).toBe('')
    for (const base of ['javascript:alert(1)', '/invalid', 'https://user:pass@example.com/api', 'http://localhost/api?secret=1']) expect(validateBase(base)).not.toBe('')
  })
})
