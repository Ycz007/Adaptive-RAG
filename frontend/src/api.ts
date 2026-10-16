import type { ChatResponse, StreamEvent } from './types'
export function apiUrl(base: string, path: string) { return `${base.replace(/\/+$/, '')}${path}` }
export function validateBase(base: string) {
  if (base === '/api/v1') return ''
  try { const u = new URL(base); if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password || u.search || u.hash) return '请输入 HTTP(S) API 地址，不含凭据、查询或锚点'; return '' } catch { return '请输入完整 API 地址，或使用 /api/v1' }
}
async function check(response: Response) {
  if (response.ok) return response
  const body = await response.json().catch(() => ({}))
  throw new Error(body.message || body.detail && typeof body.detail === 'string' && body.detail || `请求失败（HTTP ${response.status}）`)
}
// SSE may arrive split inside JSON, CRLF, or UTF-8 codepoints. Buffer complete events.
export async function consumeSSE(body: ReadableStream<Uint8Array>, onEvent: (event: StreamEvent) => void) {
  const reader = body.getReader(); const decoder = new TextDecoder(); let buffer = ''
  const flush = (final = false) => {
    let match: RegExpExecArray | null
    while ((match = /\r?\n\r?\n/.exec(buffer))) {
      const block = buffer.slice(0, match.index); buffer = buffer.slice(match.index + match[0].length)
      const data = block.split(/\r?\n/).filter(l => l.startsWith('data:')).map(l => l.slice(5).replace(/^ /, '')).join('\n')
      if (data) onEvent(JSON.parse(data))
    }
    if (final && buffer.trim() && !buffer.trim().startsWith(':')) throw new Error('响应流不完整，请检查会话状态')
  }
  try { while (true) { const { value, done } = await reader.read(); if (done) break; buffer += decoder.decode(value, { stream: true }); flush() } buffer += decoder.decode(); flush(true) } finally { reader.releaseLock() }
}
export async function sendChat(base: string, payload: Record<string, unknown>, stream: boolean, signal: AbortSignal, onEvent: (e: StreamEvent) => void): Promise<ChatResponse> {
  const response = await check(await fetch(apiUrl(base, stream ? '/chat/stream' : '/chat'), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...payload, stream }), signal }))
  if (!stream) return response.json()
  if (!response.body) throw new Error('浏览器未提供响应流')
  let answer: ChatResponse | undefined
  await consumeSSE(response.body, event => { if (event.event === 'answer') answer = event.data; onEvent(event) })
  if (!answer) throw new Error('连接已结束，但未收到最终答案。操作状态可能需要核实。')
  return answer
}
export async function health(base: string) {
  const response = await check(await fetch(apiUrl(base, '/health'), { signal: AbortSignal.timeout(6000) })); const data = await response.json()
  if (data.status !== 'ok') throw new Error('健康检查返回异常')
  return data
}
// The backend streams a JSON answer envelope, not plain answer text.
export function partialAnswer(raw: string) {
  const start = /"answer"\s*:\s*"/.exec(raw)
  if (!start) return ''
  let out = ''; const text = raw.slice(start.index + start[0].length)
  for (let i = 0; i < text.length; i++) { const c = text[i]; if (c === '"') break; if (c !== '\\') { out += c; continue } const next = text[++i]; if (!next) break; if (next === 'u') { const hex = text.slice(i + 1, i + 5); if (!/^[0-9a-f]{4}$/i.test(hex)) break; out += String.fromCharCode(parseInt(hex, 16)); i += 4 } else out += ({ n: '\n', r: '\r', t: '\t', '"': '"', '\\': '\\', '/': '/' } as Record<string, string>)[next] ?? next }
  return out
}
