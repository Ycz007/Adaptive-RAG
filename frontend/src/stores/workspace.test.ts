import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useWorkspace } from './workspace'
import type { ChatResponse } from '../types'
const { sendChat } = vi.hoisted(() => ({ sendChat: vi.fn() }))
vi.mock('../api', async importOriginal => ({ ...await importOriginal<typeof import('../api')>(), sendChat }))
const response: ChatResponse = { answer: '需要确认', session_id: 'session-1', trace_id: 'trace-1', intent: 'cancel_order', strategy: 'agent', complexity: 'tool_required', citations: [], pending_action: { action_id: 'action-1', tool_name: 'cancel_order', arguments: { order_id: 'ORD10003' }, risk: 'write', summary: '请确认' }, refused: false, token_usage: { total_tokens: 10 }, latency: { total_ms: 100 } }
beforeEach(() => { const memory = new Map<string, string>(); vi.stubGlobal('localStorage', { getItem: (key: string) => memory.get(key) ?? null, setItem: (key: string, value: string) => memory.set(key, value) }); setActivePinia(createPinia()); sendChat.mockReset() })
describe('conversation and action state', () => {
  it('preserves conversation identity and original request on confirmation; prevents duplicate submit', async () => {
    const store = useWorkspace(); sendChat.mockResolvedValue(response)
    await store.send('取消 ORD10003'); const message = store.active!.messages[1]!
    store.settings.userId = 'different-user'; store.settings.apiBase = 'http://other/api/v1'
    await store.send(message.originalQuery!, message)
    expect(sendChat.mock.calls[1]![0]).toBe('/api/v1')
    expect(sendChat.mock.calls[1]![1]).toMatchObject({ message: '取消 ORD10003', user_id: 'u_demo', session_id: 'session-1', confirm_action_id: 'action-1', idempotency_key: 'action-1' })
    await store.send(message.originalQuery!, message); expect(sendChat).toHaveBeenCalledTimes(2)
  })
  it('marks interrupted confirmation uncertain and does not allow resubmission', async () => {
    const store = useWorkspace(); sendChat.mockResolvedValueOnce(response); await store.send('取消 ORD10003')
    const message = store.active!.messages[1]!; sendChat.mockRejectedValueOnce(new Error('connection lost'))
    await store.send(message.originalQuery!, message); expect(message.actionStatus).toBe('uncertain')
    expect(store.active!.messages.at(-1)?.error).toContain('先核实业务操作结果')
    await store.send(message.originalQuery!, message); expect(sendChat).toHaveBeenCalledTimes(2)
  })
})
