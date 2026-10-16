import { computed, ref, watch } from 'vue'
import { defineStore } from 'pinia'
import { health, partialAnswer, sendChat } from '../api'
import type { Conversation, Message, Settings } from '../types'
const KEY = 'adaptive-rag-workspace-v1'
export const useWorkspace = defineStore('workspace', () => {
  let saved: { conversations?: Conversation[]; settings?: Settings; activeId?: string } = {}
  let corrupt = false
  try { const raw = JSON.parse(localStorage.getItem(KEY) || '{}'); if (!raw || typeof raw !== 'object' || raw.conversations && !Array.isArray(raw.conversations)) throw new Error(); saved = raw } catch { corrupt = true }
  const settings = ref<Settings>({ apiBase: '/api/v1', userId: 'u_demo', stream: true, ...saved.settings })
  const conversations = ref<Conversation[]>((saved.conversations || []).filter(c => c && typeof c.id === 'string' && Array.isArray(c.messages) && typeof c.title === 'string').map(c => ({ ...c, apiBase: c.apiBase || '/api/v1' })))
  const activeId = ref(saved.activeId || '')
  const active = computed(() => conversations.value.find(c => c.id === activeId.value))
  const busy = ref(false); const connection = ref<'unknown' | 'checking' | 'online' | 'offline'>('unknown'); const connectionError = ref('')
  const toast = ref(corrupt ? '本地记录无法读取，已使用空工作区。' : ''); let toastTimer: ReturnType<typeof setTimeout>
  function notify(text: string) { toast.value = text; clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.value = '', 5000) }
  let storageWarned = false
  watch([settings, conversations, activeId], () => { try { localStorage.setItem(KEY, JSON.stringify({ settings: settings.value, conversations: conversations.value, activeId: activeId.value })) } catch { if (!storageWarned) { storageWarned = true; notify('本地存储空间不足，当前变更无法持久保存。') } } }, { deep: true })
  function newChat() { if (busy.value) return; activeId.value = '' }
  function open(id: string) { if (!busy.value) activeId.value = id }
  function remove(id: string) { if (busy.value) return; conversations.value = conversations.value.filter(c => c.id !== id); if (activeId.value === id) activeId.value = ''; notify('已删除本地会话记录') }
  async function checkConnection(base = settings.value.apiBase) { connection.value = 'checking'; connectionError.value = ''; try { await health(base); connection.value = 'online'; return true } catch (e) { connection.value = 'offline'; connectionError.value = e instanceof Error ? e.message : '无法连接'; return false } }
  let controller: AbortController | undefined
  function stop() { controller?.abort() }
  async function send(text: string, proposalMessage?: Message) {
    if (busy.value || !text.trim() || text.length > 4000) return
    let conversation = active.value
    if (!conversation) { conversation = { id: crypto.randomUUID(), title: text.slice(0, 30), userId: settings.value.userId, apiBase: settings.value.apiBase, updated: new Date().toISOString(), messages: [] }; conversations.value.unshift(conversation); activeId.value = conversation.id; conversation = active.value! }
    const current = conversation
    const proposal = proposalMessage?.response?.pending_action
    if (proposalMessage?.actionStatus) return
    if (proposalMessage) proposalMessage.actionStatus = 'submitted'
    current.messages.push({ id: crypto.randomUUID(), role: 'user', content: proposal ? `确认执行：${proposal.tool_name}` : text, time: new Date().toISOString() })
    const message: Message = { id: crypto.randomUUID(), role: 'assistant', content: '', time: new Date().toISOString(), nodes: [], originalQuery: text }
    current.messages.push(message)
    const target = current.messages[current.messages.length - 1]!
    busy.value = true; controller = new AbortController(); let raw = ''; let timedOut = false
    const timer = setTimeout(() => { timedOut = true; controller?.abort() }, 120000)
    try {
      const result = await sendChat(current.apiBase, { message: text, user_id: current.userId, session_id: current.sessionId, confirm_action_id: proposal?.action_id, idempotency_key: proposal?.action_id || crypto.randomUUID() }, settings.value.stream, controller.signal, event => {
        if (event.node) target.nodes!.push(event.node)
        if (event.text) { raw += event.text; target.content = partialAnswer(raw) }
      })
      target.content = result.answer; target.response = result; current.sessionId = result.session_id; connection.value = 'online'
    } catch (e) {
      target.error = timedOut ? '请求超时，请检查服务状态后再试。' : controller.signal.aborted ? '已停止接收。服务端可能仍在处理本次请求。' : e instanceof Error ? e.message : '请求失败'
      if (proposalMessage) { proposalMessage.actionStatus = 'uncertain'; target.error += ' 请先核实业务操作结果，勿重复提交。' }
      notify(target.error)
    } finally { clearTimeout(timer); busy.value = false; controller = undefined; current.updated = new Date().toISOString() }
  }
  return { settings, conversations, activeId, active, busy, connection, connectionError, toast, notify, newChat, open, remove, checkConnection, stop, send }
})
