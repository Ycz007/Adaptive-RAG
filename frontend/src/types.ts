export type Strategy = 'direct' | 'single' | 'multi' | 'agent' | 'refuse'
export interface Action { action_id: string; tool_name: string; arguments: Record<string, unknown>; summary: string; risk: string }
export interface ChatResponse { answer: string; session_id: string; trace_id: string; intent: string | null; strategy: Strategy | null; complexity: string | null; citations: { document_id: string; title: string; chunk_id: string; section?: string | null }[]; pending_action: Action | null; refused: boolean; error_code?: string | null; token_usage: { total_tokens: number }; latency: Record<string, number> }
export interface Message { id: string; role: 'user' | 'assistant'; content: string; time: string; response?: ChatResponse; nodes?: string[]; error?: string; actionStatus?: 'cancelled' | 'submitted' | 'uncertain'; originalQuery?: string }
export interface Conversation { id: string; sessionId?: string; title: string; userId: string; apiBase: string; updated: string; messages: Message[] }
export interface Settings { apiBase: string; userId: string; stream: boolean }
export interface StreamEvent { event: string; text?: string; node?: string; strategy?: Strategy; data?: ChatResponse }
export const strategyNames: Record<Strategy, string> = { direct: '直接回答', single: '单轮检索', multi: '多轮补证', agent: '工具调用', refuse: '安全拒答' }
export const nodeNames: Record<string, string> = { apply_guardrail: '安全护栏', classify_intent: '意图识别', route_complexity: '策略路由', retrieve_docs: '知识检索', plan_tasks: '任务规划', execute_tools: '工具执行', generate_answer: '答案生成', refuse_answer: '安全拒答', check_faithfulness: '忠实度检查', compact_memory: '记忆压缩' }
