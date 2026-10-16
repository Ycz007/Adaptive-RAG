<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Layers, MessageSquare, History, Settings2, Plus, PanelLeftClose, PanelLeftOpen, Menu, ChevronDown, ArrowUpRight, X, CircleHelp } from 'lucide-vue-next'
import { useWorkspace } from './stores/workspace'
const store = useWorkspace(); const route = useRoute(); const router = useRouter(); const collapsed = ref(false); const mobile = ref(false)
const title = computed(() => route.path === '/history' ? '会话记录' : route.path === '/settings' ? '设置' : '客服工作台')
const navigation = [{ path: '/', label: '客服工作台', icon: MessageSquare }, { path: '/history', label: '会话记录', icon: History }, { path: '/settings', label: '设置', icon: Settings2 }]
function create() { store.newChat(); router.push('/'); mobile.value = false }
function select(id: string) { store.open(id); router.push('/'); mobile.value = false }
onMounted(() => store.checkConnection())
</script>
<template>
  <div class="app-shell" :class="{ collapsed }">
    <div v-if="mobile" class="nav-backdrop" @click="mobile = false" />
    <aside class="sidebar" :class="{ 'mobile-open': mobile }">
      <div class="brand"><span class="brand-mark"><Layers :size="20" /></span><span class="brand-name">Adaptive <strong>RAG</strong></span><button class="icon-button mobile-only" aria-label="关闭导航" @click="mobile = false"><X :size="18" /></button></div>
      <button class="workspace-switch" @click="router.push('/settings')"><span class="workspace-avatar">A</span><span class="nav-label">客服工作空间<small>本地工作区</small></span><ChevronDown class="nav-label" :size="14" /></button>
      <div class="nav-section-label">工作空间</div>
      <nav aria-label="主导航"><RouterLink v-for="item in navigation" :key="item.path" :to="item.path" :title="item.label" @click="mobile = false"><component :is="item.icon" :size="18" /><span class="nav-label">{{ item.label }}</span><span v-if="item.path === '/history'" class="nav-count nav-label">{{ store.conversations.length }}</span></RouterLink></nav>
      <div class="recent nav-label"><div class="recent-heading">最近会话<button class="icon-button" aria-label="新建会话" :disabled="store.busy" @click="create"><Plus :size="15" /></button></div><button v-for="c in store.conversations.slice(0, 6)" :key="c.id" class="recent-item" :class="{ selected: store.activeId === c.id && route.path === '/' }" :disabled="store.busy" @click="select(c.id)"><MessageSquare :size="14" /><span>{{ c.title }}</span></button><p v-if="!store.conversations.length" class="recent-empty">你的会话将在这里显示</p></div>
      <div class="sidebar-bottom"><div class="local-note nav-label"><span class="status-dot" />浏览器本地保存<small>会话仅保存在当前设备</small></div><button class="collapse-button" :aria-label="collapsed ? '展开导航' : '收起导航'" @click="collapsed = !collapsed"><PanelLeftOpen v-if="collapsed" :size="18" /><PanelLeftClose v-else :size="18" /><span class="nav-label">收起侧边栏</span></button></div>
    </aside>
    <div class="main-shell"><header class="topbar"><div class="breadcrumb"><button class="icon-button mobile-only" aria-label="打开导航" @click="mobile = true"><Menu :size="20" /></button><span class="muted">工作空间</span><span class="slash">/</span><span>{{ title }}</span></div><div class="topbar-actions"><button class="connection-button" @click="store.checkConnection()" :disabled="store.connection === 'checking'"><span class="status-dot" :class="store.connection" />{{ ({ online: '服务已连接', offline: '服务未连接', checking: '检查连接中', unknown: '未检查' })[store.connection] }}</button><span class="top-separator" /><a class="icon-button" href="https://github.com" title="关于工作空间" @click.prevent="store.notify('Adaptive RAG · 电商客服工作台 v0.1.0')"><CircleHelp :size="18" /></a><span class="user-avatar" :title="store.settings.userId">{{ store.settings.userId.slice(0, 1).toUpperCase() }}</span></div></header>
      <main><RouterView /></main>
    </div>
    <div v-if="store.toast" class="toast" role="status"><span>{{ store.toast }}</span><button class="icon-button" aria-label="关闭提示" @click="store.toast = ''"><X :size="16" /></button></div>
  </div>
</template>
