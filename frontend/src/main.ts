import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHashHistory } from 'vue-router'
import App from './App.vue'
import ChatView from './views/ChatView.vue'
import HistoryView from './views/HistoryView.vue'
import SettingsView from './views/SettingsView.vue'
import './styles.css'
const router = createRouter({ history: createWebHashHistory(), routes: [ { path: '/', component: ChatView }, { path: '/history', component: HistoryView }, { path: '/settings', component: SettingsView } ] })
createApp(App).use(createPinia()).use(router).mount('#app')
