<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { X } from 'lucide-vue-next'
defineProps<{ title: string; drawer?: boolean }>()
const emit = defineEmits<{ close: [] }>(); const panel = ref<HTMLElement>(); const previous = document.activeElement as HTMLElement | null
function keydown(e: KeyboardEvent) { if (e.key === 'Escape') emit('close'); if (e.key !== 'Tab') return; const items = panel.value?.querySelectorAll<HTMLElement>('button:not(:disabled),input,select,textarea,a[href],[tabindex="0"]'); if (!items?.length) { e.preventDefault(); return } const first = items[0]!; const last = items[items.length - 1]!; if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() } }
onMounted(() => { panel.value?.focus(); document.addEventListener('keydown', keydown); document.body.style.overflow = 'hidden' })
onBeforeUnmount(() => { document.removeEventListener('keydown', keydown); document.body.style.overflow = ''; previous?.focus() })
</script>
<template><Teleport to="body"><div class="overlay" @click.self="emit('close')"><section ref="panel" class="modal" :class="{ drawer }" role="dialog" aria-modal="true" aria-labelledby="dialog-title" tabindex="-1"><header><h2 id="dialog-title">{{ title }}</h2><button class="icon-button" aria-label="关闭" @click="emit('close')"><X :size="18" /></button></header><div class="modal-body"><slot /></div><footer v-if="$slots.footer"><slot name="footer" /></footer></section></div></Teleport></template>
