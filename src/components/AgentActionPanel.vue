<script setup lang="ts">
import OutboundAccessControl from '@/components/OutboundAccessControl.vue'
import type { AgentAction } from '@/types'

defineProps<{
  actions: AgentAction[]
  egressEnabled: boolean
  activeEgressApprovals: number
  egressLoading?: boolean
}>()

defineEmits<{
  propose: []
  setEgressEnabled: [enabled: boolean]
}>()
</script>

<template>
  <aside class="agent-panel" aria-label="行动草案">
    <div class="section-heading">
      <div>
        <p class="section-heading__eyebrow">AGENT</p>
        <h2>行动草案</h2>
      </div>
      <button class="icon-button" type="button" title="生成本地行动草案" @click="$emit('propose')">
        <span class="i-mdi-sparkles" aria-hidden="true" />
      </button>
    </div>

    <OutboundAccessControl
      :egress-enabled="egressEnabled"
      :profile-approvals="activeEgressApprovals"
      :disabled="egressLoading"
      @set-egress-enabled="$emit('setEgressEnabled', $event)"
    />

    <ul v-if="actions.length" class="action-list">
      <li v-for="action in actions" :key="action.id" class="action-row">
        <span
          class="action-row__icon"
          :class="action.kind === 'local_speech' ? 'i-mdi-volume-high' : 'i-mdi-web'"
          aria-hidden="true"
        />
        <div>
          <strong>{{ action.title }}</strong>
          <p>{{ action.detail }}</p>
        </div>
        <span class="action-row__state" :class="`action-row__state--${action.status}`">
          {{ action.status === 'ready' ? '待确认' : action.status === 'blocked' ? '已拦截' : '已完成' }}
        </span>
      </li>
    </ul>
    <div v-else class="agent-panel__empty">
      <span class="i-mdi-file-document-edit-outline" aria-hidden="true" />
      <p>等待主动触发</p>
    </div>
  </aside>
</template>
