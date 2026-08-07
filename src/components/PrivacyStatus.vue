<script setup lang="ts">
import { computed } from 'vue'
import type { PrivacyStatus as PrivacyStatusType } from '@/types'

const props = defineProps<{
  status: PrivacyStatusType
}>()

const label = computed(() => {
  if (!props.status.egressEnabled) {
    return '本地模式'
  }

  return props.status.activeEgressApprovals > 0 ? '受控出网' : '出网待配置'
})
</script>

<template>
  <div class="privacy-status" :class="{ 'privacy-status--egress': status.egressEnabled }">
    <span class="privacy-status__icon i-mdi-shield-check-outline" aria-hidden="true" />
    <span>{{ label }}</span>
    <span v-if="status.activeEgressApprovals > 0" class="privacy-status__count">
      {{ status.activeEgressApprovals }}
    </span>
  </div>
</template>
