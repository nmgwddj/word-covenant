<script setup lang="ts">
defineProps<{
  recording: boolean
  disabled?: boolean
}>()

defineEmits<{
  toggle: []
}>()
</script>

<template>
  <button
    class="recording-control"
    :class="{ 'recording-control--active': recording }"
    type="button"
    :disabled="disabled"
    @click="$emit('toggle')"
  >
    <span class="recording-control__mark" :class="recording ? 'i-mdi-stop' : 'i-mdi-record-circle'" aria-hidden="true" />
    <span>{{ recording ? '结束记录' : '开始记录' }}</span>
  </button>
</template>
