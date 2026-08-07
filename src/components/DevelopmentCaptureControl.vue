<script setup lang="ts">
defineProps<{
  selected: boolean
  disabled?: boolean
}>()

defineEmits<{
  select: []
}>()
</script>

<template>
  <button
    class="icon-button development-capture-control"
    :class="{ 'development-capture-control--selected': selected }"
    type="button"
    title="开发模拟音频输入"
    aria-label="开发模拟音频输入"
    :aria-pressed="selected"
    :disabled="disabled"
    data-testid="development-capture-selector"
    @click="$emit('select')"
  >
    <span class="i-mdi-waveform" aria-hidden="true" />
  </button>
</template>
