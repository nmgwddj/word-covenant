<script setup lang="ts">
import { computed } from 'vue'
import type { CaptureProjection } from '@/types'

const props = defineProps<{
  capture: CaptureProjection
  disabled?: boolean
}>()

defineEmits<{
  select: [deviceUid: string]
}>()

const statusLabel = computed(() => {
  if (props.capture.status === 'awaiting_permission') return '正在请求麦克风权限'
  if (props.capture.status === 'recording') return '麦克风记录中'
  if (props.capture.status === 'interrupted') return '输入已中断'
  if (props.capture.permission === 'denied') return '麦克风权限被拒绝'
  if (props.capture.permission === 'restricted') return '麦克风权限受系统限制'
  if (props.capture.lastIssue?.code === 'no_input_device') return '未检测到输入设备'
  if (props.capture.status === 'failed') return '麦克风不可用'
  return '麦克风待命'
})

const meterWidth = computed(() => {
  const peak = props.capture.meter?.peakDbfs ?? -96
  return `${Math.max(0, Math.min(100, ((peak + 96) / 96) * 100))}%`
})

const meterLabel = computed(() => {
  if (!props.capture.meter) return '当前没有输入电平'
  return `输入电平 ${Math.round(props.capture.meter.peakDbfs)} dBFS`
})

const deviceSelectionDisabled = computed(() => (
  Boolean(props.disabled)
  || props.capture.status === 'recording'
  || props.capture.status === 'awaiting_permission'
))
</script>

<template>
  <div class="capture-status" :class="`capture-status--${capture.status}`" aria-live="polite">
    <span class="capture-status__signal" aria-hidden="true" />
    <select
      class="capture-status__device"
      :value="capture.selectedDevice?.uid ?? ''"
      :disabled="deviceSelectionDisabled"
      aria-label="输入设备"
      @change="$emit('select', ($event.target as HTMLSelectElement).value)"
    >
      <option value="" disabled>{{ capture.devices.length ? '选择输入设备' : '未检测到输入设备' }}</option>
      <option v-for="device in capture.devices" :key="device.uid" :value="device.uid">
        {{ device.name }}
      </option>
    </select>
    <span class="capture-status__meter" :aria-label="meterLabel" role="meter">
      <span :style="{ width: meterWidth }" />
    </span>
    <span class="capture-status__label">{{ statusLabel }}</span>
  </div>
</template>
