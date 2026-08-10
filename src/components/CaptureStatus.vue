<script setup lang="ts">
import { computed } from 'vue'
import type { CaptureProjection } from '@/types'

const props = defineProps<{
  capture: CaptureProjection
  disabled?: boolean
  asrModelReady?: boolean
}>()

defineEmits<{
  select: [deviceUid: string]
}>()

const bridge = computed(() => props.capture.bridge ?? null)

const statusLabel = computed(() => {
  if (bridge.value?.status === 'parked') return '正在启动本地记录'
  if (bridge.value?.status === 'closing') return '正在完成本地记录'
  if (props.capture.status === 'awaiting_permission') return '正在请求麦克风权限'
  if (props.capture.status === 'recording') return '麦克风记录中'
  if (props.capture.status === 'interrupted') return '输入已中断'
  if (props.capture.permission === 'denied') return '麦克风权限被拒绝'
  if (props.capture.permission === 'restricted') return '麦克风权限受系统限制'
  if (props.capture.lastIssue?.code === 'no_input_device') return '未检测到输入设备'
  if (props.capture.status === 'failed') return '麦克风不可用'
  if (props.asrModelReady === false) return '请选择本地转写模型'
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

const deviceSelectionDisabled = computed(
  () =>
    Boolean(props.disabled) ||
    Boolean(bridge.value) ||
    props.capture.status === 'recording' ||
    props.capture.status === 'awaiting_permission'
)

const bridgeStatusLabel = computed(() => {
  switch (bridge.value?.status) {
    case 'parked':
      return '启动中'
    case 'armed':
      return '运行'
    case 'closing':
      return '收尾中'
    case 'drained':
      return '已收尾'
    default:
      return '待命'
  }
})

function compactMetricCount(value: number): string {
  return value > 999 ? '999+' : String(Math.max(0, value))
}

const bridgeIssueCount = computed(() => {
  const metrics = bridge.value?.metrics
  if (!metrics) return 0

  return (
    metrics.ingressDiscontinuities +
    metrics.segmenterFailures +
    metrics.jobQueueSaturated +
    metrics.resultQueueSaturated +
    metrics.unavailableEngineOutcomes +
    metrics.engineFailureOutcomes +
    metrics.shutdownOutcomes +
    metrics.outcomeClaimsAborted
  )
})

const bridgeIssueDetails = computed(() => {
  const metrics = bridge.value?.metrics
  if (!metrics) return []

  const details: string[] = []
  if (metrics.ingressDiscontinuities > 0) details.push(`输入中断 ${metrics.ingressDiscontinuities}`)
  if (metrics.segmenterFailures > 0) details.push(`分段失败 ${metrics.segmenterFailures}`)
  const saturatedQueues = metrics.jobQueueSaturated + metrics.resultQueueSaturated
  if (saturatedQueues > 0) details.push(`队列饱和 ${saturatedQueues}`)
  if (metrics.unavailableEngineOutcomes > 0) details.push(`本地引擎不可用 ${metrics.unavailableEngineOutcomes}`)
  if (metrics.engineFailureOutcomes > 0) details.push(`本地引擎失败 ${metrics.engineFailureOutcomes}`)
  if (metrics.shutdownOutcomes > 0) details.push(`收尾缺口 ${metrics.shutdownOutcomes}`)
  if (metrics.outcomeClaimsAborted > 0) details.push(`结果持久化重试 ${metrics.outcomeClaimsAborted}`)
  return details
})

const bridgeAriaLabel = computed(() => {
  const activeBridge = bridge.value
  if (!activeBridge) return ''

  const persistence = activeBridge.metrics.ownedOutcomeLeaseActive ? '有结果等待持久化' : '没有结果等待持久化'
  const label = [
    `本地推理桥接${bridgeStatusLabel.value}`,
    `任务队列 ${activeBridge.metrics.jobQueueDepth}`,
    `结果队列 ${activeBridge.metrics.resultQueueDepth}`,
    `待持久化事件 ${activeBridge.metrics.pendingEventDepth}`,
    persistence,
  ]
  if (bridgeIssueDetails.value.length > 0) {
    label.push(...bridgeIssueDetails.value)
  }
  return label.join('，')
})
</script>

<template>
  <div
    class="capture-status"
    :class="[`capture-status--${capture.status}`, capture.bridge && `capture-status--bridge-${capture.bridge.status}`]"
    aria-live="polite"
  >
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
    <span class="capture-status__label" :title="statusLabel">{{ statusLabel }}</span>
    <span
      v-if="bridge"
      class="capture-status__bridge"
      :class="`capture-status__bridge--${bridge.status}`"
      :aria-label="bridgeAriaLabel"
      :title="bridgeAriaLabel"
      role="status"
    >
      <span class="capture-status__bridge-state">桥接 {{ bridgeStatusLabel }}</span>
      <span class="capture-status__bridge-metrics" aria-hidden="true">
        <span class="capture-status__bridge-metric">
          <span>任务</span><b>{{ compactMetricCount(bridge.metrics.jobQueueDepth) }}</b>
        </span>
        <span class="capture-status__bridge-metric">
          <span>结果</span><b>{{ compactMetricCount(bridge.metrics.resultQueueDepth) }}</b>
        </span>
        <span class="capture-status__bridge-metric">
          <span>待存</span><b>{{ compactMetricCount(bridge.metrics.pendingEventDepth) }}</b>
        </span>
        <span
          class="capture-status__bridge-metric capture-status__bridge-metric--issue"
          :class="{ 'capture-status__bridge-metric--active': bridgeIssueCount > 0 }"
        >
          <span>异常</span><b>{{ compactMetricCount(bridgeIssueCount) }}</b>
        </span>
      </span>
    </span>
  </div>
</template>
