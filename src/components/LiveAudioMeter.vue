<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CaptureMeter } from '@/types'

const props = withDefaults(
  defineProps<{
    meter: CaptureMeter | null
    active: boolean
  }>(),
  {
    active: false,
  }
)

type MeterBar = {
  height: number
  opacity: number
}

const DBFS_FLOOR = -72
const BAR_COUNT = 24
const QUIET_LEVEL = 2 / 90

function clampDbfs(value: number): number {
  if (!Number.isFinite(value)) return DBFS_FLOOR
  return Math.min(0, Math.max(DBFS_FLOOR, value))
}

function normalizeDbfs(value: number): number {
  return (clampDbfs(value) - DBFS_FLOOR) / -DBFS_FLOOR
}

function rounded(value: number): number {
  return Math.round(value * 100) / 100
}

const normalizedMeter = computed(() => {
  if (!props.meter) return null

  const rmsDbfs = clampDbfs(props.meter.rmsDbfs)
  return {
    rmsDbfs,
    peakDbfs: Math.max(rmsDbfs, clampDbfs(props.meter.peakDbfs)),
  }
})

const meterHistory = ref<number[]>([])

watch(
  () => ({ active: props.active, rmsDbfs: props.meter?.rmsDbfs }),
  ({ active, rmsDbfs }) => {
    if (!active || rmsDbfs === undefined) {
      meterHistory.value = []
      return
    }

    meterHistory.value = [...meterHistory.value.slice(-(BAR_COUNT - 1)), normalizeDbfs(rmsDbfs)]
  },
  { immediate: true }
)

const state = computed(() => {
  if (!props.active) return 'idle'
  if (!props.meter) return 'waiting'
  if (props.meter.clipping) return 'clipping'
  return 'active'
})

const visualBars = computed<MeterBar[]>(() => {
  const history = props.active ? meterHistory.value.slice(-BAR_COUNT) : []
  const levels = [...Array.from({ length: BAR_COUNT - history.length }, () => QUIET_LEVEL), ...history]

  return levels.map(level => ({
    height: rounded(10 + level * 90),
    opacity: rounded(0.35 + level * 0.65),
  }))
})

const peakReadout = computed(() => {
  if (!props.active || !normalizedMeter.value) return '-- dBFS'
  return `${Math.round(normalizedMeter.value.peakDbfs)} dBFS`
})

const accessibleLabel = computed(() => {
  if (!props.active) return '输入声线未启用'
  if (!normalizedMeter.value) return '输入声线正在等待输入电平'

  const label = `实时输入电平声线，平均 ${Math.round(normalizedMeter.value.rmsDbfs)} dBFS，峰值 ${Math.round(normalizedMeter.value.peakDbfs)} dBFS`
  return props.meter?.clipping ? `${label}，检测到削波` : label
})
</script>

<template>
  <div
    class="live-audio-meter"
    :class="`live-audio-meter--${state}`"
    :data-state="state"
    :aria-label="accessibleLabel"
    :title="accessibleLabel"
    role="img"
  >
    <span class="live-audio-meter__bars" aria-hidden="true">
      <span
        v-for="(bar, index) in visualBars"
        :key="index"
        class="live-audio-meter__bar"
        :style="{ height: `${bar.height}%`, opacity: bar.opacity }"
      />
    </span>
    <output class="live-audio-meter__readout" aria-hidden="true">{{ peakReadout }}</output>
  </div>
</template>

<style scoped>
.live-audio-meter {
  display: inline-grid;
  grid-template-columns: minmax(60px, 1fr) auto;
  align-items: center;
  gap: 7px;
  min-inline-size: 108px;
  max-inline-size: 154px;
  color: var(--ink-muted);
}

.live-audio-meter__bars {
  display: flex;
  align-items: center;
  gap: 1px;
  block-size: 30px;
  min-inline-size: 0;
}

.live-audio-meter__bar {
  display: block;
  flex: 1 1 0;
  max-inline-size: 2px;
  min-inline-size: 1px;
  background: currentColor;
  transition:
    height 80ms linear,
    opacity 80ms linear;
}

.live-audio-meter__readout {
  min-inline-size: 38px;
  margin: 0;
  padding: 0;
  color: currentColor;
  background: transparent;
  font-family: var(--font-mono);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  text-align: end;
  white-space: nowrap;
}

.live-audio-meter--active {
  color: var(--ink-soft);
}

.live-audio-meter--clipping {
  color: var(--recording);
}

@media (prefers-reduced-motion: reduce) {
  .live-audio-meter__bar {
    transition: none;
  }
}
</style>
