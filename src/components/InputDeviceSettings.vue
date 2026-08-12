<script setup lang="ts">
import { computed } from 'vue'
import FlatSelect from '@/components/FlatSelect.vue'
import type { CaptureProjection } from '@/types'

const props = defineProps<{
  capture: CaptureProjection
  disabled?: boolean
  refreshing?: boolean
  refreshError?: string | null
}>()

const emit = defineEmits<{
  select: [deviceUid: string]
  refresh: []
}>()

const deviceSelectionDisabled = computed(
  () =>
    Boolean(props.disabled) ||
    Boolean(props.capture.bridge) ||
    props.capture.status === 'recording' ||
    props.capture.status === 'awaiting_permission'
)

const selectionNote = computed(() => {
  if (props.refreshError) return '无法刷新输入设备'
  if (!props.capture.devices.length) return '未检测到输入设备'
  if (deviceSelectionDisabled.value) return '录音期间不可切换输入设备'
  return null
})

const deviceOptions = computed(() =>
  props.capture.devices.map(device => ({
    value: device.uid,
    label: device.name,
  }))
)
</script>

<template>
  <div class="input-device-settings">
    <div class="input-device-settings__control">
      <label class="input-device-settings__field">
        <span>麦克风输入</span>
        <FlatSelect
          class="input-device-settings__select-control"
          :model-value="capture.selectedDevice?.uid ?? ''"
          :options="deviceOptions"
          :placeholder="capture.devices.length ? '选择输入设备' : '未检测到输入设备'"
          :disabled="deviceSelectionDisabled || !capture.devices.length"
          label="输入设备"
          test-id="input-device-select"
          @update:model-value="emit('select', $event)"
        />
      </label>
      <button
        class="icon-button input-device-settings__refresh"
        type="button"
        :disabled="deviceSelectionDisabled || refreshing"
        :aria-busy="refreshing"
        aria-label="刷新输入设备"
        title="刷新输入设备"
        data-testid="refresh-input-devices"
        @click="emit('refresh')"
      >
        <span class="i-mdi-refresh" :class="{ 'is-spinning': refreshing }" aria-hidden="true" />
      </button>
    </div>
    <p
      v-if="selectionNote"
      class="input-device-settings__note"
      :class="{ 'input-device-settings__note--error': refreshError }"
      role="status"
    >
      {{ selectionNote }}
    </p>
  </div>
</template>
