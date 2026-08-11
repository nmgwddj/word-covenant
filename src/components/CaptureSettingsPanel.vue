<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import type { CaptureMeter, SpeechDetectionSettings } from '@/types'

const minimumRmsThresholdDbfs = -60
const maximumRmsThresholdDbfs = 0
const defaultRmsThresholdDbfs = -10

const props = withDefaults(
  defineProps<{
    settings: SpeechDetectionSettings
    meter: CaptureMeter | null
    locked?: boolean
    loading?: boolean
    saving?: boolean
    error?: string | null
  }>(),
  {
    locked: false,
    loading: false,
    saving: false,
    error: null,
  }
)

const emit = defineEmits<{
  close: []
  save: [rmsThresholdDbfs: number]
  clearError: []
}>()

const panel = ref<HTMLElement | null>(null)
const rmsThresholdDbfs = ref(props.settings.rmsThresholdDbfs)
const controlsDisabled = computed(() => props.locked || props.loading || props.saving)
const lockMessage = computed(() => {
  if (!props.locked) return null
  return '记录准备或进行中，本次会话的检测门限已固定'
})

watch(
  () => props.settings.rmsThresholdDbfs,
  value => {
    rmsThresholdDbfs.value = value
  }
)

onMounted(() => {
  panel.value?.focus()
})

function clampThreshold(value: number): number {
  return Math.min(maximumRmsThresholdDbfs, Math.max(minimumRmsThresholdDbfs, Math.round(value)))
}

function updateDraft(event: Event) {
  const value = (event.target as HTMLInputElement).valueAsNumber
  if (Number.isFinite(value)) {
    rmsThresholdDbfs.value = clampThreshold(value)
  }
}

function saveDraft() {
  if (controlsDisabled.value) return
  emit('clearError')
  emit('save', rmsThresholdDbfs.value)
}

function resetThreshold() {
  if (controlsDisabled.value) return
  rmsThresholdDbfs.value = defaultRmsThresholdDbfs
  saveDraft()
}

function formatDbfs(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '-- dBFS'
  return `${Math.round(Math.min(0, Math.max(-96, value)))} dBFS`
}
</script>

<template>
  <aside
    ref="panel"
    class="capture-settings"
    aria-labelledby="capture-settings-title"
    tabindex="-1"
    @keydown.esc="emit('close')"
  >
    <header class="capture-settings__header">
      <div>
        <p class="capture-settings__eyebrow">CAPTURE TUNING</p>
        <h2 id="capture-settings-title">录音检测</h2>
      </div>
      <button
        class="icon-button capture-settings__close"
        type="button"
        aria-label="关闭录音检测设置"
        title="关闭"
        @click="emit('close')"
      >
        <span class="i-mdi-close" aria-hidden="true" />
      </button>
    </header>

    <section class="capture-settings__threshold" aria-labelledby="speech-threshold-label">
      <div class="capture-settings__section-heading">
        <div>
          <p id="speech-threshold-label">语音门限</p>
          <output data-testid="speech-threshold-value">{{ rmsThresholdDbfs }} dBFS</output>
        </div>
        <button
          class="icon-button capture-settings__reset"
          type="button"
          :disabled="controlsDisabled"
          aria-label="恢复默认 -10 dBFS"
          title="恢复默认 -10 dBFS"
          data-testid="reset-speech-threshold"
          @click="resetThreshold"
        >
          <span class="i-mdi-restore" aria-hidden="true" />
        </button>
      </div>

      <div class="capture-settings__threshold-controls">
        <input
          class="capture-settings__range"
          type="range"
          min="-60"
          max="0"
          step="1"
          :value="rmsThresholdDbfs"
          :disabled="controlsDisabled"
          aria-label="语音 RMS 门限"
          data-testid="speech-threshold-range"
          @input="updateDraft"
          @change="saveDraft"
        />
        <label class="capture-settings__number-field">
          <span class="sr-only">语音 RMS 门限 dBFS</span>
          <input
            type="number"
            min="-60"
            max="0"
            step="1"
            :value="rmsThresholdDbfs"
            :disabled="controlsDisabled"
            aria-label="语音 RMS 门限 dBFS"
            data-testid="speech-threshold-number"
            @input="updateDraft"
            @change="saveDraft"
          />
          <span aria-hidden="true">dBFS</span>
        </label>
      </div>
      <p v-if="lockMessage" class="capture-settings__lock-message" data-testid="speech-threshold-lock-message">
        {{ lockMessage }}
      </p>
      <p v-else class="capture-settings__threshold-note">默认 -10 dBFS</p>
    </section>

    <section class="capture-settings__meter" aria-label="当前输入电平">
      <div class="capture-settings__meter-row">
        <span>当前 RMS</span>
        <output data-testid="capture-settings-rms">{{ formatDbfs(meter?.rmsDbfs) }}</output>
      </div>
      <div class="capture-settings__meter-row">
        <span>当前峰值</span>
        <output data-testid="capture-settings-peak">{{ formatDbfs(meter?.peakDbfs) }}</output>
      </div>
    </section>

    <p v-if="loading" class="capture-settings__state" role="status">正在读取本地设置</p>
    <p v-else-if="saving" class="capture-settings__state" role="status">正在保存本地设置</p>
    <p v-if="error" class="capture-settings__error" role="alert">{{ error }}</p>
  </aside>
</template>
