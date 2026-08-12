<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import type { CaptureMeter, SpeechDetectionSettings } from '@/types'

const minimumRmsThresholdDbfs = -42
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
    embedded?: boolean
  }>(),
  {
    locked: false,
    loading: false,
    saving: false,
    error: null,
    embedded: false,
  }
)

const emit = defineEmits<{
  close: []
  save: [settings: SpeechDetectionSettings]
  clearError: []
}>()

const panel = ref<HTMLElement | null>(null)
const rmsThresholdDbfs = ref(props.settings.rmsThresholdDbfs)
const mode = ref(props.settings.mode)
const controlsDisabled = computed(() => props.locked || props.loading || props.saving)
const manualControlsDisabled = computed(() => controlsDisabled.value || mode.value !== 'manual')
const lockMessage = computed(() => {
  if (!props.locked) return null
  return '记录准备或进行中，本次会话的检测方式已固定'
})

watch(
  () => props.settings,
  settings => {
    mode.value = settings.mode
    rmsThresholdDbfs.value = settings.rmsThresholdDbfs
  }
)

onMounted(() => {
  if (!props.embedded) panel.value?.focus()
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
  if (manualControlsDisabled.value) return
  emit('clearError')
  emit('save', { mode: 'manual', rmsThresholdDbfs: rmsThresholdDbfs.value })
}

function resetThreshold() {
  if (manualControlsDisabled.value) return
  rmsThresholdDbfs.value = defaultRmsThresholdDbfs
  saveDraft()
}

function setMode(nextMode: SpeechDetectionSettings['mode']) {
  if (controlsDisabled.value || nextMode === mode.value) return
  mode.value = nextMode
  emit('clearError')
  emit('save', { mode: nextMode, rmsThresholdDbfs: rmsThresholdDbfs.value })
}

function formatDbfs(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '-- dBFS'
  return `${Math.round(Math.min(0, Math.max(-96, value)))} dBFS`
}

function closeFromEscape() {
  if (!props.embedded) emit('close')
}
</script>

<template>
  <component
    :is="embedded ? 'div' : 'aside'"
    ref="panel"
    :class="['capture-settings', { 'capture-settings--embedded': embedded }]"
    :aria-labelledby="embedded ? 'speech-detection-settings-title' : 'capture-settings-title'"
    :tabindex="embedded ? undefined : -1"
    @keydown.esc="closeFromEscape"
  >
    <header v-if="!embedded" class="capture-settings__header">
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

    <header v-else class="capture-settings__embedded-heading">
      <div>
        <p>VOICE DETECTION</p>
        <h4 id="speech-detection-settings-title">人声检测</h4>
      </div>
      <span>{{ mode === 'adaptive' ? '自动' : `${rmsThresholdDbfs} dBFS` }}</span>
    </header>

    <section class="capture-settings__threshold" aria-labelledby="speech-threshold-label">
      <div class="capture-settings__mode" role="group" aria-label="语音门限模式">
        <button
          type="button"
          :class="['capture-settings__mode-option', { 'is-active': mode === 'adaptive' }]"
          :aria-pressed="mode === 'adaptive'"
          :disabled="controlsDisabled"
          data-testid="speech-mode-adaptive"
          @click="setMode('adaptive')"
        >
          <span>自动</span>
          <small>底噪 +12 dB</small>
        </button>
        <button
          type="button"
          :class="['capture-settings__mode-option', { 'is-active': mode === 'manual' }]"
          :aria-pressed="mode === 'manual'"
          :disabled="controlsDisabled"
          data-testid="speech-mode-manual"
          @click="setMode('manual')"
        >
          <span>手动</span>
          <small>固定门限</small>
        </button>
      </div>

      <div class="capture-settings__tuning">
        <div class="capture-settings__section-heading">
          <div>
            <p id="speech-threshold-label">语音门限</p>
            <output data-testid="speech-threshold-value">
              {{ mode === 'adaptive' ? '自动' : `${rmsThresholdDbfs} dBFS` }}
            </output>
          </div>
          <button
            class="icon-button capture-settings__reset"
            type="button"
            :disabled="manualControlsDisabled"
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
            min="-42"
            max="0"
            step="1"
            :value="rmsThresholdDbfs"
            :disabled="manualControlsDisabled"
            aria-label="语音 RMS 门限"
            data-testid="speech-threshold-range"
            @input="updateDraft"
            @change="saveDraft"
          />
          <label class="capture-settings__number-field">
            <span class="sr-only">语音 RMS 门限 dBFS</span>
            <input
              type="number"
              min="-42"
              max="0"
              step="1"
              :value="rmsThresholdDbfs"
              :disabled="manualControlsDisabled"
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
        <p v-else class="capture-settings__threshold-note">
          {{ mode === 'adaptive' ? '自适应范围 -42 至 -24 dBFS' : '默认手动门限 -10 dBFS' }}
        </p>
      </div>
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
  </component>
</template>
