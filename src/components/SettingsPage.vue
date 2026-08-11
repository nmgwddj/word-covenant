<script setup lang="ts">
import CaptureSettingsPanel from '@/components/CaptureSettingsPanel.vue'
import InputDeviceSettings from '@/components/InputDeviceSettings.vue'
import ModelRegistryPanel from '@/components/ModelRegistryPanel.vue'
import type {
  ActiveLocalAsrProfile,
  BundledAsrStatus,
  CaptureMeter,
  CaptureProjection,
  LocalModelImportInput,
  RegisteredModel,
  SpeechDetectionSettings,
} from '@/types'

withDefaults(
  defineProps<{
    models: RegisteredModel[]
    capture: CaptureProjection
    inputDeviceSelectionDisabled?: boolean
    refreshingInputDevices?: boolean
    inputDeviceRefreshError?: string | null
    speechDetectionSettings?: SpeechDetectionSettings
    captureMeter?: CaptureMeter | null
    captureSettingsLocked?: boolean
    loadingSpeechDetection?: boolean
    savingSpeechDetection?: boolean
    speechDetectionError?: string | null
    compatibleAsrModels?: RegisteredModel[]
    bundledAsrStatus?: BundledAsrStatus | null
    activeAsrProfile?: ActiveLocalAsrProfile | null
    importing?: boolean
    selectingActiveAsrModel?: boolean
    activeAsrSelectionDisabled?: boolean
    error?: string | null
    activeAsrError?: string | null
    selectSourcePath?: () => Promise<string | null>
  }>(),
  {
    bundledAsrStatus: null,
    inputDeviceSelectionDisabled: false,
    refreshingInputDevices: false,
    inputDeviceRefreshError: null,
    speechDetectionSettings: () => ({ mode: 'adaptive', rmsThresholdDbfs: -10 }),
    captureMeter: null,
    captureSettingsLocked: false,
    loadingSpeechDetection: false,
    savingSpeechDetection: false,
    speechDetectionError: null,
    activeAsrProfile: null,
    importing: false,
    selectingActiveAsrModel: false,
    activeAsrSelectionDisabled: false,
    error: null,
    activeAsrError: null,
    selectSourcePath: undefined,
  }
)

const emit = defineEmits<{
  close: []
  import: [input: LocalModelImportInput]
  selectActiveAsrModel: [modelId: string]
  clearError: []
  clearActiveAsrError: []
  selectInputDevice: [deviceUid: string]
  refreshInputDevices: []
  saveSpeechDetection: [settings: SpeechDetectionSettings]
  clearSpeechDetectionError: []
}>()
</script>

<template>
  <section class="settings-page" aria-labelledby="settings-page-title">
    <div class="settings-page__frame">
      <header class="settings-page__header">
        <button
          class="icon-button settings-page__back"
          type="button"
          aria-label="返回工作台"
          title="返回工作台"
          @click="emit('close')"
        >
          <span class="i-mdi-arrow-left" aria-hidden="true" />
        </button>
        <div>
          <p class="settings-page__eyebrow">LOCAL SETTINGS</p>
          <h2 id="settings-page-title">设置</h2>
        </div>
      </header>

      <section class="settings-page__section" aria-labelledby="recording-settings-title">
        <header class="settings-page__section-heading">
          <span class="settings-page__section-icon i-mdi-microphone-outline" aria-hidden="true" />
          <div>
            <p>LOCAL CAPTURE</p>
            <h3 id="recording-settings-title">录音与检测</h3>
          </div>
          <span class="settings-page__local-state">本机</span>
        </header>

        <div class="settings-page__capture-group">
          <InputDeviceSettings
            :capture="capture"
            :disabled="inputDeviceSelectionDisabled"
            :refreshing="refreshingInputDevices"
            :refresh-error="inputDeviceRefreshError"
            @select="emit('selectInputDevice', $event)"
            @refresh="emit('refreshInputDevices')"
          />
          <CaptureSettingsPanel
            embedded
            :settings="speechDetectionSettings"
            :meter="captureMeter"
            :locked="captureSettingsLocked"
            :loading="loadingSpeechDetection"
            :saving="savingSpeechDetection"
            :error="speechDetectionError"
            @clear-error="emit('clearSpeechDetectionError')"
            @save="emit('saveSpeechDetection', $event)"
          />
        </div>
      </section>

      <section class="settings-page__section" aria-labelledby="model-settings-title">
        <header class="settings-page__section-heading">
          <span class="settings-page__section-icon i-mdi-waveform" aria-hidden="true" />
          <div>
            <p>LOCAL INFERENCE</p>
            <h3 id="model-settings-title">模型与转写</h3>
          </div>
          <span class="settings-page__local-state">本机 · 离线</span>
        </header>

        <ModelRegistryPanel
          :models="models"
          :compatible-asr-models="compatibleAsrModels"
          :bundled-asr-status="bundledAsrStatus"
          :active-asr-profile="activeAsrProfile"
          :importing="importing"
          :selecting-active-asr-model="selectingActiveAsrModel"
          :active-asr-selection-disabled="activeAsrSelectionDisabled"
          :error="error"
          :active-asr-error="activeAsrError"
          :select-source-path="selectSourcePath"
          @clear-error="emit('clearError')"
          @clear-active-asr-error="emit('clearActiveAsrError')"
          @import="emit('import', $event)"
          @select-active-asr-model="emit('selectActiveAsrModel', $event)"
        />
      </section>
    </div>
  </section>
</template>
