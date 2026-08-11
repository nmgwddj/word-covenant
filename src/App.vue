<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import AgentActionPanel from '@/components/AgentActionPanel.vue'
import CaptureSettingsPanel from '@/components/CaptureSettingsPanel.vue'
import CaptureStatus from '@/components/CaptureStatus.vue'
import DevelopmentCaptureControl from '@/components/DevelopmentCaptureControl.vue'
import LiveAudioMeter from '@/components/LiveAudioMeter.vue'
import PrivacyStatus from '@/components/PrivacyStatus.vue'
import RecordingControl from '@/components/RecordingControl.vue'
import SettingsPage from '@/components/SettingsPage.vue'
import SpeakerManager from '@/components/SpeakerManager.vue'
import TimelinePanel from '@/components/TimelinePanel.vue'
import { usePrivacyStore } from '@/stores/privacy'
import { useModelStore } from '@/stores/models'
import { useSettingsStore } from '@/stores/settings'
import { useSessionStore } from '@/stores/session'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { LocalModelImportInput } from '@/types'

const privacyStore = usePrivacyStore()
const modelStore = useModelStore()
const settingsStore = useSettingsStore()
const sessionStore = useSessionStore()
const isDevelopmentBuild = import.meta.env.DEV
const activeView = ref<'workspace' | 'settings'>('workspace')
const isSettingsView = computed(() => activeView.value === 'settings')
const asrModelReadyForCurrentInput = computed(
  () => sessionStore.captureInput === 'development_mock' || modelStore.hasActiveCompatibleAsrModel
)
const isNativeMicrophoneLifecycleActive = computed(() => {
  if (sessionStore.captureInput !== 'microphone') return false

  const bridgeStatus = sessionStore.capture.bridge?.status
  if (bridgeStatus === 'drained') return false

  return (
    sessionStore.capture.status === 'recording' ||
    bridgeStatus === 'parked' ||
    bridgeStatus === 'armed' ||
    bridgeStatus === 'closing'
  )
})
const recordingControlDisabled = computed(
  () =>
    sessionStore.isLoading ||
    sessionStore.isAwaitingPermission ||
    (!sessionStore.isRecording && !asrModelReadyForCurrentInput.value)
)
const recordingControlHint = computed(() =>
  !sessionStore.isRecording && !asrModelReadyForCurrentInput.value ? '请选择兼容的本地转写模型后开始录音' : undefined
)
const isCaptureSettingsLocked = computed(
  () =>
    sessionStore.isLoading ||
    sessionStore.isAwaitingPermission ||
    sessionStore.isRecording ||
    isNativeMicrophoneLifecycleActive.value
)
let developmentMockTimer: number | undefined
let unlistenCaptureProjection: (() => void) | undefined
let unlistenFinalTranscriptProjection: (() => void) | undefined
const selectedSpeakerSpanId = ref<string | null>(null)
const speakerManagerTrigger = ref<HTMLElement | null>(null)
const isCaptureSettingsOpen = ref(false)
const captureSettingsTrigger = ref<HTMLElement | null>(null)
const isRefreshingInputDevices = ref(false)
const inputDeviceRefreshError = ref<string | null>(null)
const applicationSettingsButton = ref<HTMLButtonElement | null>(null)
const isSpeakerManagerOpen = computed(() =>
  sessionStore.timeline.some(span => span.id === selectedSpeakerSpanId.value && span.isFinal)
)

onMounted(async () => {
  unlistenCaptureProjection = await wordCovenantApi.onCaptureProjection(projection => {
    sessionStore.applyCaptureProjection(projection)
  })
  unlistenFinalTranscriptProjection = await wordCovenantApi.onFinalTranscriptProjection(projection => {
    void sessionStore.applyFinalTranscriptProjection(projection)
  })
  await Promise.all([
    privacyStore.refresh(),
    sessionStore.initialize(),
    modelStore.initialize(),
    settingsStore.initialize(),
  ])
})

async function toggleRecording() {
  await sessionStore.toggleRecording()
  await privacyStore.refresh()
  synchronizeDevelopmentMockTimer()
}

async function setEgressEnabled(enabled: boolean) {
  await privacyStore.setEgressEnabled(enabled)
}

function toggleDevelopmentCaptureInput() {
  sessionStore.setCaptureInput(sessionStore.captureInput === 'development_mock' ? 'microphone' : 'development_mock')
}

async function selectInputDevice(deviceUid: string) {
  if (deviceUid) {
    await sessionStore.selectInputDevice(deviceUid)
    inputDeviceRefreshError.value = null
  }
}

async function refreshInputDevices() {
  if (isRefreshingInputDevices.value) return

  isRefreshingInputDevices.value = true
  inputDeviceRefreshError.value = null
  try {
    sessionStore.applyCaptureProjection(await wordCovenantApi.getCaptureProjection())
  } catch {
    inputDeviceRefreshError.value = '无法刷新输入设备'
  } finally {
    isRefreshingInputDevices.value = false
  }
}

function openCaptureSettings() {
  captureSettingsTrigger.value =
    typeof document !== 'undefined' && document.activeElement instanceof HTMLElement ? document.activeElement : null
  isCaptureSettingsOpen.value = true
}

function closeCaptureSettings() {
  const trigger = captureSettingsTrigger.value
  isCaptureSettingsOpen.value = false
  captureSettingsTrigger.value = null
  void nextTick(() => trigger?.focus())
}

function openApplicationSettings() {
  isCaptureSettingsOpen.value = false
  captureSettingsTrigger.value = null
  selectedSpeakerSpanId.value = null
  speakerManagerTrigger.value = null
  sessionStore.clearSpeakerError()
  activeView.value = 'settings'
}

function closeApplicationSettings() {
  activeView.value = 'workspace'
  void nextTick(() => applicationSettingsButton.value?.focus())
}

async function saveSpeechDetectionThreshold(rmsThresholdDbfs: number) {
  await settingsStore.setRmsThresholdDbfs(rmsThresholdDbfs)
}

async function importLocalModel(input: LocalModelImportInput) {
  await modelStore.importLocalModel(input)
}

async function selectActiveLocalAsrModel(modelId: string) {
  try {
    await modelStore.selectActiveLocalAsrModel(modelId)
  } catch {
    // The model panel renders the local selection error without exposing native details.
  }
}

async function selectLocalModelFile() {
  return modelStore.selectLocalModelFile()
}

function clearLocalModelImportError() {
  modelStore.clearImportError()
}

function openSpeakerManager(spanId: string) {
  const selectedSpan = sessionStore.timeline.find(span => span.id === spanId)
  if (!selectedSpan?.isFinal) return

  speakerManagerTrigger.value =
    typeof document !== 'undefined' && document.activeElement instanceof HTMLElement ? document.activeElement : null
  selectedSpeakerSpanId.value = spanId
  sessionStore.clearSpeakerError()
}

function closeSpeakerManager() {
  const trigger = speakerManagerTrigger.value
  selectedSpeakerSpanId.value = null
  speakerManagerTrigger.value = null
  sessionStore.clearSpeakerError()
  void nextTick(() => trigger?.focus())
}

async function createSpeakerCluster(sessionId: string) {
  await sessionStore.createSpeakerCluster(sessionId)
}

async function renameSpeakerCluster(input: {
  sessionId: string
  clusterId: string
  expectedLabelRevision: number
  label: string
}) {
  await sessionStore.renameSpeakerCluster(input)
}

async function reassignTranscriptSpeaker(input: {
  sessionId: string
  logicalSpanId: string
  expectedRevision: number
  targetClusterId: string | null
}) {
  await sessionStore.reassignTranscriptSpeaker(input)
}

function stopDevelopmentMockTimer() {
  if (developmentMockTimer !== undefined) {
    window.clearInterval(developmentMockTimer)
    developmentMockTimer = undefined
  }
}

function synchronizeDevelopmentMockTimer() {
  if (!isDevelopmentBuild || !sessionStore.isDevelopmentMockActive) {
    stopDevelopmentMockTimer()
    return
  }
  if (developmentMockTimer !== undefined) return

  developmentMockTimer = window.setInterval(() => {
    void advanceDevelopmentMock()
  }, 200)
}

async function advanceDevelopmentMock() {
  try {
    await sessionStore.advanceDevelopmentMock()
  } catch {
    stopDevelopmentMockTimer()
  }
  synchronizeDevelopmentMockTimer()
}

watch(() => sessionStore.isDevelopmentMockActive, synchronizeDevelopmentMockTimer)

watch(
  () => sessionStore.capture.devices.length,
  count => {
    if (count > 0) inputDeviceRefreshError.value = null
  }
)

onBeforeUnmount(() => {
  stopDevelopmentMockTimer()
  unlistenCaptureProjection?.()
  unlistenFinalTranscriptProjection?.()
})
</script>

<template>
  <main class="workspace-shell">
    <header class="workspace-header">
      <div class="brand-lockup">
        <div class="brand-lockup__mark" aria-hidden="true"><span /></div>
        <div>
          <h1>WordCovenant</h1>
          <p>凡口头所言，皆立为契约。有据可查，事事落单。</p>
        </div>
      </div>

      <div class="workspace-header__actions">
        <button
          v-if="!isSettingsView"
          ref="applicationSettingsButton"
          class="icon-button application-settings-toggle"
          type="button"
          aria-label="应用设置"
          title="应用设置"
          data-testid="open-application-settings"
          @click="openApplicationSettings"
        >
          <span class="i-mdi-cog-outline" aria-hidden="true" />
        </button>
        <button
          class="icon-button capture-settings-toggle"
          type="button"
          :aria-pressed="isCaptureSettingsOpen"
          aria-label="录音检测设置"
          title="录音检测设置"
          data-testid="open-capture-settings"
          @click="openCaptureSettings"
        >
          <span class="i-mdi-tune-variant" aria-hidden="true" />
        </button>
        <DevelopmentCaptureControl
          v-if="isDevelopmentBuild"
          :selected="sessionStore.captureInput === 'development_mock'"
          :disabled="sessionStore.isLoading || sessionStore.isRecording"
          @select="toggleDevelopmentCaptureInput"
        />
        <RecordingControl
          :recording="sessionStore.isRecording"
          :disabled="recordingControlDisabled"
          :title="recordingControlHint"
          @toggle="toggleRecording"
        />
      </div>
    </header>

    <SettingsPage
      v-if="isSettingsView"
      :models="modelStore.models"
      :capture="sessionStore.capture"
      :input-device-selection-disabled="sessionStore.isLoading || sessionStore.isDevelopmentMockActive"
      :refreshing-input-devices="isRefreshingInputDevices"
      :input-device-refresh-error="inputDeviceRefreshError"
      :compatible-asr-models="modelStore.compatibleAsrModels"
      :bundled-asr-status="modelStore.bundledAsrStatus"
      :active-asr-profile="modelStore.activeAsrProfile"
      :importing="modelStore.isImporting"
      :selecting-active-asr-model="modelStore.isSelectingActiveAsrModel"
      :active-asr-selection-disabled="sessionStore.isRecording"
      :error="modelStore.importError"
      :active-asr-error="modelStore.activeAsrError"
      :select-source-path="selectLocalModelFile"
      @close="closeApplicationSettings"
      @clear-error="clearLocalModelImportError"
      @clear-active-asr-error="modelStore.clearActiveAsrError"
      @select-input-device="selectInputDevice"
      @refresh-input-devices="refreshInputDevices"
      @import="importLocalModel"
      @select-active-asr-model="selectActiveLocalAsrModel"
    />

    <section v-else class="workspace-grid">
      <aside class="session-rail" aria-label="本地会话">
        <p class="session-rail__label">LOCAL ARCHIVE</p>
        <button class="session-item session-item--active" type="button">
          <span class="session-item__dot" aria-hidden="true" />
          <span>当前会话</span>
          <time>今天</time>
        </button>
        <button class="icon-button session-rail__new" type="button" title="新建本地会话">
          <span class="i-mdi-plus" aria-hidden="true" />
        </button>
      </aside>

      <TimelinePanel
        :spans="sessionStore.timeline"
        :speaker-clusters="sessionStore.speakerClusters"
        :session-start-ns="sessionStore.activeSession?.startedMonotonicNs ?? 0"
        :use-wall-clock="!sessionStore.activeSession"
        @open-speaker-manager="openSpeakerManager"
      />
      <AgentActionPanel
        :actions="sessionStore.actions"
        :egress-enabled="privacyStore.status.egressEnabled"
        :active-egress-approvals="privacyStore.status.activeEgressApprovals"
        :egress-loading="privacyStore.isUpdatingEgress"
        @propose="sessionStore.proposeLocalSpeech"
        @set-egress-enabled="setEgressEnabled"
      />
    </section>

    <footer class="workspace-statusbar" aria-label="应用运行状态">
      <div class="workspace-statusbar__runtime">
        <CaptureStatus
          :capture="sessionStore.capture"
          :asr-model-ready="asrModelReadyForCurrentInput"
          :development-mock-active="sessionStore.isDevelopmentMockActive"
        />
        <LiveAudioMeter v-if="isNativeMicrophoneLifecycleActive" :meter="sessionStore.capture.meter" active />
      </div>
      <PrivacyStatus :status="privacyStore.status" />
    </footer>

    <SpeakerManager
      v-if="isSpeakerManagerOpen"
      :clusters="sessionStore.speakerClusters"
      :spans="sessionStore.timeline"
      :selected-span-id="selectedSpeakerSpanId"
      :session-start-ns="sessionStore.activeSession?.startedMonotonicNs ?? 0"
      :use-wall-clock="!sessionStore.activeSession"
      :pending="sessionStore.isSpeakerOperationPending"
      :error="sessionStore.speakerError"
      @close="closeSpeakerManager"
      @create="createSpeakerCluster"
      @rename="renameSpeakerCluster"
      @reassign="reassignTranscriptSpeaker"
      @clear-error="sessionStore.clearSpeakerError"
    />

    <CaptureSettingsPanel
      v-if="isCaptureSettingsOpen"
      :settings="settingsStore.speechDetection"
      :meter="sessionStore.capture.meter"
      :locked="isCaptureSettingsLocked"
      :loading="settingsStore.isLoadingSpeechDetection"
      :saving="settingsStore.isSavingSpeechDetection"
      :error="settingsStore.speechDetectionError"
      @close="closeCaptureSettings"
      @clear-error="settingsStore.clearSpeechDetectionError"
      @save="saveSpeechDetectionThreshold"
    />
  </main>
</template>
