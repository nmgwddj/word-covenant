<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import AgentActionPanel from '@/components/AgentActionPanel.vue'
import CaptureStatus from '@/components/CaptureStatus.vue'
import DevelopmentCaptureControl from '@/components/DevelopmentCaptureControl.vue'
import LiveAudioMeter from '@/components/LiveAudioMeter.vue'
import PrivacyStatus from '@/components/PrivacyStatus.vue'
import RecordingControl from '@/components/RecordingControl.vue'
import SessionRail from '@/components/SessionRail.vue'
import SettingsPage from '@/components/SettingsPage.vue'
import SpeakerManager from '@/components/SpeakerManager.vue'
import TimelinePanel from '@/components/TimelinePanel.vue'
import { usePrivacyStore } from '@/stores/privacy'
import { useModelStore } from '@/stores/models'
import { useSettingsStore } from '@/stores/settings'
import { useSessionStore } from '@/stores/session'
import { wordCovenantApi } from '@/lib/wordCovenantApi'
import type { LocalModelImportInput, SpeechDetectionSettings, VoiceProfile } from '@/types'

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
const isRefreshingInputDevices = ref(false)
const inputDeviceRefreshError = ref<string | null>(null)
const applicationSettingsButton = ref<HTMLButtonElement | null>(null)
const isSpeakerManagerOpen = computed(() =>
  sessionStore.timeline.some(span => span.id === selectedSpeakerSpanId.value && span.isFinal)
)
const selectedSessionStartNs = computed(
  () => sessionStore.selectedSession?.startedMonotonicNs ?? sessionStore.activeSession?.startedMonotonicNs ?? 0
)
const selectedSessionUsesWallClock = computed(() => sessionStore.selectedSession?.state === 'stopped')

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

async function selectSession(sessionId: string) {
  selectedSpeakerSpanId.value = null
  speakerManagerTrigger.value = null
  sessionStore.clearSpeakerError()
  await sessionStore.selectSession(sessionId)
}

async function deleteSession(sessionId: string) {
  const deletesSelectedSession = sessionStore.selectedSessionId === sessionId
  const deleted = await sessionStore.deleteSession(sessionId)
  if (deleted && deletesSelectedSession) {
    selectedSpeakerSpanId.value = null
    speakerManagerTrigger.value = null
    sessionStore.clearSpeakerError()
  }
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

function openApplicationSettings() {
  selectedSpeakerSpanId.value = null
  speakerManagerTrigger.value = null
  sessionStore.clearSpeakerError()
  activeView.value = 'settings'
}

function closeApplicationSettings() {
  activeView.value = 'workspace'
  void nextTick(() => applicationSettingsButton.value?.focus())
}

async function saveSpeechDetection(settings: SpeechDetectionSettings) {
  await settingsStore.setSpeechDetection(settings)
}

async function renameVoiceProfile(profile: VoiceProfile, displayName: string) {
  await settingsStore.renameVoiceProfile(profile, displayName)
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
  consent: boolean
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
      :speech-detection-settings="settingsStore.speechDetection"
      :capture-meter="sessionStore.capture.meter"
      :capture-settings-locked="isCaptureSettingsLocked"
      :loading-speech-detection="settingsStore.isLoadingSpeechDetection"
      :saving-speech-detection="settingsStore.isSavingSpeechDetection"
      :speech-detection-error="settingsStore.speechDetectionError"
      :compatible-asr-models="modelStore.compatibleAsrModels"
      :bundled-asr-status="modelStore.bundledAsrStatus"
      :active-asr-profile="modelStore.activeAsrProfile"
      :importing="modelStore.isImporting"
      :selecting-active-asr-model="modelStore.isSelectingActiveAsrModel"
      :active-asr-selection-disabled="sessionStore.isRecording"
      :error="modelStore.importError"
      :active-asr-error="modelStore.activeAsrError"
      :select-source-path="selectLocalModelFile"
      :voice-profiles="settingsStore.voiceProfiles"
      :loading-voice-profiles="settingsStore.isLoadingVoiceProfiles"
      :pending-voice-profile-id="settingsStore.pendingVoiceProfileId"
      :voice-profile-error="settingsStore.voiceProfileError"
      @close="closeApplicationSettings"
      @clear-error="clearLocalModelImportError"
      @clear-active-asr-error="modelStore.clearActiveAsrError"
      @select-input-device="selectInputDevice"
      @refresh-input-devices="refreshInputDevices"
      @clear-speech-detection-error="settingsStore.clearSpeechDetectionError"
      @save-speech-detection="saveSpeechDetection"
      @import="importLocalModel"
      @select-active-asr-model="selectActiveLocalAsrModel"
      @clear-voice-profile-error="settingsStore.clearVoiceProfileError"
      @rename-voice-profile="renameVoiceProfile"
      @relearn-voice-profile="settingsStore.relearnVoiceProfile"
      @add-voice-profile-sample="settingsStore.addVoiceProfileConfirmedSample"
      @delete-voice-profile="settingsStore.deleteVoiceProfile"
    />

    <section v-else class="workspace-grid">
      <SessionRail
        :sessions="sessionStore.sessions"
        :selected-session-id="sessionStore.selectedSessionId"
        :recording="sessionStore.isRecording"
        :loading="sessionStore.isSessionHistoryLoading"
        :deleting-session-id="sessionStore.deletingSessionId"
        :error="sessionStore.sessionHistoryError"
        @select="selectSession"
        @delete="deleteSession"
      />

      <TimelinePanel
        :spans="sessionStore.timeline"
        :speaker-clusters="sessionStore.speakerClusters"
        :session-start-ns="selectedSessionStartNs"
        :use-wall-clock="selectedSessionUsesWallClock"
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
      :session-start-ns="selectedSessionStartNs"
      :use-wall-clock="selectedSessionUsesWallClock"
      :pending="sessionStore.isSpeakerOperationPending"
      :error="sessionStore.speakerError"
      @close="closeSpeakerManager"
      @create="createSpeakerCluster"
      @rename="renameSpeakerCluster"
      @reassign="reassignTranscriptSpeaker"
      @clear-error="sessionStore.clearSpeakerError"
    />
  </main>
</template>
