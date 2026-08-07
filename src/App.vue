<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, watch } from 'vue'
import AgentActionPanel from '@/components/AgentActionPanel.vue'
import DevelopmentCaptureControl from '@/components/DevelopmentCaptureControl.vue'
import PrivacyStatus from '@/components/PrivacyStatus.vue'
import RecordingControl from '@/components/RecordingControl.vue'
import TimelinePanel from '@/components/TimelinePanel.vue'
import { usePrivacyStore } from '@/stores/privacy'
import { useSessionStore } from '@/stores/session'

const privacyStore = usePrivacyStore()
const sessionStore = useSessionStore()
const isDevelopmentBuild = import.meta.env.DEV
const recordingLabel = computed(() => {
  if (!sessionStore.isRecording) return '待命'
  return sessionStore.captureInput === 'development_mock' ? '模拟记录中' : '记录中'
})
let developmentMockTimer: number | undefined

onMounted(async () => {
  await Promise.all([privacyStore.refresh(), sessionStore.initialize()])
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
  sessionStore.setCaptureInput(
    sessionStore.captureInput === 'development_mock' ? 'microphone' : 'development_mock',
  )
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

onBeforeUnmount(stopDevelopmentMockTimer)
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
        <PrivacyStatus :status="privacyStore.status" />
        <span class="recording-state" :class="{ 'recording-state--active': sessionStore.isRecording }">
          <span aria-hidden="true" />{{ recordingLabel }}
        </span>
        <DevelopmentCaptureControl
          v-if="isDevelopmentBuild"
          :selected="sessionStore.captureInput === 'development_mock'"
          :disabled="sessionStore.isLoading || sessionStore.isRecording"
          @select="toggleDevelopmentCaptureInput"
        />
        <RecordingControl
          :recording="sessionStore.isRecording"
          :disabled="sessionStore.isLoading"
          @toggle="toggleRecording"
        />
      </div>
    </header>

    <section class="workspace-grid">
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
        :session-start-ns="sessionStore.activeSession?.startedMonotonicNs ?? 0"
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
  </main>
</template>
