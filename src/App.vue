<script setup lang="ts">
import { computed, onMounted } from 'vue'
import AgentActionPanel from '@/components/AgentActionPanel.vue'
import PrivacyStatus from '@/components/PrivacyStatus.vue'
import RecordingControl from '@/components/RecordingControl.vue'
import TimelinePanel from '@/components/TimelinePanel.vue'
import { usePrivacyStore } from '@/stores/privacy'
import { useSessionStore } from '@/stores/session'

const privacyStore = usePrivacyStore()
const sessionStore = useSessionStore()
const recordingLabel = computed(() => (sessionStore.isRecording ? '记录中' : '待命'))

onMounted(async () => {
  await Promise.all([privacyStore.refresh(), sessionStore.initialize()])
})

async function toggleRecording() {
  await sessionStore.toggleRecording()
  await privacyStore.refresh()
}

async function setEgressEnabled(enabled: boolean) {
  await privacyStore.setEgressEnabled(enabled)
}
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

      <TimelinePanel :spans="sessionStore.timeline" />
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
