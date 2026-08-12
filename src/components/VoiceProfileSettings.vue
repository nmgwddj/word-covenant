<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { VoiceProfile } from '@/types'

const props = withDefaults(
  defineProps<{
    profiles: VoiceProfile[]
    loading?: boolean
    pendingProfileId?: string | null
    error?: string | null
  }>(),
  { loading: false, pendingProfileId: null, error: null }
)

const emit = defineEmits<{
  rename: [profile: VoiceProfile, displayName: string]
  relearn: [profile: VoiceProfile]
  addSample: [profile: VoiceProfile]
  delete: [profile: VoiceProfile]
  clearError: []
}>()

const edits = ref<Record<string, string>>({})
const deleteCandidate = ref<VoiceProfile | null>(null)

watch(
  () => props.profiles,
  profiles => {
    edits.value = Object.fromEntries(profiles.map(profile => [profile.id, profile.displayName]))
  },
  { deep: true, immediate: true }
)

function profileName(profile: VoiceProfile): string {
  return edits.value[profile.id] ?? profile.displayName
}

const empty = computed(() => !props.loading && props.profiles.length === 0)

function stateLabel(profile: VoiceProfile): string {
  if (profile.state === 'ready') return '可自动识别'
  if (profile.state === 'relearn_required') return '需重新学习'
  return '学习中'
}

function progress(profile: VoiceProfile): number {
  if (profile.readyConfirmedDurationNs <= 0) return 0
  return Math.min(100, Math.round((profile.confirmedDurationNs / profile.readyConfirmedDurationNs) * 100))
}

function seconds(ns: number): string {
  return `${(ns / 1_000_000_000).toFixed(1)} 秒`
}

function dateTime(value: string | null): string {
  if (!value) return '尚未确认'
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return '尚未确认'
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(date)
}

function submitRename(profile: VoiceProfile) {
  const name = edits.value[profile.id]?.trim() ?? ''
  if (!name || name === profile.displayName || props.pendingProfileId) return
  emit('clearError')
  emit('rename', profile, name)
}
</script>

<template>
  <div class="voice-profiles">
    <p v-if="loading" class="voice-profiles__empty" aria-live="polite">正在读取本机声纹档案…</p>
    <p v-else-if="empty" class="voice-profiles__empty">为匿名说话人命名后，声纹档案会显示在这里。</p>

    <ul v-else class="voice-profiles__list">
      <li v-for="profile in profiles" :key="profile.id" class="voice-profile">
        <div class="voice-profile__identity">
          <form @submit.prevent="submitRename(profile)">
            <label class="sr-only" :for="`voice-profile-name-${profile.id}`">声纹档案名称</label>
            <input
              :id="`voice-profile-name-${profile.id}`"
              :value="profileName(profile)"
              type="text"
              maxlength="80"
              autocomplete="off"
              :disabled="Boolean(pendingProfileId)"
              @input="edits[profile.id] = ($event.target as HTMLInputElement).value"
            />
            <button
              class="icon-button"
              type="submit"
              :aria-label="`保存${profile.displayName}的新名称`"
              title="保存名称"
              :disabled="
                Boolean(pendingProfileId) ||
                !profileName(profile).trim() ||
                profileName(profile).trim() === profile.displayName
              "
            >
              <span class="i-mdi-check" aria-hidden="true" />
            </button>
          </form>
          <span class="voice-profile__state" :data-state="profile.state">{{ stateLabel(profile) }}</span>
        </div>

        <div class="voice-profile__learning">
          <div class="voice-profile__progress" aria-hidden="true"><span :style="{ width: `${progress(profile)}%` }" /></div>
          <p>{{ seconds(profile.confirmedDurationNs) }} / {{ seconds(profile.readyConfirmedDurationNs) }}</p>
        </div>

        <dl class="voice-profile__meta">
          <div><dt>模型</dt><dd>{{ profile.modelVersion }}</dd></div>
          <div><dt>最近确认</dt><dd>{{ dateTime(profile.lastConfirmationAt) }}</dd></div>
        </dl>

        <div class="voice-profile__actions">
          <button
            type="button"
            :disabled="Boolean(pendingProfileId) || !profile.canAddConfirmedSample"
            @click="emit('addSample', profile)"
          >
            <span class="i-mdi-waveform-plus" aria-hidden="true" />
            添加确认样本
          </button>
          <button type="button" :disabled="Boolean(pendingProfileId)" @click="emit('relearn', profile)">
            <span class="i-mdi-refresh" aria-hidden="true" />
            重新学习
          </button>
          <button
            class="voice-profile__delete"
            type="button"
            :disabled="Boolean(pendingProfileId)"
            :aria-label="`永久删除${profile.displayName}的声纹档案`"
            @click="deleteCandidate = profile"
          >
            <span class="i-mdi-delete-outline" aria-hidden="true" />
            删除
          </button>
        </div>
      </li>
    </ul>

    <p v-if="error" class="voice-profiles__error" role="alert">{{ error }}</p>

    <section
      v-if="deleteCandidate"
      class="voice-profiles__confirm"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="delete-voice-profile-title"
    >
      <span class="i-mdi-fingerprint-off" aria-hidden="true" />
      <h4 id="delete-voice-profile-title">永久删除“{{ deleteCandidate.displayName }}”的声纹？</h4>
      <p>删除后将停止未来的自动识别，并物理清除声纹向量；历史转写中的姓名快照仍会保留。</p>
      <div>
        <button type="button" @click="deleteCandidate = null">取消</button>
        <button
          type="button"
          data-testid="confirm-delete-voice-profile"
          @click="emit('delete', deleteCandidate); deleteCandidate = null"
        >
          永久删除
        </button>
      </div>
    </section>
  </div>
</template>
