<script setup lang="ts">
import { computed, ref } from 'vue'
import type { SessionSummary } from '@/types'

const props = withDefaults(
  defineProps<{
    sessions: SessionSummary[]
    selectedSessionId: string | null
    recording: boolean
    loading?: boolean
    deletingSessionId?: string | null
    error?: string | null
  }>(),
  {
    loading: false,
    deletingSessionId: null,
    error: null,
  }
)

const emit = defineEmits<{
  select: [sessionId: string]
  delete: [sessionId: string]
}>()

const pendingDeletion = ref<SessionSummary | null>(null)

const dateFormatter = new Intl.DateTimeFormat('zh-CN', {
  year: 'numeric',
  month: 'numeric',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23',
})

const displayedSessions = computed(() =>
  props.sessions.map(session => ({
    session,
    startedAtLabel: formatStartTime(session.startedAt),
    durationLabel: formatDuration(session),
    transcriptCountLabel: `${Math.max(0, Math.floor(session.transcriptCount)).toLocaleString('zh-CN')} 条转写`,
  }))
)

function formatStartTime(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return '时间未知'

  const parts = Object.fromEntries(dateFormatter.formatToParts(date).map(part => [part.type, part.value]))
  return `${parts.year}年${parts.month}月${parts.day}日 ${parts.hour}:${parts.minute}`
}

function formatDuration(session: SessionSummary): string {
  if (session.state === 'starting' || session.state === 'recording') return '录音中'
  if (!session.stoppedAt) return '未正常结束'

  const durationSeconds = Math.max(
    0,
    Math.floor((new Date(session.stoppedAt).valueOf() - new Date(session.startedAt).valueOf()) / 1_000)
  )
  if (!Number.isFinite(durationSeconds)) return '--:--'

  const hours = Math.floor(durationSeconds / 3_600)
  const minutes = Math.floor((durationSeconds % 3_600) / 60)
  const seconds = durationSeconds % 60
  const clock = [minutes, seconds].map(part => String(part).padStart(2, '0')).join(':')

  return hours > 0 ? `${String(hours).padStart(2, '0')}:${clock}` : clock
}

function selectSession(sessionId: string) {
  if (props.recording || props.loading || props.deletingSessionId || sessionId === props.selectedSessionId) return
  emit('select', sessionId)
}

function requestDeletion(session: SessionSummary) {
  if (props.recording || props.loading || props.deletingSessionId || session.state !== 'stopped') return
  pendingDeletion.value = session
}

function cancelDeletion() {
  if (props.deletingSessionId) return
  pendingDeletion.value = null
}

function confirmDeletion() {
  const session = pendingDeletion.value
  if (!session || props.deletingSessionId) return
  pendingDeletion.value = null
  emit('delete', session.id)
}
</script>

<template>
  <aside class="session-rail" aria-label="本地会话" :aria-busy="props.loading">
    <div class="session-rail__header">
      <p class="session-rail__label">LOCAL ARCHIVE</p>
      <span v-if="props.loading && props.sessions.length" class="session-rail__loading" aria-label="正在更新会话">
        <span class="i-mdi-loading" aria-hidden="true" />
      </span>
    </div>

    <div v-if="props.error" class="session-rail__notice session-rail__notice--error" role="alert">
      <span class="i-mdi-alert-circle-outline" aria-hidden="true" />
      <span>{{ props.error }}</span>
    </div>

    <ol v-if="displayedSessions.length" class="session-rail__list">
      <li v-for="item in displayedSessions" :key="item.session.id" class="session-item-row">
        <button
          class="session-item"
          :class="{
            'session-item--active': item.session.id === props.selectedSessionId,
            'session-item--recording': item.session.state !== 'stopped',
          }"
          type="button"
          :disabled="props.recording || props.loading || Boolean(props.deletingSessionId)"
          :aria-current="item.session.id === props.selectedSessionId ? 'true' : undefined"
          :title="props.recording ? '录音期间不可切换会话' : undefined"
          :data-session-id="item.session.id"
          @click="selectSession(item.session.id)"
        >
          <span class="session-item__dot" aria-hidden="true" />
          <span class="session-item__body">
            <time class="session-item__start" :datetime="item.session.startedAt">
              {{ item.startedAtLabel }}
            </time>
            <span class="session-item__meta">
              <span>{{ item.durationLabel }}</span>
              <span aria-hidden="true">/</span>
              <span>{{ item.transcriptCountLabel }}</span>
            </span>
          </span>
        </button>
        <button
          v-if="item.session.state === 'stopped'"
          class="session-item__delete"
          type="button"
          :disabled="props.recording || props.loading || Boolean(props.deletingSessionId)"
          :aria-label="`删除 ${item.startedAtLabel} 的本地会话`"
          :title="props.recording ? '录音期间不可删除会话' : '删除本地会话'"
          :data-delete-session-id="item.session.id"
          @click="requestDeletion(item.session)"
        >
          <span
            :class="props.deletingSessionId === item.session.id ? 'i-mdi-loading' : 'i-mdi-delete-outline'"
            aria-hidden="true"
          />
        </button>
      </li>
    </ol>

    <div v-else-if="props.loading && !props.error" class="session-rail__notice" role="status">
      <span class="i-mdi-loading session-rail__notice-icon--spinning" aria-hidden="true" />
      <span>正在读取本地会话</span>
    </div>

    <div v-else-if="!props.error" class="session-rail__notice session-rail__notice--empty">
      <span class="i-mdi-archive-clock-outline" aria-hidden="true" />
      <span>暂无本地会话</span>
    </div>

    <Teleport to="body">
      <div v-if="pendingDeletion" class="session-delete-dialog" role="presentation" @click.self="cancelDeletion">
        <section
          class="session-delete-dialog__panel"
          role="alertdialog"
          aria-modal="true"
          aria-labelledby="session-delete-title"
          aria-describedby="session-delete-description"
        >
          <span class="i-mdi-delete-outline session-delete-dialog__icon" aria-hidden="true" />
          <div class="session-delete-dialog__copy">
            <h2 id="session-delete-title">删除这次本地会话？</h2>
            <p id="session-delete-description">
              {{ formatStartTime(pendingDeletion.startedAt) }} 的转写、说话人和本地检索记录将永久删除。
            </p>
          </div>
          <div class="session-delete-dialog__actions">
            <button type="button" class="session-delete-dialog__cancel" autofocus @click="cancelDeletion">取消</button>
            <button type="button" class="session-delete-dialog__confirm" @click="confirmDeletion">删除</button>
          </div>
        </section>
      </div>
    </Teleport>
  </aside>
</template>
