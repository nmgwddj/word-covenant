<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import type { SpeakerCluster, TranscriptSpan } from '@/types'

const props = withDefaults(
  defineProps<{
    clusters: SpeakerCluster[]
    spans: TranscriptSpan[]
    selectedSpanId: string | null
    sessionStartNs?: number
    useWallClock?: boolean
    pending?: boolean
    error?: string | null
  }>(),
  {
    sessionStartNs: 0,
    useWallClock: false,
    pending: false,
    error: null,
  }
)

const emit = defineEmits<{
  close: []
  create: [sessionId: string]
  rename: [
    input: {
      sessionId: string
      clusterId: string
      expectedLabelRevision: number
      label: string
    },
  ]
  reassign: [
    input: {
      sessionId: string
      logicalSpanId: string
      expectedRevision: number
      targetClusterId: string | null
    },
  ]
  clearError: []
}>()

const targetClusterId = ref<string | null>(null)
const labelEdits = ref<Record<string, string>>({})
const manager = ref<HTMLElement | null>(null)

const selectedSpan = computed(() => {
  const span = props.spans.find(item => item.id === props.selectedSpanId)
  return span?.isFinal ? span : null
})

const visibleClusters = computed(() => {
  const sessionId = selectedSpan.value?.sessionId
  return props.clusters.filter(cluster => cluster.sessionId === sessionId && cluster.mergedIntoClusterId === null)
})

const canReassign = computed(
  () => Boolean(selectedSpan.value) && targetClusterId.value !== selectedSpan.value?.speakerClusterId && !props.pending
)

watch(
  () => [selectedSpan.value?.id, selectedSpan.value?.revision, selectedSpan.value?.speakerClusterId],
  () => {
    targetClusterId.value = selectedSpan.value?.speakerClusterId ?? null
  },
  { immediate: true }
)

watch(
  () => props.clusters,
  clusters => {
    labelEdits.value = Object.fromEntries(clusters.map(cluster => [cluster.id, cluster.label]))
  },
  { deep: true, immediate: true }
)

function timestamp(ns: number): string {
  const totalSeconds = Math.floor(ns / 1_000_000_000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function captureTimestamp(span: TranscriptSpan): string {
  if (props.useWallClock && span.wallClockStart) {
    const date = new Date(span.wallClockStart)
    if (!Number.isNaN(date.valueOf())) {
      return new Intl.DateTimeFormat('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      }).format(date)
    }
  }

  return timestamp(Math.max(0, span.captureStartNs - props.sessionStartNs))
}

onMounted(() => {
  manager.value?.focus()
})

function requestCreate() {
  if (!selectedSpan.value || props.pending) return
  emit('clearError')
  emit('create', selectedSpan.value.sessionId)
}

function submitRename(cluster: SpeakerCluster) {
  const label = labelEdits.value[cluster.id]?.trim() ?? ''
  if (!label || label === cluster.label || props.pending) return

  emit('clearError')
  emit('rename', {
    sessionId: cluster.sessionId,
    clusterId: cluster.id,
    expectedLabelRevision: cluster.labelRevision,
    label,
  })
}

function submitReassignment() {
  if (!selectedSpan.value || !canReassign.value) return

  emit('clearError')
  emit('reassign', {
    sessionId: selectedSpan.value.sessionId,
    logicalSpanId: selectedSpan.value.id,
    expectedRevision: selectedSpan.value.revision,
    targetClusterId: targetClusterId.value,
  })
}
</script>

<template>
  <aside
    ref="manager"
    class="speaker-manager"
    aria-labelledby="speaker-manager-title"
    tabindex="-1"
    @keydown.esc="emit('close')"
  >
    <header class="speaker-manager__header">
      <div>
        <p class="speaker-manager__eyebrow">SPEAKER NOTES</p>
        <h2 id="speaker-manager-title">说话人归类</h2>
      </div>
      <button
        class="icon-button speaker-manager__close"
        type="button"
        aria-label="关闭说话人归类"
        title="关闭"
        @click="emit('close')"
      >
        <span class="i-mdi-close" aria-hidden="true" />
      </button>
    </header>

    <template v-if="selectedSpan">
      <section class="speaker-manager__selection" aria-label="当前记录片段">
        <p class="speaker-manager__selection-time">{{ captureTimestamp(selectedSpan) }}</p>
        <p>{{ selectedSpan.text }}</p>
      </section>

      <form class="speaker-manager__assignment" @submit.prevent="submitReassignment">
        <label class="speaker-manager__field" for="speaker-target">
          <span>归类到</span>
          <select id="speaker-target" v-model="targetClusterId" data-testid="speaker-target">
            <option :value="null">未归类</option>
            <option v-for="cluster in visibleClusters" :key="cluster.id" :value="cluster.id">
              {{ cluster.label }} · {{ cluster.spanCount }} 条记录
            </option>
          </select>
        </label>
        <button
          class="speaker-manager__apply"
          type="submit"
          data-testid="apply-speaker-reassignment"
          :disabled="!canReassign"
          :aria-busy="pending"
        >
          <span class="i-mdi-check" aria-hidden="true" />
          <span>{{ pending ? '保存中' : '应用归类' }}</span>
        </button>
      </form>

      <section class="speaker-manager__catalog" aria-label="说话人目录">
        <div class="speaker-manager__catalog-heading">
          <div>
            <p class="speaker-manager__eyebrow">LOCAL CATALOG</p>
            <h3>说话人名称</h3>
          </div>
          <button
            class="icon-button speaker-manager__create"
            type="button"
            data-testid="create-speaker-cluster"
            aria-label="新增说话人归类"
            title="新增说话人归类"
            :disabled="pending"
            :aria-busy="pending"
            @click="requestCreate"
          >
            <span class="i-mdi-plus" aria-hidden="true" />
          </button>
        </div>

        <ul v-if="visibleClusters.length" class="speaker-manager__list">
          <li v-for="cluster in visibleClusters" :key="cluster.id" class="speaker-manager__row">
            <form @submit.prevent="submitRename(cluster)">
              <label :for="`speaker-label-${cluster.id}`" class="sr-only">{{ cluster.label }} 的名称</label>
              <input
                :id="`speaker-label-${cluster.id}`"
                v-model="labelEdits[cluster.id]"
                :data-testid="`speaker-label-${cluster.id}`"
                type="text"
                maxlength="80"
                autocomplete="off"
                :disabled="pending"
              />
              <span>{{ cluster.spanCount }} 条</span>
              <button
                class="icon-button speaker-manager__rename"
                type="submit"
                :data-testid="`rename-speaker-${cluster.id}`"
                :disabled="
                  pending || !labelEdits[cluster.id]?.trim() || labelEdits[cluster.id]?.trim() === cluster.label
                "
                :aria-label="`保存${cluster.label}的新名称`"
                title="保存名称"
              >
                <span class="i-mdi-check" aria-hidden="true" />
              </button>
            </form>
          </li>
        </ul>
        <p v-else class="speaker-manager__empty">尚无说话人归类</p>
      </section>
    </template>
    <p v-else class="speaker-manager__empty">未选择记录片段</p>

    <p v-if="error" class="speaker-manager__error" role="alert">{{ error }}</p>
  </aside>
</template>
