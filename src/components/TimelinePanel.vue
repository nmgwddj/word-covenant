<script setup lang="ts">
import type { SpeakerCluster, TranscriptSpan } from '@/types'

const props = withDefaults(
  defineProps<{
    spans: TranscriptSpan[]
    speakerClusters?: SpeakerCluster[]
    sessionStartNs?: number
    useWallClock?: boolean
  }>(),
  {
    speakerClusters: () => [],
    sessionStartNs: 0,
    useWallClock: false,
  }
)

const emit = defineEmits<{
  openSpeakerManager: [spanId: string]
}>()

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

function speakerLabel(speakerClusterId: string | null): string {
  if (!speakerClusterId) return '未归类'
  const cluster = props.speakerClusters.find(item => item.id === speakerClusterId)
  const canonicalCluster = cluster ? props.speakerClusters.find(item => item.id === cluster.canonicalClusterId) : null
  const label = canonicalCluster?.label ?? cluster?.label
  if (label) return label
  return '未归类'
}
</script>

<template>
  <section class="timeline-panel" aria-label="记录时间线">
    <div class="section-heading">
      <div>
        <p class="section-heading__eyebrow">SESSION LOG</p>
        <h2>对话记录</h2>
      </div>
      <span class="section-heading__meta">{{ props.spans.length }} 条</span>
    </div>

    <ol v-if="props.spans.length" class="timeline-list">
      <li v-for="span in props.spans" :key="span.id" class="timeline-entry">
        <time :datetime="span.wallClockStart ?? String(span.captureStartNs)">{{ captureTimestamp(span) }}</time>
        <div class="timeline-entry__rail" aria-hidden="true"><span /></div>
        <article class="timeline-entry__body">
          <div class="timeline-entry__meta">
            <button
              v-if="span.isFinal"
              class="speaker-tag speaker-tag--interactive"
              type="button"
              data-testid="open-speaker-manager"
              :aria-label="`管理${speakerLabel(span.speakerClusterId)}`"
              @click="emit('openSpeakerManager', span.id)"
            >
              {{ speakerLabel(span.speakerClusterId) }}
            </button>
            <span v-else class="speaker-tag">{{ speakerLabel(span.speakerClusterId) }}</span>
            <span v-if="!span.isFinal" class="draft-tag">转写中</span>
            <span class="timeline-entry__duration">{{ timestamp(span.captureEndNs - span.captureStartNs) }}</span>
          </div>
          <p>{{ span.text }}</p>
        </article>
      </li>
    </ol>
    <div v-else class="timeline-empty">
      <span class="i-mdi-waveform" aria-hidden="true" />
      <p>尚无本地记录</p>
    </div>
  </section>
</template>
