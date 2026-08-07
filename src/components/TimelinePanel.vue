<script setup lang="ts">
import type { TranscriptSpan } from '@/types'

const props = defineProps<{
  spans: TranscriptSpan[]
}>()

function timestamp(ns: number): string {
  const totalSeconds = Math.floor(ns / 1_000_000_000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function speakerLabel(speakerClusterId: string | null): string {
  if (!speakerClusterId) return '未归类'
  const number = speakerClusterId.replace('speaker-', '')
  return `说话人 ${number}`
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
        <time :datetime="String(span.captureStartNs)">{{ timestamp(span.captureStartNs) }}</time>
        <div class="timeline-entry__rail" aria-hidden="true"><span /></div>
        <article class="timeline-entry__body">
          <div class="timeline-entry__meta">
            <span class="speaker-tag">{{ speakerLabel(span.speakerClusterId) }}</span>
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
