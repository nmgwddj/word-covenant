<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import type { LocalModelImportInput, LocalModelKind, RegisteredModel } from '@/types'

const props = withDefaults(defineProps<{
  models: RegisteredModel[]
  importing?: boolean
  error?: string | null
  selectSourcePath?: () => Promise<string | null>
}>(), {
  importing: false,
  error: null,
  selectSourcePath: undefined,
})

const emit = defineEmits<{
  import: [input: LocalModelImportInput]
  clearError: []
}>()

const formOpen = ref(false)
const selectingSourcePath = ref(false)
const modelCountAtSubmit = ref<number | null>(null)
const form = reactive<LocalModelImportInput>({
  sourcePath: '',
  modelKind: 'speech_recognition',
  version: '',
  inputFormat: 'gguf',
  expectedSha256: '',
  modelCardId: '',
  licenseId: '',
  licenseAcknowledged: false,
})

const modelKinds: Array<{ value: LocalModelKind; label: string }> = [
  { value: 'speech_recognition', label: '语音识别' },
  { value: 'voice_activity_detection', label: '语音活动检测' },
  { value: 'speaker_embedding', label: '说话人嵌入' },
]

const canImport = computed(() => (
  form.licenseAcknowledged
  && form.sourcePath.trim().length > 0
  && form.version.trim().length > 0
  && form.inputFormat.trim().length > 0
  && form.expectedSha256.trim().length === 64
  && form.modelCardId.trim().length > 0
  && form.licenseId.trim().length > 0
))

watch(() => props.models.length, (count) => {
  if (modelCountAtSubmit.value !== null && count > modelCountAtSubmit.value && !props.error) {
    closeForm()
  }
})

function kindLabel(kind: LocalModelKind): string {
  return modelKinds.find((option) => option.value === kind)?.label ?? kind
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function openForm() {
  emit('clearError')
  formOpen.value = true
}

function closeForm() {
  emit('clearError')
  formOpen.value = false
  modelCountAtSubmit.value = null
  Object.assign(form, {
    sourcePath: '',
    modelKind: 'speech_recognition',
    version: '',
    inputFormat: 'gguf',
    expectedSha256: '',
    modelCardId: '',
    licenseId: '',
    licenseAcknowledged: false,
  } satisfies LocalModelImportInput)
}

async function chooseSourcePath() {
  if (!props.selectSourcePath || selectingSourcePath.value) return

  emit('clearError')
  selectingSourcePath.value = true
  try {
    const selectedPath = await props.selectSourcePath()
    if (selectedPath) form.sourcePath = selectedPath
  } catch {
    // The store exposes a picker error through the same import error channel.
  } finally {
    selectingSourcePath.value = false
  }
}

function submitImport() {
  if (!canImport.value || props.importing) return
  modelCountAtSubmit.value = props.models.length
  emit('import', { ...form })
}
</script>

<template>
  <section class="model-registry" aria-label="本地模型">
    <div class="model-registry__heading">
      <div>
        <p class="session-rail__label">LOCAL MODELS</p>
        <h2>本地模型</h2>
      </div>
      <button
        class="icon-button model-registry__add"
        type="button"
        title="导入本地模型"
        data-testid="open-model-import"
        @click="openForm"
      >
        <span class="i-mdi-plus" aria-hidden="true" />
      </button>
    </div>

    <ul v-if="models.length" class="model-registry__list" aria-label="已导入模型">
      <li v-for="model in models" :key="model.id" class="model-row">
        <span class="model-row__icon i-mdi-chip" aria-hidden="true" />
        <div class="model-row__body">
          <strong>{{ kindLabel(model.modelKind) }}</strong>
          <span>{{ model.version }} · {{ formatSize(model.fileSizeBytes) }}</span>
          <code>{{ model.sha256.slice(0, 12) }}</code>
        </div>
      </li>
    </ul>
    <p v-else class="model-registry__empty">暂无模型</p>

    <form v-if="formOpen" class="model-import" aria-label="导入本地模型" @submit.prevent="submitImport">
      <label class="model-import__field">
        <span>文件路径</span>
        <div class="model-import__source-row">
          <input
            v-model="form.sourcePath"
            data-testid="model-source-path"
            type="text"
            autocomplete="off"
            readonly
            required
          >
          <button
            class="model-import__choose-source"
            type="button"
            data-testid="choose-model-source"
            :disabled="!selectSourcePath || selectingSourcePath"
            :aria-busy="selectingSourcePath"
            @click="chooseSourcePath"
          >
            <span class="i-mdi-folder-open-outline" aria-hidden="true" />
            <span>{{ selectingSourcePath ? '打开中' : '选择文件' }}</span>
          </button>
        </div>
      </label>
      <label class="model-import__field">
        <span>模型类型</span>
        <select v-model="form.modelKind" data-testid="model-kind">
          <option v-for="kind in modelKinds" :key="kind.value" :value="kind.value">{{ kind.label }}</option>
        </select>
      </label>
      <label class="model-import__field">
        <span>版本</span>
        <input v-model.trim="form.version" data-testid="model-version" type="text" autocomplete="off" required>
      </label>
      <label class="model-import__field">
        <span>输入格式</span>
        <input v-model.trim="form.inputFormat" type="text" autocomplete="off" required>
      </label>
      <label class="model-import__field">
        <span>SHA-256</span>
        <input v-model.trim="form.expectedSha256" data-testid="model-sha256" type="text" autocomplete="off" required>
      </label>
      <label class="model-import__field">
        <span>模型卡</span>
        <input v-model.trim="form.modelCardId" data-testid="model-card-id" type="text" autocomplete="off" required>
      </label>
      <label class="model-import__field">
        <span>许可证</span>
        <input v-model.trim="form.licenseId" data-testid="model-license-id" type="text" autocomplete="off" required>
      </label>
      <label class="model-import__acknowledgement">
        <input v-model="form.licenseAcknowledged" data-testid="model-license-acknowledged" type="checkbox">
        <span>已确认模型卡与许可证</span>
      </label>
      <p v-if="error" class="model-import__error" role="alert">{{ error }}</p>
      <div class="model-import__actions">
        <button
          class="model-import__cancel"
          type="button"
          data-testid="close-model-import"
          @click="closeForm"
        >
          取消
        </button>
        <button
          class="model-import__submit"
          data-testid="submit-model-import"
          type="submit"
          :disabled="!canImport || importing"
        >
          <span class="i-mdi-file-import-outline" aria-hidden="true" />
          <span>{{ importing ? '导入中' : '导入模型' }}</span>
        </button>
      </div>
    </form>
  </section>
</template>
