<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { isWhisperCppAsrModel, WHISPER_CPP_GGML_INPUT_FORMAT } from '@/stores/models'
import type {
  ActiveLocalAsrProfile,
  BundledAsrStatus,
  LocalModelImportInput,
  LocalModelKind,
  RegisteredModel,
} from '@/types'

const props = withDefaults(
  defineProps<{
    models: RegisteredModel[]
    compatibleAsrModels?: RegisteredModel[]
    bundledAsrStatus?: BundledAsrStatus | null
    activeAsrProfile?: ActiveLocalAsrProfile | null
    importing?: boolean
    selectingActiveAsrModel?: boolean
    activeAsrSelectionDisabled?: boolean
    error?: string | null
    activeAsrError?: string | null
    selectSourcePath?: () => Promise<string | null>
  }>(),
  {
    bundledAsrStatus: null,
    activeAsrProfile: null,
    importing: false,
    selectingActiveAsrModel: false,
    activeAsrSelectionDisabled: false,
    error: null,
    activeAsrError: null,
    selectSourcePath: undefined,
  }
)

const emit = defineEmits<{
  import: [input: LocalModelImportInput]
  selectActiveAsrModel: [modelId: string]
  clearError: []
  clearActiveAsrError: []
}>()

const formOpen = ref(false)
const selectingSourcePath = ref(false)
const modelCountAtSubmit = ref<number | null>(null)
const form = reactive<LocalModelImportInput>({
  sourcePath: '',
  modelKind: 'speech_recognition',
  version: '',
  inputFormat: WHISPER_CPP_GGML_INPUT_FORMAT,
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

const bundledAsrUnavailableMessage = '内置离线转写模型不可用，请重新安装应用'

const canImport = computed(
  () =>
    form.licenseAcknowledged &&
    form.sourcePath.trim().length > 0 &&
    form.version.trim().length > 0 &&
    form.inputFormat.trim().length > 0 &&
    form.expectedSha256.trim().length === 64 &&
    form.modelCardId.trim().length > 0 &&
    form.licenseId.trim().length > 0
)

const selectableAsrModels = computed(() =>
  (props.compatibleAsrModels ?? props.models).filter(
    model =>
      isWhisperCppAsrModel(model) &&
      (model.id !== props.bundledAsrStatus?.modelId || props.bundledAsrStatus?.available === true)
  )
)

const activeAsrModel = computed(() => {
  const modelId = props.activeAsrProfile?.modelId
  if (!modelId) return null
  return selectableAsrModels.value.find(model => model.id === modelId) ?? null
})

const bundledAsrModel = computed(() => {
  const bundledModelId = props.bundledAsrStatus?.modelId
  if (!props.bundledAsrStatus?.available || !bundledModelId) return null
  return selectableAsrModels.value.find(model => model.id === bundledModelId) ?? null
})

const displayedAsrModel = computed(() => activeAsrModel.value ?? bundledAsrModel.value)

const hasStaleActiveAsrProfile = computed(() => Boolean(props.activeAsrProfile && !activeAsrModel.value))

const bundledAsrIssue = computed(() => {
  const status = props.bundledAsrStatus
  if (!status) return null
  if (!status.available) return safeBundledAsrMessage(status.message)
  if (!status.modelId || !bundledAsrModel.value) return bundledAsrUnavailableMessage
  return null
})

const asrProfileState = computed(() => {
  if (activeAsrModel.value) {
    return isBundledAsrModel(activeAsrModel.value) ? '内置默认 · 已启用' : '高级本地模型 · 已启用'
  }
  if (bundledAsrModel.value) return '内置默认 · 就绪'
  if (bundledAsrIssue.value) return '内置模型不可用'
  return hasStaleActiveAsrProfile.value ? '当前选择不可用' : '未选择'
})

const emptyAsrModelNote = computed(() =>
  bundledAsrIssue.value
    ? `可导入已在本机准备的 ${WHISPER_CPP_GGML_INPUT_FORMAT} 语音识别模型作为高级本地模型`
    : `导入 ${WHISPER_CPP_GGML_INPUT_FORMAT} 语音识别模型后可开始本地转写`
)

const activeAsrSelectorDisabled = computed(
  () => props.activeAsrSelectionDisabled || props.selectingActiveAsrModel || !selectableAsrModels.value.length
)

const listedModels = computed(() =>
  [...props.models].sort((left, right) => {
    const leftBundled = isBundledAsrModel(left)
    const rightBundled = isBundledAsrModel(right)
    if (leftBundled === rightBundled) return 0
    return leftBundled ? -1 : 1
  })
)

watch(
  () => props.models.length,
  count => {
    if (modelCountAtSubmit.value !== null && count > modelCountAtSubmit.value && !props.error) {
      closeForm()
    }
  }
)

function kindLabel(kind: LocalModelKind): string {
  return modelKinds.find(option => option.value === kind)?.label ?? kind
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function isBundledAsrModel(model: RegisteredModel): boolean {
  return props.bundledAsrStatus?.modelId === model.id
}

function asrOptionLabel(model: RegisteredModel): string {
  if (isBundledAsrModel(model)) return `内置默认 · ${model.version}`
  return `${model.version} · ${model.sha256.slice(0, 8)}`
}

function modelOriginLabel(model: RegisteredModel): string | null {
  if (isBundledAsrModel(model)) return '内置默认'
  return isWhisperCppAsrModel(model) ? '高级本地模型' : null
}

function safeBundledAsrMessage(_message: string | null): string {
  return bundledAsrUnavailableMessage
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
    inputFormat: WHISPER_CPP_GGML_INPUT_FORMAT,
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

function selectActiveAsrModel(event: Event) {
  const modelId = (event.target as HTMLSelectElement).value
  if (!modelId || activeAsrSelectorDisabled.value) return
  emit('clearActiveAsrError')
  emit('selectActiveAsrModel', modelId)
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

    <div
      class="model-registry__asr-profile"
      :class="{ 'model-registry__asr-profile--unavailable': bundledAsrIssue }"
      data-testid="active-asr-profile"
    >
      <div class="model-registry__asr-profile-heading">
        <span class="i-mdi-waveform" aria-hidden="true" />
        <span>本地转写模型</span>
      </div>
      <strong v-if="displayedAsrModel">{{ displayedAsrModel.version }}</strong>
      <span v-else>{{ asrProfileState }}</span>
      <span v-if="displayedAsrModel" class="model-registry__asr-profile-state" data-testid="active-asr-source">
        {{ asrProfileState }}
      </span>
      <code v-if="displayedAsrModel">{{ displayedAsrModel.inputFormat }}</code>
    </div>
    <p v-if="bundledAsrIssue" class="model-registry__bundled-error" data-testid="bundled-asr-status" role="alert">
      {{ bundledAsrIssue }}
    </p>
    <label class="model-registry__asr-selector">
      <span>本次使用的模型</span>
      <select
        data-testid="active-asr-model-select"
        :value="activeAsrModel?.id ?? ''"
        :disabled="activeAsrSelectorDisabled"
        aria-label="本地转写模型"
        @change="selectActiveAsrModel"
      >
        <option value="" disabled>选择兼容模型</option>
        <option v-for="model in selectableAsrModels" :key="model.id" :value="model.id">
          {{ asrOptionLabel(model) }}
        </option>
      </select>
    </label>
    <p v-if="!selectableAsrModels.length" class="model-registry__asr-note">
      {{ emptyAsrModelNote }}
    </p>
    <p v-else-if="activeAsrSelectionDisabled" class="model-registry__asr-note">记录中不可更换本地转写模型</p>
    <p v-if="activeAsrError" class="model-registry__asr-error" role="alert">{{ activeAsrError }}</p>

    <ul v-if="models.length" class="model-registry__list" aria-label="本地模型">
      <li v-for="model in listedModels" :key="model.id" class="model-row">
        <span class="model-row__icon i-mdi-chip" aria-hidden="true" />
        <div class="model-row__body">
          <strong>{{ isBundledAsrModel(model) ? '内置默认 · 语音识别' : kindLabel(model.modelKind) }}</strong>
          <span>{{ model.version }} · {{ formatSize(model.fileSizeBytes) }}</span>
          <code>{{ model.sha256.slice(0, 12) }}</code>
          <span v-if="modelOriginLabel(model)" class="model-row__origin">{{ modelOriginLabel(model) }}</span>
          <span v-if="activeAsrModel?.id === model.id" class="model-row__active">当前转写</span>
        </div>
      </li>
    </ul>
    <p v-else class="model-registry__empty">暂无模型</p>

    <form v-if="formOpen" class="model-import" aria-label="导入本地模型" @submit.prevent="submitImport">
      <label class="model-import__field">
        <span>本地模型文件</span>
        <div class="model-import__source-row">
          <span
            class="model-import__source-state"
            data-testid="model-source-path"
            :data-selected="Boolean(form.sourcePath)"
          >
            {{ form.sourcePath ? '已选择本地文件' : '尚未选择文件' }}
          </span>
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
        <input v-model.trim="form.version" data-testid="model-version" type="text" autocomplete="off" required />
      </label>
      <label class="model-import__field">
        <span>输入格式</span>
        <input
          v-model.trim="form.inputFormat"
          data-testid="model-input-format"
          type="text"
          autocomplete="off"
          required
        />
      </label>
      <label class="model-import__field">
        <span>SHA-256</span>
        <input v-model.trim="form.expectedSha256" data-testid="model-sha256" type="text" autocomplete="off" required />
      </label>
      <label class="model-import__field">
        <span>模型卡</span>
        <input v-model.trim="form.modelCardId" data-testid="model-card-id" type="text" autocomplete="off" required />
      </label>
      <label class="model-import__field">
        <span>许可证</span>
        <input v-model.trim="form.licenseId" data-testid="model-license-id" type="text" autocomplete="off" required />
      </label>
      <label class="model-import__acknowledgement">
        <input v-model="form.licenseAcknowledged" data-testid="model-license-acknowledged" type="checkbox" />
        <span>已确认模型卡与许可证</span>
      </label>
      <p v-if="error" class="model-import__error" role="alert">{{ error }}</p>
      <div class="model-import__actions">
        <button class="model-import__cancel" type="button" data-testid="close-model-import" @click="closeForm">
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
