<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import FlatSelect from '@/components/FlatSelect.vue'
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

const activeAsrOptions = computed(() =>
  selectableAsrModels.value.map(model => ({
    value: model.id,
    label: asrOptionLabel(model),
  }))
)

const bundledAsrModel = computed(() => {
  const bundledModelId = props.bundledAsrStatus?.modelId
  if (!props.bundledAsrStatus?.available || !bundledModelId) return null
  return selectableAsrModels.value.find(model => model.id === bundledModelId) ?? null
})

const hasStaleActiveAsrProfile = computed(() => Boolean(props.activeAsrProfile && !activeAsrModel.value))

const bundledAsrIssue = computed(() => {
  const status = props.bundledAsrStatus
  if (!status) return null
  if (!status.available) return safeBundledAsrMessage(status.message)
  if (!status.modelId || !bundledAsrModel.value) return bundledAsrUnavailableMessage
  return null
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
  if (isBundledAsrModel(model)) return '应用内置'
  return isWhisperCppAsrModel(model) ? '本机导入' : null
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

function selectActiveAsrModel(modelId: string) {
  if (!modelId || activeAsrSelectorDisabled.value) return
  emit('clearActiveAsrError')
  emit('selectActiveAsrModel', modelId)
}

function selectModelKind(modelKind: string) {
  form.modelKind = modelKind as LocalModelKind
}
</script>

<template>
  <section class="model-registry" aria-label="本地模型">
    <div class="model-registry__toolbar">
      <label class="model-registry__asr-selector">
        <span>当前转写模型</span>
        <FlatSelect
          :model-value="activeAsrModel?.id ?? ''"
          :options="activeAsrOptions"
          placeholder="选择兼容模型"
          :disabled="activeAsrSelectorDisabled"
          label="当前转写模型"
          test-id="active-asr-model-select"
          @update:model-value="selectActiveAsrModel"
        />
      </label>
      <button
        class="model-registry__import-action"
        type="button"
        title="导入本地模型"
        data-testid="open-model-import"
        @click="openForm"
      >
        <span class="i-mdi-plus" aria-hidden="true" />
        <span>导入模型</span>
      </button>
    </div>
    <p v-if="bundledAsrIssue" class="model-registry__bundled-error" data-testid="bundled-asr-status" role="alert">
      {{ bundledAsrIssue }}
    </p>
    <p v-if="!selectableAsrModels.length" class="model-registry__asr-note">
      {{ emptyAsrModelNote }}
    </p>
    <p v-else-if="activeAsrSelectionDisabled" class="model-registry__asr-note">记录中不可更换本地转写模型</p>
    <p v-else-if="selectingActiveAsrModel" class="model-registry__asr-note" role="status">正在切换模型</p>
    <p v-else-if="hasStaleActiveAsrProfile" class="model-registry__asr-error" role="alert">
      当前选择不可用，请重新选择模型
    </p>
    <p v-if="activeAsrError" class="model-registry__asr-error" role="alert">{{ activeAsrError }}</p>

    <div class="model-registry__list-heading">
      <h4>已安装模型</h4>
      <span>{{ models.length }} 个</span>
    </div>
    <ul v-if="models.length" class="model-registry__list" aria-label="本地模型">
      <li
        v-for="model in listedModels"
        :key="model.id"
        class="model-row"
        :class="{ 'model-row--active': activeAsrModel?.id === model.id }"
      >
        <span class="model-row__icon i-mdi-chip" aria-hidden="true" />
        <div class="model-row__identity">
          <strong>{{ model.version }}</strong>
          <span>
            {{ kindLabel(model.modelKind)
            }}<template v-if="modelOriginLabel(model)"> · {{ modelOriginLabel(model) }}</template>
          </span>
        </div>
        <span class="model-row__size">{{ formatSize(model.fileSizeBytes) }}</span>
        <span v-if="activeAsrModel?.id === model.id" class="model-row__active">使用中</span>
        <details class="model-row__details">
          <summary>
            <span class="i-mdi-information-outline" aria-hidden="true" />
            <span>详情</span>
          </summary>
          <dl>
            <div>
              <dt>输入格式</dt>
              <dd>
                <code>{{ model.inputFormat }}</code>
              </dd>
            </div>
            <div>
              <dt>SHA-256</dt>
              <dd>
                <code>{{ model.sha256 }}</code>
              </dd>
            </div>
            <div>
              <dt>模型卡</dt>
              <dd>{{ model.modelCardId }}</dd>
            </div>
            <div>
              <dt>许可证</dt>
              <dd>{{ model.licenseId }}</dd>
            </div>
          </dl>
        </details>
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
        <FlatSelect
          :model-value="form.modelKind"
          :options="modelKinds"
          label="模型类型"
          test-id="model-kind"
          @update:model-value="selectModelKind"
        />
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
