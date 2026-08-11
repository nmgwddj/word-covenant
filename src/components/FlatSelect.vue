<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

export interface FlatSelectOption {
  value: string
  label: string
  disabled?: boolean
}

const props = withDefaults(
  defineProps<{
    modelValue: string
    options: FlatSelectOption[]
    placeholder?: string
    label: string
    disabled?: boolean
    testId?: string
  }>(),
  {
    placeholder: '请选择',
    disabled: false,
    testId: undefined,
  }
)

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const root = ref<HTMLElement | null>(null)
const trigger = ref<HTMLButtonElement | null>(null)
const open = ref(false)
const activeIndex = ref(-1)
const listboxId = `flat-select-${Math.random().toString(36).slice(2)}`

const selectedIndex = computed(() => props.options.findIndex(option => option.value === props.modelValue))
const selectedOption = computed(() => props.options[selectedIndex.value] ?? null)
const enabledIndexes = computed(() =>
  props.options.reduce<number[]>((indexes, option, index) => {
    if (!option.disabled) indexes.push(index)
    return indexes
  }, [])
)
const activeOptionId = computed(() =>
  open.value && activeIndex.value >= 0 ? `${listboxId}-option-${activeIndex.value}` : undefined
)

watch(
  () => props.disabled,
  disabled => {
    if (disabled) close()
  }
)

onMounted(() => {
  document.addEventListener('pointerdown', closeFromOutside)
})

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', closeFromOutside)
})

function closeFromOutside(event: PointerEvent) {
  if (!root.value?.contains(event.target as Node)) close()
}

function openMenu(preferredIndex = selectedIndex.value) {
  if (props.disabled || !enabledIndexes.value.length) return
  open.value = true
  activeIndex.value = enabledIndexes.value.includes(preferredIndex) ? preferredIndex : enabledIndexes.value[0]!
}

function close() {
  open.value = false
  activeIndex.value = -1
}

function toggle() {
  if (open.value) close()
  else openMenu()
}

function moveActive(step: 1 | -1) {
  if (!open.value) {
    openMenu(step === 1 ? enabledIndexes.value[0] : enabledIndexes.value.at(-1))
    return
  }
  const current = enabledIndexes.value.indexOf(activeIndex.value)
  const next = current < 0 ? 0 : (current + step + enabledIndexes.value.length) % enabledIndexes.value.length
  activeIndex.value = enabledIndexes.value[next]!
}

function choose(index: number) {
  const option = props.options[index]
  if (!option || option.disabled || props.disabled) return
  if (option.value !== props.modelValue) emit('update:modelValue', option.value)
  close()
  trigger.value?.focus()
}

function onKeydown(event: KeyboardEvent) {
  if (props.disabled) return
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    moveActive(event.key === 'ArrowDown' ? 1 : -1)
    return
  }
  if (event.key === 'Home' || event.key === 'End') {
    event.preventDefault()
    openMenu(event.key === 'Home' ? enabledIndexes.value[0] : enabledIndexes.value.at(-1))
    return
  }
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    if (open.value && activeIndex.value >= 0) choose(activeIndex.value)
    else openMenu()
    return
  }
  if (event.key === 'Escape' && open.value) {
    event.preventDefault()
    close()
  }
}
</script>

<template>
  <div ref="root" class="flat-select" :class="{ 'is-open': open, 'is-disabled': disabled }">
    <button
      ref="trigger"
      class="flat-select__trigger"
      type="button"
      role="combobox"
      aria-haspopup="listbox"
      :aria-label="label"
      :aria-expanded="open"
      :aria-controls="listboxId"
      :aria-activedescendant="activeOptionId"
      :disabled="disabled"
      :data-testid="testId"
      @click="toggle"
      @keydown="onKeydown"
    >
      <span :class="{ 'is-placeholder': !selectedOption }">{{ selectedOption?.label ?? placeholder }}</span>
      <span class="flat-select__chevron i-mdi-chevron-down" aria-hidden="true" />
    </button>

    <Transition name="flat-select-menu">
      <ul v-if="open" :id="listboxId" class="flat-select__menu" role="listbox" :aria-label="label">
        <li
          v-for="(option, index) in options"
          :id="`${listboxId}-option-${index}`"
          :key="option.value"
          class="flat-select__option"
          :class="{ 'is-active': activeIndex === index, 'is-selected': modelValue === option.value }"
          role="option"
          :aria-selected="modelValue === option.value"
          :aria-disabled="option.disabled || undefined"
          :data-value="option.value"
          @mouseenter="!option.disabled && (activeIndex = index)"
          @click="choose(index)"
        >
          <span>{{ option.label }}</span>
          <span v-if="modelValue === option.value" class="i-mdi-check" aria-hidden="true" />
        </li>
      </ul>
    </Transition>
  </div>
</template>
