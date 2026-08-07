<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  egressEnabled: boolean
  profileApprovals: number
  disabled?: boolean
}>()

const emit = defineEmits<{
  setEgressEnabled: [enabled: boolean]
}>()

const confirmationOpen = ref(false)

function openConfirmation() {
  confirmationOpen.value = true
}

function cancelConfirmation() {
  confirmationOpen.value = false
}

function confirmEnable() {
  confirmationOpen.value = false
  emit('setEgressEnabled', true)
}
</script>

<template>
  <section class="outbound-access" aria-label="会话出网访问">
    <div class="outbound-access__heading">
      <span
        class="outbound-access__icon"
        :class="egressEnabled ? 'i-mdi-lan-connect' : 'i-mdi-lan-disconnect'"
        aria-hidden="true"
      />
      <div>
        <p class="outbound-access__eyebrow">SESSION NETWORK</p>
        <h3>会话出网访问</h3>
      </div>
      <span class="outbound-access__state" :class="{ 'outbound-access__state--enabled': egressEnabled }">
        {{ egressEnabled ? '已开启' : '已关闭' }}
      </span>
    </div>

    <p class="outbound-access__copy">
      {{ egressEnabled ? '仅允许此会话申请受控出网。' : '此会话当前不会发起任何外部请求。' }}
    </p>
    <p class="outbound-access__profile" :class="{ 'outbound-access__profile--approved': profileApprovals > 0 }">
      <span class="i-mdi-shield-key-outline" aria-hidden="true" />
      <span v-if="profileApprovals > 0">已批准 {{ profileApprovals }} 个命名配置</span>
      <span v-else>尚未批准命名配置，外部请求仍会被拦截</span>
    </p>

    <button
      v-if="!egressEnabled"
      class="outbound-access__button"
      data-testid="enable-egress"
      type="button"
      :disabled="disabled"
      @click="openConfirmation"
    >
      <span class="i-mdi-lan-connect" aria-hidden="true" />
      <span>启用出网访问</span>
    </button>
    <button
      v-else
      class="outbound-access__button outbound-access__button--disable"
      data-testid="disable-egress"
      type="button"
      :disabled="disabled"
      @click="emit('setEgressEnabled', false)"
    >
      <span class="i-mdi-lan-disconnect" aria-hidden="true" />
      <span>立即关闭出网</span>
    </button>

    <div
      v-if="confirmationOpen"
      class="outbound-access__confirmation"
      role="dialog"
      aria-labelledby="outbound-confirmation-title"
    >
      <span class="outbound-access__confirmation-icon i-mdi-alert-outline" aria-hidden="true" />
      <div>
        <strong id="outbound-confirmation-title">确认开启本会话出网访问？</strong>
        <p>开启后仍需逐个批准命名配置，未批准的外部请求会继续被拦截。</p>
        <div class="outbound-access__confirmation-actions">
          <button
            class="outbound-access__text-button"
            data-testid="cancel-enable-egress"
            type="button"
            @click="cancelConfirmation"
          >
            取消
          </button>
          <button
            class="outbound-access__confirm-button"
            data-testid="confirm-enable-egress"
            type="button"
            :disabled="disabled"
            @click="confirmEnable"
          >
            <span class="i-mdi-check" aria-hidden="true" />
            <span>确认开启</span>
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
