import { describe, expect, it } from 'vitest'

import { overageLabel, planUsageLabel } from '@/lib/statusbar'
import type { UnifiedUsage, UsageStats } from '@/types/hermes'

const BASE: UsageStats = { calls: 1, input: 10, output: 5, total: 15 }

function withUnified(overrides: Partial<UnifiedUsage> = {}): UsageStats {
  return {
    ...BASE,
    unified: {
      five_hour_percent: 27,
      seven_day_percent: 42,
      overage_percent: 0,
      on_overage: false,
      status: 'allowed',
      representative_claim: 'five_hour',
      five_hour_resets_in: 600,
      seven_day_resets_in: 6000,
      ...overrides
    }
  }
}

describe('planUsageLabel', () => {
  it('renders the 5h and 7d plan buckets', () => {
    expect(planUsageLabel(withUnified())).toBe('27% · 42%')
  })

  it('is empty without unified data so the caller hides the item', () => {
    // Non-Anthropic providers and API-key auth never send these headers.
    // Rendering "0% · 0%" there would read as "no usage" rather than "unknown".
    expect(planUsageLabel(BASE)).toBe('')
  })

  it('rounds fractional percentages', () => {
    expect(planUsageLabel(withUnified({ five_hour_percent: 27.8 }))).toBe('28% · 42%')
  })

  it('clamps out-of-range values into 0..100', () => {
    expect(planUsageLabel(withUnified({ five_hour_percent: -5, seven_day_percent: 150 }))).toBe('0% · 100%')
  })

  it('renders a fully consumed plan bucket', () => {
    expect(planUsageLabel(withUnified({ five_hour_percent: 100 }))).toBe('100% · 42%')
  })
})

describe('overageLabel', () => {
  it('is empty while the plan bucket is paying', () => {
    // The healthy case must stay visually quiet.
    expect(overageLabel(withUnified())).toBe('')
  })

  it('renders a badge once metered extra usage is consumed', () => {
    expect(overageLabel(withUnified({ on_overage: true, overage_percent: 13 }))).toBe('+13% extra')
  })

  it('is empty without unified data', () => {
    expect(overageLabel(BASE)).toBe('')
  })

  it('rounds and floors the overage percentage', () => {
    expect(overageLabel(withUnified({ on_overage: true, overage_percent: 12.6 }))).toBe('+13% extra')
    expect(overageLabel(withUnified({ on_overage: true, overage_percent: -1 }))).toBe('+0% extra')
  })

  it('honours the on_overage flag over a stale percentage', () => {
    // Backend owns the decision; the label must not second-guess it.
    expect(overageLabel(withUnified({ on_overage: false, overage_percent: 5 }))).toBe('')
  })
})
