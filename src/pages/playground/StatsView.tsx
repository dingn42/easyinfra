import { Stat } from '@/components/ui'
import { useT } from '@/lib/i18n'
import type { SimStats } from '../../lib/cudasim/types'

interface Diag {
  tone: 'volt' | 'cyan' | 'amber' | 'rose'
  text: string
}

type Tr = <T>(en: T, zh: T) => T

function diagnose(s: SimStats, eff: number | null, t: Tr): Diag[] {
  const out: Diag[] = []
  if (eff != null && eff < 55) {
    const factor = Math.max(2, Math.round(100 / Math.max(1, eff)))
    out.push({
      tone: 'rose',
      text: t(
        `Coalescing efficiency is only ${eff.toFixed(0)}%: adjacent threads in a warp aren't hitting adjacent addresses, so the transaction count runs well over the ideal. Rewrite the index in the contiguous form idx = blockIdx.x * blockDim.x + threadIdx.x to cut transactions by roughly 1/${factor}.`,
        `合并效率仅 ${eff.toFixed(0)}%：同 warp 相邻线程没有访问相邻地址，事务数远超理想值。把下标改成 idx = blockIdx.x * blockDim.x + threadIdx.x 这种连续形式，事务数能降到 1/${factor} 左右。`,
      ),
    })
  } else if (eff != null && eff < 90) {
    out.push({
      tone: 'amber',
      text: t(
        `Coalescing efficiency is ${eff.toFixed(0)}%: some accesses straddle a 32B segment. Check the index for a stride, a misalignment, or an offset that doesn't divide evenly.`,
        `合并效率 ${eff.toFixed(0)}%：部分访问跨 32B 段。检查索引里是否带 stride、错位或除不尽的偏移。`,
      ),
    })
  }
  if (s.bankConflicts > 0) {
    out.push({
      tone: 'amber',
      text: t(
        `Detected ${s.bankConflicts} bank conflict(s): several threads in a warp address different locations in the same bank on the same cycle, forcing the hardware to serialize. The usual fixes: pad the shared array (e.g. widen 32 to 33), or adjust the index to avoid colliding on index % 32.`,
        `检测到 ${s.bankConflicts} 次 bank conflict：同 warp 多个线程同拍读写同一 bank 的不同地址，硬件只能串行。常用解法：给 shared 数组做 padding（如把宽 32 改成 33），或调整索引避开 index % 32 撞车。`,
      ),
    })
  }
  if (s.divergentBranches > s.warps) {
    out.push({
      tone: 'rose',
      text: t(
        `Branch divergence ${s.divergentBranches} times (across ${s.warps} warps): threads in a warp take different paths, so both sides run serially. Try to align branch conditions to warp boundaries (e.g. condition on threadIdx.x / 32 rather than threadIdx.x % 2).`,
        `分支分化 ${s.divergentBranches} 次（warp 数 ${s.warps}）：同 warp 线程走了不同路径，两边都要串行跑一遍。尽量让分支条件按 warp 对齐（例如用 threadIdx.x / 32 而不是 threadIdx.x % 2 做条件）。`,
      ),
    })
  } else if (s.divergentBranches > 0) {
    out.push({
      tone: 'cyan',
      text: t(
        `${s.divergentBranches} minor branch divergence(s) — usually just the last warp doing a bounds check, nothing to worry about.`,
        `有 ${s.divergentBranches} 次轻微分支分化 —— 通常来自边界检查的最后一个 warp，无伤大雅。`,
      ),
    })
  }
  if (s.sharedWrites > 0 && s.sharedReads > 0 && s.syncBarriers === 0) {
    out.push({
      tone: 'rose',
      text: t(
        'Shared memory is read after a write with no __syncthreads() in between: the result depends on thread-scheduling order and can break under a different schedule. Add a barrier between the write and the read.',
        '写入 shared 后没有任何 __syncthreads() 就读取：结果依赖线程推进时序，换个调度顺序就可能出错。在写与读之间补一道屏障。',
      ),
    })
  }
  if (out.length === 0) {
    out.push({
      tone: 'volt',
      text: t(
        'Perfectly coalesced access, no bank conflicts, branches in lockstep — a textbook kernel, ready for the whiteboard interview.',
        '访存完美合并、没有 bank conflict、分支整齐 —— 教科书级的 kernel，可以拿去面试现场了。',
      ),
    })
  }
  return out
}

const DIAG_COLOR: Record<Diag['tone'], string> = {
  volt: 'text-volt',
  cyan: 'text-cyan',
  amber: 'text-amber',
  rose: 'text-rose',
}

export function StatsView({ stats }: { stats: SimStats | null }) {
  const t = useT()
  if (!stats) {
    return (
      <div className="flex flex-col items-center gap-2 py-14 text-center">
        <span className="microlabel">NO STATS</span>
        <p className="text-[13px] text-ink3">{t('No stats yet — RUN once first.', '还没有统计数据，先 RUN 一次。')}</p>
      </div>
    )
  }
  const gAccess = stats.globalReads + stats.globalWrites
  const ideal = gAccess / 8 // 完美合并：每 8 个 4B 元素 = 1 个 32B 段
  const eff = stats.globalTransactions > 0 ? Math.min(100, (ideal / stats.globalTransactions) * 100) : null
  const diags = diagnose(stats, eff, t)

  return (
    <div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-5 sm:grid-cols-3">
        <Stat label="THREADS" value={stats.threads.toLocaleString('en-US')} />
        <Stat label="WARPS" value={stats.warps} />
        <Stat label="MAX STEPS" value={stats.maxSteps} />
        <Stat label="GLOBAL READ / WRITE" value={`${stats.globalReads.toLocaleString('en-US')} / ${stats.globalWrites.toLocaleString('en-US')}`} size="sm" tone="cyan" />
        <Stat label="GLOBAL TXN" value={stats.globalTransactions.toLocaleString('en-US')} tone="volt" />
        <Stat
          label={t('COALESCING EFF', '合并效率')}
          value={eff == null ? '—' : `${eff.toFixed(0)}%`}
          tone={eff == null ? 'ink' : eff >= 90 ? 'volt' : eff >= 55 ? 'amber' : 'rose'}
        />
        <Stat label="SHARED R / W" value={`${stats.sharedReads.toLocaleString('en-US')} / ${stats.sharedWrites.toLocaleString('en-US')}`} size="sm" tone="amber" />
        <Stat label="BANK CONFLICTS" value={stats.bankConflicts} tone={stats.bankConflicts > 0 ? 'rose' : 'ink'} />
        <Stat label="DIVERGENT BRANCHES" value={stats.divergentBranches} tone={stats.divergentBranches > stats.warps ? 'rose' : 'ink'} />
        <Stat label="BARRIERS" value={stats.syncBarriers} tone="violet" />
      </div>

      <div className="mt-6 space-y-2 border-t border-line pt-4">
        <div className="microlabel">AUTO DIAGNOSIS</div>
        {diags.map((d, i) => (
          <p key={i} className="flex gap-2 text-[13px] leading-relaxed text-ink2">
            <span className={`shrink-0 font-mono ${DIAG_COLOR[d.tone]}`}>{d.tone === 'volt' ? '✓' : '▸'}</span>
            <span>{d.text}</span>
          </p>
        ))}
      </div>
    </div>
  )
}
