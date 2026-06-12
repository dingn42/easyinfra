import { useState } from 'react'
import { Slider, Stat, Widget } from '@/components/ui'
import { pct } from '@/lib/format'

/** LAB 03 占用率计算器：A100 单个 SM 的四种资源约束分别允许多少个 block 驻留 */

const SM = {
  regs: 65536, // 每 SM 寄存器数
  sharedKB: 164, // 每 SM shared memory（A100 可配置上限 164KB）
  threads: 2048, // 每 SM 最大驻留线程
  blocks: 32, // 每 SM 最大驻留 block
}

export function OccupancyLab() {
  const [regs, setRegs] = useState(64) // 每线程寄存器
  const [sharedKB, setSharedKB] = useState(8) // 每 block shared KB
  const [blockDim, setBlockDim] = useState(256) // 每 block 线程数

  const byRegs = Math.floor(SM.regs / (regs * blockDim))
  const byShared = sharedKB === 0 ? Infinity : Math.floor(SM.sharedKB / sharedKB)
  const byThreads = Math.floor(SM.threads / blockDim)
  const byBlocks = SM.blocks

  const blocksPerSM = Math.min(byRegs, byShared, byThreads, byBlocks)
  const cantLaunch = blocksPerSM === 0
  const occupancy = (blocksPerSM * blockDim) / SM.threads
  const warps = (blocksPerSM * blockDim) / 32

  const rows: { label: string; value: number; detail: string }[] = [
    { label: '寄存器', value: byRegs, detail: `${SM.regs.toLocaleString('en-US')} ÷ (${regs} × ${blockDim})` },
    { label: 'Shared Mem', value: byShared, detail: sharedKB === 0 ? '不占用 shared' : `${SM.sharedKB} KB ÷ ${sharedKB} KB` },
    { label: '线程上限', value: byThreads, detail: `${SM.threads} ÷ ${blockDim}` },
    { label: 'Block 上限', value: byBlocks, detail: '硬件固定 32' },
  ]

  return (
    <Widget
      index={3}
      title="占用率计算器"
      subtitle="A100 单个 SM：四种资源，谁先卡住 block 数"
      onReset={() => {
        setRegs(64)
        setSharedKB(8)
        setBlockDim(256)
      }}
      footer={
        <>
          occupancy 不是越高越好 —— 它只需要「够隐藏延迟」。高性能 GEMM 常常故意把 occupancy
          压到 25%~50%，换来每线程 100+ 个寄存器去放 micro-tile 的累加器：每个 warp 干的活多了，
          需要的 warp 自然就少了。
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <Slider label="每线程寄存器" value={regs} min={16} max={128} step={8} onChange={setRegs} />
        <Slider label="每 block shared" value={sharedKB} min={0} max={48} step={2} onChange={setSharedKB} unit="KB" />
        <Slider label="blockDim（线程/block）" value={blockDim} min={64} max={1024} step={32} onChange={setBlockDim} />
      </div>

      <div className="mt-5 grid grid-cols-3 gap-4">
        <Stat
          label="OCCUPANCY"
          value={cantLaunch ? '—' : pct(occupancy)}
          tone={cantLaunch ? 'rose' : occupancy >= 0.5 ? 'volt' : 'amber'}
          size="lg"
        />
        <Stat label="驻留 BLOCK/SM" value={cantLaunch ? 0 : blocksPerSM} tone="cyan" />
        <Stat label="活跃 WARP/SM" value={cantLaunch ? 0 : `${warps}/64`} tone="cyan" />
      </div>

      {cantLaunch && (
        <div className="mt-3 rounded-md border border-rose/40 bg-rose/10 px-3.5 py-2 font-mono text-[12px] text-rose">
          LAUNCH FAILURE：单个 block 需要 {regs} × {blockDim} ={' '}
          {(regs * blockDim).toLocaleString('en-US')} 个寄存器，超过 SM 总量 65,536 —— kernel 根本启动不了。
        </div>
      )}

      <div className="mt-5">
        <div className="microlabel mb-2">各约束允许的驻留 block 数（rose = 当前瓶颈）</div>
        <div className="space-y-1.5">
          {rows.map((r) => {
            const isBottleneck = !cantLaunch && r.value === blocksPerSM
            const w = (Math.min(r.value, SM.blocks) / SM.blocks) * 100
            return (
              <div key={r.label} className="flex items-center gap-2">
                <span className={`w-24 shrink-0 font-mono text-[11px] ${isBottleneck ? 'text-rose' : 'text-ink3'}`}>
                  {r.label}
                </span>
                <span className="relative h-5 flex-1 overflow-hidden rounded-sm bg-bg2">
                  <span
                    className={`absolute inset-y-0 left-0 rounded-sm transition-all duration-300 ${
                      r.value === 0 ? 'bg-rose/80' : isBottleneck ? 'bg-rose/60' : 'bg-cyan/30'
                    }`}
                    style={{ width: `${Math.max(w, r.value === 0 ? 2 : 4)}%` }}
                  />
                </span>
                <span
                  className={`w-16 shrink-0 text-right font-mono text-[11px] tabular-nums ${
                    isBottleneck ? 'text-rose' : 'text-ink2'
                  }`}
                >
                  {r.value === Infinity ? '不限' : `${r.value} blk`}
                </span>
                <span className="hidden w-44 shrink-0 truncate text-right font-mono text-[10px] text-ink3 sm:block">
                  {r.detail}
                </span>
              </div>
            )
          })}
        </div>
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink3">
        occupancy = 驻留线程 ÷ 2048。实际硬件按 256 个一组的粒度分配寄存器、warp 为单位调度，这里取理想化公式 ——
        但「四个约束取最小值」这个结构是精确的，CUDA Occupancy Calculator 用的就是同一套算法。
      </p>
    </Widget>
  )
}
