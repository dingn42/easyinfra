import { useMemo, useState } from 'react'
import { Slider, Stat, Widget } from '@/components/ui'
import { useT } from '@/lib/i18n'

/**
 * LAB 03 Bank Conflict 检查器：lane i 访问 s[(i*stride) % 1024]，看 32 个 bank 怎么排队。
 * 用 1024 个 word（32 bank × 32 行）保证 stride=32 时 32 个不同地址全砸进 bank 0 —— 32 路全冲突。
 */

const LANES = 32
const BANKS = 32
const WORDS = 1024
const ROWS = WORDS / BANKS // 32

const W = 720
const COL_W = W / BANKS
const LANE_Y = 24
const GRID_Y = 60
const CELL_H = 7
const GRID_H = ROWS * CELL_H
const VH = GRID_Y + GRID_H + 26

const DEFAULT_STRIDE = 2

export function BankConflictLab() {
  const t = useT()
  const [stride, setStride] = useState(DEFAULT_STRIDE)

  const { words, bankWords, ways, conflicted, hasBroadcast } = useMemo(() => {
    const ws = Array.from({ length: LANES }, (_, i) => (i * stride) % WORDS)
    const bw = new Map<number, Set<number>>()
    ws.forEach((w) => {
      const b = w % BANKS
      if (!bw.has(b)) bw.set(b, new Set())
      bw.get(b)!.add(w)
    })
    let mx = 1
    const conf = new Set<number>()
    bw.forEach((s, b) => {
      mx = Math.max(mx, s.size)
      if (s.size > 1) conf.add(b)
    })
    return {
      words: ws,
      bankWords: bw,
      ways: mx,
      conflicted: conf,
      hasBroadcast: new Set(ws).size < ws.length,
    }
  }, [stride])

  const laneX = (i: number) => (i + 0.5) * COL_W
  const cellCX = (w: number) => (w % BANKS) * COL_W + COL_W / 2
  const cellCY = (w: number) => GRID_Y + Math.floor(w / BANKS) * CELL_H + CELL_H / 2
  const tone = ways === 1 ? 'volt' : ways === 2 ? 'amber' : 'rose'

  return (
    <Widget
      index={3}
      title={t('Bank conflict inspector', 'Bank Conflict 检查器')}
      subtitle={t('lane i accesses s[(i x stride) mod 1024]', 'lane i 访问 s[(i × stride) mod 1024]')}
      wide
      onReset={() => setStride(DEFAULT_STRIDE)}
      footer={
        <>
          {t(
            <>
              Sweep the slider: stride=1 is perfectly parallel; 2 -&gt; 2-way conflict; 32 -&gt; the whole warp
              jams into bank 0, 32x serialization; 33 -&gt; perfect again. That mysterious +1 column in
              tile[32][33] is, in essence, turning the column-access stride from 32 into 33. Also: several lanes
              reading the <em>same</em> address is not a conflict — the hardware broadcasts.
            </>,
            <>
              拖一遍滑杆：stride=1 完美并行；2 → 2 路冲突；32 → 全 warp 挤进 bank 0，32 倍串行；33 →
              又完美。tile[32][33] 那个神秘的 +1 列，本质就是把列访问的 stride 从 32 变成 33。另：多个
              lane 读<em>同一个</em>地址不算冲突，硬件会广播（broadcast）。
            </>,
          )}
        </>
      }
    >
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Slider
          className="w-64 max-w-full"
          label="STRIDE"
          value={stride}
          min={1}
          max={33}
          onChange={setStride}
          unit="word"
        />
        <div className="font-mono text-sm">
          {ways === 1 ? (
            <span className="text-volt">
              {t('No conflict -> 32 accesses in 1 cycle', '无冲突 → 32 个访问 1 拍完成')}
              {hasBroadcast && t(' (incl. same-address broadcast)', '（含同址广播）')}
            </span>
          ) : (
            <span className="text-rose">
              {t(`${ways}-way conflict -> ${ways}x serialization`, `${ways} 路冲突 → ${ways} 倍串行化`)}
              {hasBroadcast && t(' (some lanes broadcast same address)', '（部分 lane 同址广播）')}
            </span>
          )}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${VH}`} className="mt-4 w-full font-mono">
        <text x={0} y={10} fontSize={10} fill="currentColor" className="text-ink3">
          {t(
            'WARP 32 LANES -> __shared__ float s[1024] (grid below: 32 BANK columns x 32 word rows)',
            'WARP 32 LANE → __shared__ float s[1024]（下方网格：32 个 BANK 列 × 32 行 word）',
          )}
        </text>

        {/* 冲突 bank 列的 rose 底色 */}
        <g>
          {Array.from(conflicted).map((b) => (
            <rect
              key={b}
              x={b * COL_W}
              y={GRID_Y - 3}
              width={COL_W}
              height={GRID_H + 6}
              fill="var(--color-rose)"
              fillOpacity={0.09}
              stroke="var(--color-rose)"
              strokeOpacity={0.4}
            />
          ))}
        </g>

        {/* lane → bank 单元连线 */}
        <g>
          {words.map((w, i) => {
            const bad = conflicted.has(w % BANKS)
            return (
              <line
                key={i}
                x1={laneX(i)}
                y1={LANE_Y + 7}
                x2={cellCX(w)}
                y2={cellCY(w)}
                stroke={bad ? 'var(--color-rose)' : 'var(--color-cyan)'}
                strokeOpacity={bad ? 0.55 : 0.4}
                strokeWidth={1}
              />
            )
          })}
        </g>

        {/* 256 word 网格 */}
        <g>
          {Array.from({ length: WORDS }, (_, w) => (
            <rect
              key={w}
              x={(w % BANKS) * COL_W + 1}
              y={GRID_Y + Math.floor(w / BANKS) * CELL_H + 0.5}
              width={COL_W - 2}
              height={CELL_H - 1}
              fill="none"
              stroke="var(--color-line)"
              strokeOpacity={0.5}
            />
          ))}
        </g>

        {/* 被访问的 word */}
        <g>
          {words.map((w, i) => {
            const bad = conflicted.has(w % BANKS)
            return (
              <rect
                key={i}
                x={(w % BANKS) * COL_W + 2}
                y={GRID_Y + Math.floor(w / BANKS) * CELL_H + 1}
                width={COL_W - 4}
                height={CELL_H - 2}
                rx={1.5}
                fill={bad ? 'var(--color-rose)' : 'var(--color-volt)'}
                fillOpacity={0.85}
              />
            )
          })}
        </g>

        {/* lane 圆点 */}
        <g>
          {Array.from({ length: LANES }, (_, i) => (
            <circle key={i} cx={laneX(i)} cy={LANE_Y} r={4.5} fill="var(--color-volt)" fillOpacity={0.85} />
          ))}
        </g>

        {/* bank 编号 */}
        <g className="text-ink3" fontSize={9} fill="currentColor">
          {Array.from({ length: BANKS }, (_, b) =>
            b % 4 === 0 || b === 31 ? (
              <text key={b} x={b * COL_W + COL_W / 2} y={GRID_Y + GRID_H + 14} textAnchor="middle">
                {b}
              </text>
            ) : null,
          )}
        </g>
      </svg>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <Stat
          label={t('Max conflict ways', '最大冲突路数')}
          value={t(`${ways}-way`, `${ways} 路`)}
          tone={tone}
          size="lg"
        />
        <Stat label={t('Serialization cost', '串行化代价')} value={`${ways}×`} tone={tone} size="lg" />
        <Stat label={t('Banks touched', '触及 BANK')} value={`${bankWords.size}/32`} unit="" tone="cyan" />
      </div>
    </Widget>
  )
}
