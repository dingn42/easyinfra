import { useState } from 'react'
import { useT } from '@/lib/i18n'

/** 「支持的语法」可展开速查表 —— 从 src/lib/cudasim/LANGUAGE.md 精炼 */

export function SyntaxPanel() {
  const t = useT()
  const [open, setOpen] = useState(false)

  const ROWS: { cat: string; tone: string; items: { code: string; note: string }[] }[] = [
    {
      cat: t('TOP-LEVEL', '顶层'),
      tone: 'text-cyan',
      items: [
        {
          code: '__global__ void k(...) { }',
          note: t(
            'Exactly one kernel function; no #include / #define / device functions.',
            '有且仅有一个核函数；无 #include / #define / 设备函数',
          ),
        },
        {
          code: 'float* / int* / float / int',
          note: t(
            'Only these four parameter types; pointers bind to a buffer by matching name, scalars come from the config inputs.',
            '仅这四种参数类型；指针按同名绑定 buffer，标量来自配置区输入框',
          ),
        },
      ],
    },
    {
      cat: t('SHARED MEM', '共享内存'),
      tone: 'text-amber',
      items: [
        {
          code: '__shared__ float s[256];',
          note: t(
            'Declared at the top of the body; length is an integer constant in 1..16384; one copy per block, zero-initialized.',
            '函数体顶层声明；长度为 1..16384 的整数常量；每 block 一份，初值全 0',
          ),
        },
      ],
    },
    {
      cat: t('STATEMENTS', '语句'),
      tone: 'text-volt',
      items: [
        {
          code: 'int i = 0;  float x = 1.5f;',
          note: t(
            'Local scalars only; no local arrays or local pointers.',
            '仅局部标量；不支持局部数组 / 局部指针',
          ),
        },
        {
          code: 'x = e;  x += -= *= /=  i++ i--',
          note: t(
            'Assignment target is a variable or arr[idx]; ++/-- only as standalone statements.',
            '赋值目标为变量或 arr[idx]；自增减仅作独立语句',
          ),
        },
        {
          code: 'if / else · for · while · break · continue · return;',
          note: t(
            'Kernel is void, so return takes no value.',
            '核函数为 void，return 不带值',
          ),
        },
        {
          code: '__syncthreads();',
          note: t(
            'Block-level barrier; if some threads return early, the rest wait → flagged as "barrier divergence".',
            'block 级屏障；线程提前 return 后其余线程等待 → 报「屏障分化」',
          ),
        },
      ],
    },
    {
      cat: t('EXPRESSIONS', '表达式'),
      tone: 'text-ink',
      items: [
        {
          code: '+ - * / %  · == != < <= > >=  · && || !  · c ? a : b',
          note: t(
            'Integer / and % truncate as in C; % rejects floats; logical operators short-circuit.',
            '整数除法 / % 按 C 截断；% 不接受浮点；逻辑短路求值',
          ),
        },
        {
          code: 'A[i]',
          note: t(
            '1-D subscript read/write (pointer params and shared arrays); out of bounds → runtime error (with thread coordinate).',
            '一维下标读写（指针参数与 shared 数组）；越界 → 运行时错误（含线程坐标）',
          ),
        },
      ],
    },
    {
      cat: t('BUILT-INS', '内建'),
      tone: 'text-violet',
      items: [
        {
          code: 'threadIdx / blockIdx / blockDim / gridDim (.x .y .z) · warpSize',
          note: 'warpSize = 32',
        },
        {
          code: 'min max abs fabsf sqrtf expf logf powf floorf sinf cosf',
          note: t(
            'Floats are computed as JS doubles — no fp32 rounding is simulated.',
            'float 按 JS double 计算，不模拟 fp32 舍入',
          ),
        },
      ],
    },
    {
      cat: t('UNSUPPORTED', '不支持'),
      tone: 'text-rose',
      items: [
        {
          code: t(
            'pointer arithmetic · atomicAdd · bitwise ops · double/unsigned · switch · printf · multi-dim subscript',
            '指针运算 · atomicAdd · 位运算 · double/unsigned · switch · printf · 多维下标',
          ),
          note: t(
            'Outside the subset → a friendly compile-time error with a line number.',
            '超出子集 → 编译期给出带行号的友好错误',
          ),
        },
      ],
    },
  ]

  return (
    <div className="panel overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-panel2/60"
        aria-expanded={open}
      >
        <span className={`font-mono text-xs text-volt transition-transform duration-200 ${open ? 'rotate-90' : ''}`}>
          ▸
        </span>
        <span className="microlabel text-ink2">SUPPORTED SYNTAX</span>
        <span className="text-[13px] text-ink2">
          {t('Supported syntax — a CUDA C subset cheat sheet', '支持的语法 —— CUDA C 子集速查')}
        </span>
        <span className="ml-auto font-mono text-[11px] text-ink3">{open ? t('Collapse', '收起') : t('Expand', '展开')}</span>
      </button>
      {open && (
        <div className="border-t border-line px-4 py-4">
          <div className="space-y-4">
            {ROWS.map((r) => (
              <div key={r.cat} className="grid gap-x-4 gap-y-1.5 sm:grid-cols-[88px_1fr]">
                <div className={`microlabel pt-0.5 ${r.tone}`}>{r.cat}</div>
                <div className="space-y-1.5">
                  {r.items.map((it) => (
                    <div key={it.code} className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
                      <code className="rounded border border-line bg-bg2 px-1.5 py-0.5 font-mono text-[11.5px] text-cyan">
                        {it.code}
                      </code>
                      <span className="text-[12.5px] text-ink2">{it.note}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 border-t border-line pt-3 text-[12px] leading-relaxed text-ink3">
            {t(
              'Execution model: threads in a block advance in lockstep through logical steps; a warp is 32 consecutive threads. Same-warp, same-step global accesses coalesce into 32B-segment transactions (perfect coalescing = 4 transactions per warp); shared accesses are tallied for conflicts by bank = index % 32. A block holds ≤ 1024 threads, and the grid holds ≤ 65536 total.',
              '执行模型：block 内线程按逻辑步同拍（lockstep）推进，warp = 连续 32 线程；同 warp 同拍的全局访问按 32B 段聚合成事务（完美合并 = 每 warp 4 事务），shared 访问按 bank = index % 32 统计冲突。block ≤ 1024 线程，grid 总线程 ≤ 65536。',
            )}
          </p>
        </div>
      )}
    </div>
  )
}
