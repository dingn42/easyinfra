/**
 * CUDA 模拟器自测：npx tsx src/lib/cudasim/selftest.ts
 * 覆盖正确性 / 越界 / 死循环 / 合并度 / bank conflict / 分支分化 / 屏障分化 / 编译错误。
 */
import { compile, run } from './index'
import type { CompiledKernel, SimOutcome } from './types'

let passCount = 0
let failCount = 0

function check(name: string, cond: boolean, detail?: string): void {
  if (cond) {
    passCount++
    console.log(`PASS  ${name}`)
  } else {
    failCount++
    console.log(`FAIL  ${name}${detail ? `  —— ${detail}` : ''}`)
  }
}

function mustCompile(src: string): CompiledKernel {
  const r = compile(src)
  if (!r.ok) throw new Error(`编译失败（第 ${r.error.line} 行）：${r.error.message}`)
  return r.kernel
}

function testWrap(name: string, fn: () => void): void {
  try {
    fn()
  } catch (e) {
    failCount++
    console.log(`FAIL  ${name}  —— 异常：${e instanceof Error ? e.message : String(e)}`)
  }
}

const d1 = (x: number) => ({ x, y: 1, z: 1 })

// ------------------------------------------------------------
// 1. vecAdd 正确性（N=1000, block=256, 含越界保护）
// ------------------------------------------------------------
testWrap('1. vecAdd 正确性', () => {
  const k = mustCompile(`
__global__ void vecAdd(const float* A, const float* B, float* C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}`)
  const N = 1000
  const runOnce = (): SimOutcome =>
    run(
      k,
      { grid: d1(4), block: d1(256) },
      [
        { name: 'A', length: N, init: 'iota' },
        { name: 'B', length: N, init: 'random' },
        { name: 'C', length: N, init: 'zero' },
      ],
      { scalarArgs: { n: N } },
    )
  const r = runOnce()
  if (!r.ok) {
    check('1. vecAdd 正确性', false, `运行失败：${r.error.message}`)
    return
  }
  let good = true
  for (let i = 0; i < N; i++) {
    if (Math.abs(r.buffers.C[i] - (r.buffers.A[i] + r.buffers.B[i])) > 1e-9) {
      good = false
      break
    }
  }
  check('1. vecAdd 正确性', good && r.stats.threads === 1024 && r.stats.warps === 32,
    `threads=${r.stats.threads} warps=${r.stats.warps}`)
  // 附带：random 固定种子可复现
  const r2 = runOnce()
  const same = r2.ok && r2.buffers.B.every((v, i) => v === r.buffers.B[i])
  check('1b. random 初始化可复现（seed=42）', same)
})

// ------------------------------------------------------------
// 2. shared memory 数组反转（两次 __syncthreads）
// ------------------------------------------------------------
testWrap('2. shared 数组反转', () => {
  const k = mustCompile(`
__global__ void reverse(float* A, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  s[t] = A[t];
  __syncthreads();
  float v = s[n - 1 - t];
  __syncthreads();
  A[t] = v;
}`)
  const N = 256
  const r = run(k, { grid: d1(1), block: d1(N) }, [{ name: 'A', length: N, init: 'iota' }], {
    scalarArgs: { n: N },
  })
  if (!r.ok) {
    check('2. shared 数组反转', false, `运行失败：${r.error.message}`)
    return
  }
  const good = r.buffers.A.every((v, i) => v === N - 1 - i)
  check('2. shared 数组反转', good && r.stats.syncBarriers === 2,
    `syncBarriers=${r.stats.syncBarriers}`)
})

// ------------------------------------------------------------
// 3. 越界 → ok:false 且错误信息含线程坐标
// ------------------------------------------------------------
testWrap('3. 越界访问报错', () => {
  const k = mustCompile(`
__global__ void oob(float* A, int n) {
  int i = threadIdx.x;
  A[i + n] = 1.0f;
}`)
  const r = run(k, { grid: d1(1), block: d1(32) }, [{ name: 'A', length: 32 }], { scalarArgs: { n: 32 } })
  const good =
    !r.ok &&
    r.error.message.includes('越界') &&
    r.error.message.includes('A[32]') &&
    r.error.message.includes('长度 32') &&
    r.error.message.includes('thread(0,0,0)') &&
    r.error.thread !== undefined &&
    r.error.line === 4
  check('3. 越界访问报错（含线程坐标与行号）', good, r.ok ? '未报错' : r.error.message)
})

// ------------------------------------------------------------
// 4. while(true) → 死循环防护
// ------------------------------------------------------------
testWrap('4. 死循环防护', () => {
  const k = mustCompile(`
__global__ void spin() {
  while (true) {
  }
}`)
  const r = run(k, { grid: d1(1), block: d1(1) }, [], { maxStepsPerThread: 2000 })
  const good = !r.ok && r.error.message.includes('死循环')
  check('4. while(true) 死循环防护', good, r.ok ? '未报错' : r.error.message)
})

// ------------------------------------------------------------
// 5. 合并 vs stride=2：globalTransactions 4 vs 8（N=32 单 warp）
// ------------------------------------------------------------
testWrap('5. 全局访问合并度', () => {
  const k = mustCompile(`
__global__ void gread(const float* A, int stride) {
  __shared__ float s[32];
  int i = threadIdx.x;
  s[i] = A[i * stride];
}`)
  const bufs = () => [{ name: 'A', length: 64, init: 'iota' as const }]
  const r1 = run(k, { grid: d1(1), block: d1(32) }, bufs(), { scalarArgs: { stride: 1 } })
  const r2 = run(k, { grid: d1(1), block: d1(32) }, bufs(), { scalarArgs: { stride: 2 } })
  if (!r1.ok || !r2.ok) {
    check('5. 全局访问合并度', false, '运行失败')
    return
  }
  check('5a. 合并访问 = 4 事务/warp', r1.stats.globalTransactions === 4,
    `globalTransactions=${r1.stats.globalTransactions}`)
  check('5b. stride=2 = 8 事务/warp', r2.stats.globalTransactions === 8,
    `globalTransactions=${r2.stats.globalTransactions}`)
})

// ------------------------------------------------------------
// 6. shared stride=2 → bankConflicts > 0；stride=1 → 0
// ------------------------------------------------------------
testWrap('6. shared bank conflict', () => {
  const k = mustCompile(`
__global__ void sread(float* out, int stride) {
  __shared__ float s[64];
  int i = threadIdx.x;
  s[i] = i;
  s[i + 32] = i;
  __syncthreads();
  out[i] = s[i * stride];
}`)
  const bufs = () => [{ name: 'out', length: 32 }]
  const r1 = run(k, { grid: d1(1), block: d1(32) }, bufs(), { scalarArgs: { stride: 1 } })
  const r2 = run(k, { grid: d1(1), block: d1(32) }, bufs(), { scalarArgs: { stride: 2 } })
  if (!r1.ok || !r2.ok) {
    check('6. shared bank conflict', false, '运行失败')
    return
  }
  check('6a. stride=1 无 bank conflict', r1.stats.bankConflicts === 0,
    `bankConflicts=${r1.stats.bankConflicts}`)
  check('6b. stride=2 有 bank conflict', r2.stats.bankConflicts > 0,
    `bankConflicts=${r2.stats.bankConflicts}`)
})

// ------------------------------------------------------------
// 7. warp 内 if(lane<16) → divergentBranches ≥ 1
// ------------------------------------------------------------
testWrap('7. 分支分化统计', () => {
  const k = mustCompile(`
__global__ void diverge(float* out) {
  int lane = threadIdx.x;
  if (lane < 16) {
    out[lane] = 1.0f;
  } else {
    out[lane] = 2.0f;
  }
}`)
  const r = run(k, { grid: d1(1), block: d1(32) }, [{ name: 'out', length: 32 }])
  const good = r.ok && r.stats.divergentBranches >= 1
  check('7. warp 内分支分化 ≥ 1', good, r.ok ? `divergentBranches=${r.stats.divergentBranches}` : r.error.message)
})

// ------------------------------------------------------------
// 8. barrier 分化（一半线程提前 return）→ 运行时错误
// ------------------------------------------------------------
testWrap('8. 屏障分化报错', () => {
  const k = mustCompile(`
__global__ void bardiv(float* out) {
  int i = threadIdx.x;
  if (i < 16) {
    return;
  }
  __syncthreads();
  out[i] = 1.0f;
}`)
  const r = run(k, { grid: d1(1), block: d1(32) }, [{ name: 'out', length: 32 }])
  const good = !r.ok && r.error.message.includes('屏障分化')
  check('8. __syncthreads 屏障分化报错', good, r.ok ? '未报错' : r.error.message)
})

// ------------------------------------------------------------
// 9. 编译错误：友好信息 + 行号
// ------------------------------------------------------------
testWrap('9. 编译错误', () => {
  // 9a. printf
  const r1 = compile(`
__global__ void f(float* A) {
  printf("hello");
}`)
  check('9a. printf → 暂不支持', !r1.ok && r1.error.message.includes('暂不支持') && r1.error.line === 3,
    r1.ok ? '未报错' : `${r1.error.message} @${r1.error.line}`)

  // 9b. 指针运算
  const r2 = compile(`
__global__ void f(float* A, float* B) {
  B[0] = A + 1;
}`)
  check('9b. 指针运算 → 暂不支持', !r2.ok && r2.error.message.includes('指针') && r2.error.line === 3,
    r2.ok ? '未报错' : `${r2.error.message} @${r2.error.line}`)

  // 9c. atomicAdd
  const r3 = compile(`
__global__ void f(int* A) {
  atomicAdd(A, 1);
}`)
  check('9c. atomicAdd → 暂不支持', !r3.ok && r3.error.message.includes('atomicAdd'),
    r3.ok ? '未报错' : r3.error.message)

  // 9d. 两个核函数
  const r4 = compile(`
__global__ void f(float* A) { A[0] = 1.0f; }
__global__ void g(float* A) { A[0] = 2.0f; }`)
  check('9d. 多个核函数 → 报错', !r4.ok && r4.error.message.includes('一个'),
    r4.ok ? '未报错' : r4.error.message)

  // 9e. 未声明变量带行号
  const r5 = compile(`
__global__ void f(float* A) {
  A[0] = x;
}`)
  check('9e. 未声明变量（带行号）', !r5.ok && r5.error.message.includes('未声明') && r5.error.line === 3,
    r5.ok ? '未报错' : `${r5.error.message} @${r5.error.line}`)
})

// ------------------------------------------------------------
// 10. 综合：for / while / 三元 / 数学函数 / int 语义
// ------------------------------------------------------------
testWrap('10. 综合语义', () => {
  const k = mustCompile(`
__global__ void misc(float* out, int* iv, int n) {
  int i = threadIdx.x;
  if (i >= n) {
    return;
  }
  float acc = 0.0f;
  for (int j = 0; j < 4; j++) {
    acc += sqrtf(powf(2.0f, j * 1.0f));
    if (j == 2) {
      continue;
    }
  }
  int q = 7 / 2;
  int m = -7 % 3;
  iv[i] = q * 100 + m + (i % 2 == 0 ? 1 : 0) + min(3, max(1, 2)) + abs(-2);
  out[i] = acc + fabsf(-1.5f) + floorf(2.9f) + expf(0.0f) + logf(1.0f) + sinf(0.0f) + cosf(0.0f) + warpSize;
}`)
  const r = run(k, { grid: d1(1), block: d1(8) }, [
    { name: 'out', length: 8 },
    { name: 'iv', length: 8 },
  ], { scalarArgs: { n: 8 } })
  if (!r.ok) {
    check('10. 综合语义', false, `运行失败：${r.error.message}`)
    return
  }
  // acc = sqrt(1) + sqrt(2) + sqrt(4) + sqrt(8)
  const acc = Math.sqrt(1) + Math.sqrt(2) + Math.sqrt(4) + Math.sqrt(8)
  const expFloat = acc + 1.5 + 2 + 1 + 0 + 0 + 1 + 32
  // q=3（截断），m=-1（C 截断取余），三元 1/0，min/max=2，abs=2
  const expInt = (even: boolean) => 300 + -1 + (even ? 1 : 0) + 2 + 2
  let good = true
  for (let i = 0; i < 8; i++) {
    if (Math.abs(r.buffers.out[i] - expFloat) > 1e-9) good = false
    if (r.buffers.iv[i] !== expInt(i % 2 === 0)) good = false
  }
  check('10. 综合语义（for/三元/数学/int 截断）', good,
    `out[0]=${r.buffers.out[0]} 期望 ${expFloat}；iv[0]=${r.buffers.iv[0]} 期望 ${expInt(true)}`)
})

// ------------------------------------------------------------
// 11. 访问轨迹完整性
// ------------------------------------------------------------
testWrap('11. 访问轨迹', () => {
  const k = mustCompile(`
__global__ void copy(const float* A, float* B) {
  int i = threadIdx.x;
  B[i] = A[i];
}`)
  const r = run(k, { grid: d1(1), block: d1(32) }, [
    { name: 'A', length: 32, init: 'iota' },
    { name: 'B', length: 32 },
  ])
  if (!r.ok) {
    check('11. 访问轨迹', false, r.error.message)
    return
  }
  const reads = r.accesses.filter((a) => a.kind === 'read' && a.buffer === 'A')
  const writes = r.accesses.filter((a) => a.kind === 'write' && a.buffer === 'B')
  const sameStep = new Set(reads.map((a) => a.step)).size === 1
  const good =
    reads.length === 32 &&
    writes.length === 32 &&
    sameStep &&
    reads.every((a) => a.warpId === 0 && a.laneId === a.index && a.line === 4) &&
    !r.traceTruncated
  check('11. 访问轨迹（step/warp/lane/line 正确）', good,
    `reads=${reads.length} writes=${writes.length} sameStep=${sameStep}`)
})

// ------------------------------------------------------------
console.log('—'.repeat(48))
console.log(`汇总：${passCount} 项 PASS，${failCount} 项 FAIL`)
console.log(failCount === 0 ? 'ALL PASS' : 'SOME FAILED')
