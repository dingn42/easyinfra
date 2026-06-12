# CUDA 模拟器 —— 支持的语言子集

引擎：`src/lib/cudasim/`（纯 TypeScript，零依赖）。
入口：`compile(source)` 与 `run(kernel, config, buffers, opts)`，契约见 `types.ts`。
本文档列出解释器接受的 CUDA C 子集；任何超出范围的写法都会在**编译期**给出带行号的
友好错误（`模拟器暂不支持 X`）。

## 顶层结构

- 源码必须且只能包含 **一个** 核函数定义：

  ```c
  __global__ void kernelName(参数列表) { ... }
  ```

- 参数类型仅限四种：`float*`、`int*`、`float`、`int`（允许 `const` 修饰，会被忽略）。
  - 指针参数按 **同名** 绑定到 `run()` 的 `buffers` 声明；缺失 → 运行时错误。
  - 标量参数通过 `opts.scalarArgs`（如 `{ n: 1024 }`）传入；缺失 → 运行时错误。
- 不允许 `#include` / `#define` 等预处理指令、多个函数、设备函数（`__device__`）。

## 共享内存

```c
__shared__ float s[256];   // 也支持 int；可声明多个
```

- 必须位于核函数体**顶层**；长度必须是整数常量，范围 `1..16384`；
- 每个 block 拥有独立副本，初始值全 0；
- 只能以 `s[i]` 下标形式读写。

## 语句

| 类别 | 形式 |
|---|---|
| 局部变量 | `int i = 0, j;` / `float x = 1e-5f;`（仅标量；不支持局部数组/局部指针） |
| 赋值 | `x = e;`、`x += e;`、`-=`、`*=`、`/=`（目标为变量或 `arr[idx]`） |
| 自增自减 | `i++; i--; ++i; --i;`（仅独立语句，仅普通变量） |
| 分支 | `if (c) { ... } else { ... }`（else if 链 OK，单语句可省大括号） |
| 循环 | `for (init; cond; update) { ... }`、`while (c) { ... }`、`break;`、`continue;` |
| 返回 | `return;`（核函数为 void，不能带值） |
| 同步 | `__syncthreads();`（仅独立语句） |

## 表达式

- 算术：`+ - * / %`（`%` 与整数除法均按 C 截断语义；`%` 不接受浮点操作数）
- 比较：`== != < <= > >=`（结果为 int 0/1）
- 逻辑：`&& || !`（短路求值）
- 三元：`c ? a : b`；括号；一元负号 / 一元加
- 字面量：`123`、`1.5f`、`1e-5f`、`.5`、`2.f`、`true`、`false`
- 下标读写：`A[i]`（指针参数与共享数组；一维；下标必须为整数表达式）

数值语义：float 按 JS double 计算（不模拟 fp32 舍入）；写入 int 变量 / `int*` 缓冲区 /
`int` 共享数组时截断取整；整数除以 0 → 运行时错误。

## 内建标识符与函数

- `threadIdx.x/y/z`、`blockIdx.x/y/z`、`blockDim.x/y/z`、`gridDim.x/y/z`、`warpSize`（= 32）
- `__syncthreads()`
- 数学：`min(a,b)`、`max(a,b)`、`abs(x)`、`fabsf(x)`、`sqrtf(x)`、`expf(x)`、`logf(x)`、
  `powf(x,y)`、`floorf(x)`、`sinf(x)`、`cosf(x)`

## 明确不支持（编译期报错）

指针运算 / 解引用（`*p`、`A + 1`、`&x`）、多级指针、struct/union/enum、`printf`、
原子操作（`atomicAdd` 等）、warp shuffle、`double` / `unsigned` / `long` 等类型、
位运算（`& | ^ ~ << >>`）、`switch`、`do-while`、十六进制字面量、字符串、
局部数组、多维下标、`%=`、设备端 `malloc`。

## 执行模型

- 线程线性化：`lane = tid.x + tid.y*bd.x + tid.z*bd.x*bd.y`；warp = 连续 32 个 lane。
- 上限：block ≤ 1024 线程，grid 总线程 ≤ 65536。
- 每个线程是一个 generator 协程：每条语句开头与每次循环回边产生一个 **逻辑步**，
  block 内所有存活线程按逻辑步同拍（lockstep）推进；每次内存访问 / 分支条件 /
  `__syncthreads` 都会产出事件。
- **globalTransactions**：同 warp 同 step 的全局访问按 32 字节段（每元素 4 字节，即
  8 元素一段）聚合，事务数 = 触及的不同段数（完美合并：32 个 float 读 = 4 事务）。
- **bankConflicts**：同 warp 同 step 的 shared 访问按 `bank = index % 32` 聚合，
  每组贡献 `max(同 bank 不同地址数) - 1`。
- **divergentBranches**：if/while/for/三元 条件在同 warp 同 step 取值不一致 → +1。
- `__syncthreads()`：block 内所有未返回线程都到达才放行；若有线程已 `return` 而其余
  线程在等待 → 运行时错误「屏障分化」。
- 越界访问 → 运行时错误（含缓冲区名、下标、长度、线程坐标、行号）；
  单线程逻辑步数超过 `maxStepsPerThread`（默认 100000）→ 判定疑似死循环。
- `'random'` 缓冲区初始化使用固定种子 LCG（seed=42），结果可复现。

## 自测

```bash
npx tsx src/lib/cudasim/selftest.ts
```
