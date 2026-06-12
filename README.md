# EasyInfra

**交互式学习 GPU、CUDA 与大模型推理系统。** 在浏览器里拆开 GPU、写 CUDA、搭推理系统 ——
12 章内容、30+ 个可交互实验、一个能真正执行代码的 CUDA 模拟器，零环境配置。

## 运行

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # 类型检查 + 生产构建（输出 dist/，纯静态可任意托管）
```

## 课程结构

| 部分 | 章节 |
|---|---|
| **I · GPU 与并行计算** | 01 为什么需要 GPU · 02 GPU 硬件解剖 |
| **II · CUDA 编程** | 03 CUDA 编程模型 · 04 访存为王 · 05 实战矩阵乘法 · 06 Roofline 与性能分析 |
| **III · 大模型推理系统** | 07 Transformer 计算解剖 · 08 FlashAttention · 09 KV Cache 与 PagedAttention · 10 推理服务系统 · 11 量化 · 12 分布式并行 |

另有 **CUDA Playground**：内置 CUDA C 子集解释器（纯 TypeScript，见
`src/lib/cudasim/`），可编辑、运行 kernel，观察每一次内存访问、合并事务数、
bank conflict 与分支分化，附 6 个示例与 3 个修 bug 挑战。

## 技术栈与代码地图

Vite + React 18 + TypeScript (strict) + Tailwind CSS v4 · KaTeX · CodeMirror 6

```
src/
├── lib/
│   ├── cudasim/        # CUDA C 子集解释器（lexer/parser/interp，语言规范见 LANGUAGE.md）
│   ├── chapters.ts     # 课程目录元数据（唯一注册表）
│   └── hooks.ts …      # rAF 循环、视口测量等共享 hooks
├── components/
│   ├── ui/             # 设计系统组件：Section/Widget/Quiz/CodeBlock/MathTex/控件族
│   └── layout/         # 侧边栏与外壳
├── chapters/01-… 12-…  # 每章一个目录：正文 + 专属交互实验组件
└── pages/              # Home / ChapterPage / Playground
```

设计语言与贡献规范见 [DESIGN.md](DESIGN.md)。
