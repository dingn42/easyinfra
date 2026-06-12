/** CH07 共享：Transformer 参数量 / FLOPs 计算器（纯函数，便于三个 LAB 复用） */

/** LLaMA-7B 的真实超参 */
export const LLAMA7B = {
  d: 4096,
  nHeads: 32,
  headDim: 128,
  dff: 11008,
  L: 32,
  V: 32000,
} as const

/** LLaMA-7B 官方参数量（6,738,415,616） */
export const LLAMA7B_PARAMS = 6_738_415_616

export interface ModelCfg {
  /** hidden size d_model */
  d: number
  /** 层数 */
  L: number
  /** FFN 中间维度 */
  dff: number
  /** 词表大小 */
  V: number
  /** n_kv / n_head（MHA=1，GQA<1） */
  kvRatio: number
  /** MLP 矩阵个数：SwiGLU=3，经典 GELU MLP=2 */
  mlpMats: number
}

/** 参数量分解（embedding 与 lm_head 不共享，各算一份 V·d） */
export function calcParams(c: ModelCfg) {
  const attn = c.L * 2 * c.d * c.d * (1 + c.kvRatio) // Q、O 各 d²；K、V 各 r·d²
  const mlp = c.L * c.mlpMats * c.d * c.dff
  const norm = c.L * 2 * c.d + c.d
  const embed = 2 * c.V * c.d
  return { attn, mlp, norm, embed, total: attn + mlp + norm + embed }
}

/** decode 单 token 前向 FLOPs 分解（上下文长度 S；乘加记 2 FLOPs） */
export function calcDecodeFlops(c: ModelCfg, S: number) {
  const qkv = c.L * 2 * c.d * c.d * (1 + 2 * c.kvRatio) // Q: 2d²，K/V: 各 2rd²
  const score = c.L * 4 * S * c.d // QKᵀ 2Sd + AV 2Sd（GQA 不减少这部分）
  const oproj = c.L * 2 * c.d * c.d
  const mlp = c.L * 2 * c.mlpMats * c.d * c.dff
  const head = 2 * c.V * c.d // lm_head logits
  return { qkv, score, oproj, mlp, head, total: qkv + score + oproj + mlp + head }
}

/** prefill 总 FLOPs（S 个 token、batch B，causal mask 下 attention 平均上下文 S/2） */
export function calcPrefillFlops(c: ModelCfg, S: number, B: number) {
  const per = calcDecodeFlops(c, 0)
  const linear = per.qkv + per.oproj + per.mlp // 线性部分与位置无关
  const attn = c.L * 4 * c.d * ((S * (S + 1)) / 2) // Σ_{i=1..S} 4·d·i
  return B * (linear * S + attn)
}
