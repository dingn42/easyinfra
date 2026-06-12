/**
 * CUDA C 子集 —— 递归下降解析器 + 语义/类型检查
 *
 * parse(tokens)  → 带行号的 AST（KernelAst）
 * analyze(ast)   → 作用域/类型检查，并就地标注每个表达式的 isFloat（f 字段）、
 *                  下标访问的内存空间（global/shared）。超出子集 → CompileErrorException。
 */
import { CompileErrorException, type Token } from './lexer'

export type ScalarType = 'int' | 'float'

export interface ParamAst {
  name: string
  type: 'float*' | 'int*' | 'float' | 'int'
  isPointer: boolean
  elem: ScalarType
  line: number
  col: number
}

export interface SharedAst {
  name: string
  elem: ScalarType
  length: number
  line: number
  col: number
}

export interface KernelAst {
  name: string
  params: ParamAst[]
  shared: SharedAst[]
  body: Stmt[]
}

interface NodeBase {
  line: number
  col: number
}

// ---------- 表达式 ----------
export interface NumExpr extends NodeBase { k: 'num'; v: number; f: boolean }
export interface VarExpr extends NodeBase { k: 'var'; name: string; f: boolean }
export interface DimExpr extends NodeBase { k: 'dim'; base: string; field: 'x' | 'y' | 'z'; f: boolean }
export interface IndexExpr extends NodeBase { k: 'index'; name: string; idx: Expr; space: 'global' | 'shared'; f: boolean }
export interface UnExpr extends NodeBase { k: 'un'; op: '-' | '!'; e: Expr; f: boolean }
export interface BinExpr extends NodeBase {
  k: 'bin'
  op: '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '<=' | '>' | '>='
  l: Expr
  r: Expr
  f: boolean
}
export interface LogicExpr extends NodeBase { k: 'logic'; op: '&&' | '||'; l: Expr; r: Expr; f: boolean }
export interface CondExpr extends NodeBase { k: 'cond'; c: Expr; t: Expr; e: Expr; f: boolean; site: number }
export interface CallExpr extends NodeBase { k: 'call'; name: string; args: Expr[]; f: boolean }
export type Expr = NumExpr | VarExpr | DimExpr | IndexExpr | UnExpr | BinExpr | LogicExpr | CondExpr | CallExpr

// ---------- 语句 ----------
export interface DeclStmt extends NodeBase {
  k: 'decl'
  type: ScalarType
  decls: { name: string; init: Expr | null; line: number; col: number }[]
}
export interface AssignStmt extends NodeBase {
  k: 'assign'
  target: VarExpr | IndexExpr
  op: '=' | '+=' | '-=' | '*=' | '/='
  value: Expr
}
export interface IncDecStmt extends NodeBase { k: 'incdec'; name: string; op: '++' | '--' }
export interface IfStmt extends NodeBase { k: 'if'; c: Expr; t: Stmt[]; e: Stmt[] | null; site: number }
export interface WhileStmt extends NodeBase { k: 'while'; c: Expr; body: Stmt[]; site: number }
export interface ForStmt extends NodeBase {
  k: 'for'
  init: Stmt | null
  c: Expr | null
  update: Stmt | null
  body: Stmt[]
  site: number
}
export interface BlockStmt extends NodeBase { k: 'blk'; body: Stmt[] }
export interface SimpleStmt extends NodeBase { k: 'break' | 'continue' | 'return' | 'sync' }
export type Stmt = DeclStmt | AssignStmt | IncDecStmt | IfStmt | WhileStmt | ForStmt | BlockStmt | SimpleStmt

// ---------- 常量表 ----------
const KEYWORDS = new Set([
  'if', 'else', 'for', 'while', 'break', 'continue', 'return', 'void', 'int', 'float',
  'true', 'false', 'const', '__global__', '__shared__', 'do', 'switch', 'case', 'default',
  'struct', 'union', 'enum', 'double', 'unsigned', 'signed', 'long', 'short', 'char', 'bool',
  'static', 'goto', 'sizeof',
])

export const DIM_BUILTINS = new Set(['threadIdx', 'blockIdx', 'blockDim', 'gridDim'])
const BUILTIN_NAMES = new Set([...DIM_BUILTINS, 'warpSize', '__syncthreads'])

const MATH_FLOAT1 = new Set(['fabsf', 'sqrtf', 'expf', 'logf', 'floorf', 'sinf', 'cosf'])

/** 已知但不支持的函数 → 友好提示 */
const UNSUPPORTED_FUNCS: Record<string, string> = {
  printf: '模拟器暂不支持 printf',
  atomicAdd: '模拟器暂不支持 atomicAdd（原子操作）',
  atomicSub: '模拟器暂不支持 atomicSub（原子操作）',
  atomicMax: '模拟器暂不支持 atomicMax（原子操作）',
  atomicMin: '模拟器暂不支持 atomicMin（原子操作）',
  atomicExch: '模拟器暂不支持 atomicExch（原子操作）',
  atomicCAS: '模拟器暂不支持 atomicCAS（原子操作）',
  __syncwarp: '模拟器暂不支持 __syncwarp（请用 __syncthreads()）',
  __shfl_sync: '模拟器暂不支持 warp shuffle（__shfl_sync 系列）',
  __shfl_down_sync: '模拟器暂不支持 warp shuffle（__shfl_sync 系列）',
  __shfl_up_sync: '模拟器暂不支持 warp shuffle（__shfl_sync 系列）',
  __shfl_xor_sync: '模拟器暂不支持 warp shuffle（__shfl_sync 系列）',
  malloc: '模拟器暂不支持 malloc（设备端动态内存）',
  free: '模拟器暂不支持 free',
  memcpy: '模拟器暂不支持 memcpy',
  memset: '模拟器暂不支持 memset',
  assert: '模拟器暂不支持 assert',
  fmodf: '模拟器暂不支持 fmodf（浮点取模）',
  fminf: '模拟器暂不支持 fminf（请用 min）',
  fmaxf: '模拟器暂不支持 fmaxf（请用 max）',
  rsqrtf: '模拟器暂不支持 rsqrtf（请用 1.0f / sqrtf(x)）',
  exp2f: '模拟器暂不支持 exp2f（请用 expf / powf）',
  log2f: '模拟器暂不支持 log2f（请用 logf）',
  ceilf: '模拟器暂不支持 ceilf（请用 floorf 改写）',
  tanf: '模拟器暂不支持 tanf（请用 sinf/cosf）',
}

const BITWISE_OPS = new Set(['&', '|', '^', '<<', '>>'])

// ============================================================
// 解析器
// ============================================================
class Parser {
  private readonly toks: Token[]
  private p = 0
  private siteCounter = 0
  private readonly shared: SharedAst[] = []

  constructor(toks: Token[]) {
    this.toks = toks
  }

  private peek(): Token {
    return this.toks[this.p]
  }
  private next(): Token {
    return this.toks[this.p++]
  }
  private fail(msg: string, tok?: Token): never {
    const t = tok ?? this.peek()
    throw new CompileErrorException(msg, t.line, t.col)
  }
  private atPunct(text: string): boolean {
    const t = this.peek()
    return t.kind === 'punct' && t.text === text
  }
  private atIdent(text: string): boolean {
    const t = this.peek()
    return t.kind === 'ident' && t.text === text
  }
  private eatPunct(text: string): boolean {
    if (this.atPunct(text)) {
      this.p++
      return true
    }
    return false
  }
  private eatIdent(text: string): boolean {
    if (this.atIdent(text)) {
      this.p++
      return true
    }
    return false
  }
  private expectPunct(text: string, what?: string): Token {
    if (!this.atPunct(text)) this.fail(`此处需要 '${text}'${what ? `（${what}）` : ''}，得到 '${this.peek().text}'`)
    return this.next()
  }
  private expectName(what: string): Token {
    const t = this.peek()
    if (t.kind !== 'ident') this.fail(`此处需要${what}，得到 '${t.text}'`)
    if (KEYWORDS.has(t.text)) this.fail(`不能用关键字 '${t.text}' 作为${what}`)
    if (BUILTIN_NAMES.has(t.text)) this.fail(`不能用内建名 '${t.text}' 作为${what}`)
    return this.next()
  }

  // ---------- 顶层 ----------
  parseKernel(): KernelAst {
    const first = this.peek()
    if (first.kind === 'eof') this.fail('源码为空：请提供一个 __global__ void 核函数定义', first)
    if (!this.eatIdent('__global__')) {
      this.fail(`核函数必须以 __global__ void 开头（得到 '${first.text}'）`, first)
    }
    if (!this.eatIdent('void')) {
      this.fail(`__global__ 后必须是 void（核函数没有返回值）`)
    }
    const nameTok = this.expectName('核函数名')
    this.expectPunct('(', '参数列表')
    const params: ParamAst[] = []
    if (!this.atPunct(')')) {
      for (;;) {
        params.push(this.parseParam())
        if (this.eatPunct(',')) continue
        break
      }
    }
    this.expectPunct(')')
    this.expectPunct('{', '核函数体')
    const body = this.parseStmtList(true)
    this.expectPunct('}')
    const after = this.peek()
    if (after.kind !== 'eof') {
      if (after.text === '__global__') this.fail('只支持一个 __global__ 核函数定义', after)
      this.fail(`核函数定义之后存在多余代码：'${after.text}'`, after)
    }
    return { name: nameTok.text, params, shared: this.shared, body }
  }

  private parseParam(): ParamAst {
    this.eatIdent('const')
    const t = this.peek()
    if (t.kind !== 'ident' || (t.text !== 'int' && t.text !== 'float')) {
      if (t.kind === 'ident' && KEYWORDS.has(t.text)) {
        this.fail(`模拟器暂不支持 参数类型 '${t.text}'（仅支持 float* / int* / float / int）`, t)
      }
      this.fail(`非法的参数声明：需要类型 int 或 float，得到 '${t.text}'`, t)
    }
    this.next()
    const elem = t.text as ScalarType
    const isPointer = this.eatPunct('*')
    this.eatIdent('const')
    if (this.atPunct('*')) this.fail('模拟器暂不支持 多级指针（如 float**）')
    const nameTok = this.expectName('参数名')
    if (this.atPunct('[')) this.fail('模拟器暂不支持 数组形参（请改用指针，如 float* A）')
    return {
      name: nameTok.text,
      type: (isPointer ? `${elem}*` : elem) as ParamAst['type'],
      isPointer,
      elem,
      line: t.line,
      col: t.col,
    }
  }

  // ---------- 语句 ----------
  /** 解析到 '}'（不消费）为止 */
  private parseStmtList(topLevel: boolean): Stmt[] {
    const out: Stmt[] = []
    while (!this.atPunct('}')) {
      if (this.peek().kind === 'eof') this.fail("核函数体未闭合：缺少 '}'")
      const s = this.parseStmt(topLevel)
      if (s) out.push(s)
    }
    return out
  }

  /** 返回 null 表示 __shared__ 声明（已收进 this.shared，不进 body） */
  private parseStmt(topLevel: boolean): Stmt | null {
    const t = this.peek()
    if (t.kind === 'punct') {
      if (t.text === '{') {
        this.next()
        const body = this.parseStmtList(false)
        this.expectPunct('}')
        return { k: 'blk', body, line: t.line, col: t.col }
      }
      if (t.text === ';') {
        this.next()
        return { k: 'blk', body: [], line: t.line, col: t.col }
      }
      if (t.text === '*') this.fail('模拟器暂不支持 指针解引用 *p（指针只能以 p[i] 形式访问）', t)
      if (t.text === '++' || t.text === '--') {
        const s = this.parseSimpleStmt()
        this.expectPunct(';')
        return s
      }
      this.fail(`无效的语句开头 '${t.text}'`, t)
    }
    // t.kind === 'ident'
    switch (t.text) {
      case '__shared__': {
        if (!topLevel) this.fail('__shared__ 声明必须位于核函数体的顶层（不能在 if/for 等内部）', t)
        this.next()
        this.parseSharedDecl(t)
        return null
      }
      case 'const':
      case 'int':
      case 'float':
        return this.parseDecl()
      case 'if':
        return this.parseIf()
      case 'for':
        return this.parseFor()
      case 'while':
        return this.parseWhile()
      case 'break':
        this.next()
        this.expectPunct(';')
        return { k: 'break', line: t.line, col: t.col }
      case 'continue':
        this.next()
        this.expectPunct(';')
        return { k: 'continue', line: t.line, col: t.col }
      case 'return': {
        this.next()
        if (!this.atPunct(';')) this.fail('核函数返回类型为 void，return 不能携带返回值')
        this.next()
        return { k: 'return', line: t.line, col: t.col }
      }
      case '__syncthreads': {
        this.next()
        this.expectPunct('(')
        this.expectPunct(')')
        this.expectPunct(';')
        return { k: 'sync', line: t.line, col: t.col }
      }
      case 'else':
        this.fail("else 缺少对应的 if", t)
        break
      case 'do':
        this.fail('模拟器暂不支持 do-while 循环（请改用 while）', t)
        break
      case 'switch':
        this.fail('模拟器暂不支持 switch 语句（请改用 if/else）', t)
        break
      case 'struct':
      case 'union':
      case 'enum':
        this.fail(`模拟器暂不支持 ${t.text}`, t)
        break
      case 'double':
        this.fail('模拟器暂不支持 double 类型（请用 float）', t)
        break
      case 'unsigned':
      case 'signed':
      case 'long':
      case 'short':
      case 'char':
      case 'bool':
        this.fail(`模拟器暂不支持 ${t.text} 类型（仅支持 int / float）`, t)
        break
      default: {
        const s = this.parseSimpleStmt()
        this.expectPunct(';')
        return s
      }
    }
    /* istanbul ignore next */
    throw new Error('unreachable')
  }

  private parseSharedDecl(kw: Token): void {
    const t = this.peek()
    if (!this.eatIdent('float') && !this.eatIdent('int')) {
      this.fail(`__shared__ 后需要 float 或 int，得到 '${t.text}'`, t)
    }
    const elem = t.text as ScalarType
    const nameTok = this.expectName('共享数组名')
    this.expectPunct('[', '共享数组长度')
    const lenTok = this.peek()
    if (lenTok.kind !== 'num') this.fail('共享数组长度必须是整数常量（不支持表达式）', lenTok)
    if (lenTok.isFloat) this.fail('共享数组长度必须是整数常量（不能是浮点数）', lenTok)
    this.next()
    const length = lenTok.value ?? 0
    if (length < 1 || length > 16384) this.fail(`共享数组长度需在 1..16384 之间（得到 ${length}）`, lenTok)
    this.expectPunct(']')
    if (this.atPunct('=')) this.fail('模拟器暂不支持 共享数组初始化（声明后默认全 0）')
    this.expectPunct(';')
    if (this.shared.some((s) => s.name === nameTok.text)) {
      this.fail(`共享数组 '${nameTok.text}' 重复声明`, nameTok)
    }
    this.shared.push({ name: nameTok.text, elem, length, line: kw.line, col: kw.col })
  }

  private parseDecl(): DeclStmt {
    const start = this.peek()
    this.eatIdent('const')
    const t = this.peek()
    if (!this.eatIdent('int') && !this.eatIdent('float')) {
      this.fail(`局部变量声明需要类型 int 或 float，得到 '${t.text}'`, t)
    }
    const type = t.text as ScalarType
    if (this.atPunct('*')) this.fail('模拟器暂不支持 局部指针变量', this.peek())
    const decls: DeclStmt['decls'] = []
    for (;;) {
      const nameTok = this.expectName('变量名')
      if (this.atPunct('[')) this.fail('模拟器暂不支持 局部数组（请用 __shared__ 数组）')
      let init: Expr | null = null
      if (this.eatPunct('=')) init = this.parseExpr()
      decls.push({ name: nameTok.text, init, line: nameTok.line, col: nameTok.col })
      if (this.eatPunct(',')) continue
      break
    }
    this.expectPunct(';')
    return { k: 'decl', type, decls, line: start.line, col: start.col }
  }

  private parseIf(): IfStmt {
    const kw = this.next()
    this.expectPunct('(', 'if 条件')
    const c = this.parseExpr()
    this.expectPunct(')')
    const t = this.parseBlockOrSingle()
    let e: Stmt[] | null = null
    if (this.eatIdent('else')) e = this.parseBlockOrSingle()
    return { k: 'if', c, t, e, site: this.siteCounter++, line: kw.line, col: kw.col }
  }

  private parseWhile(): WhileStmt {
    const kw = this.next()
    this.expectPunct('(', 'while 条件')
    const c = this.parseExpr()
    this.expectPunct(')')
    const body = this.parseBlockOrSingle()
    return { k: 'while', c, body, site: this.siteCounter++, line: kw.line, col: kw.col }
  }

  private parseFor(): ForStmt {
    const kw = this.next()
    this.expectPunct('(', 'for 头部')
    let init: Stmt | null = null
    if (this.atIdent('int') || this.atIdent('float') || this.atIdent('const')) {
      init = this.parseDecl() // 含 ';'
    } else if (this.eatPunct(';')) {
      init = null
    } else {
      init = this.parseSimpleStmt()
      this.expectPunct(';')
    }
    let c: Expr | null = null
    if (!this.atPunct(';')) c = this.parseExpr()
    this.expectPunct(';')
    let update: Stmt | null = null
    if (!this.atPunct(')')) update = this.parseSimpleStmt()
    this.expectPunct(')')
    const body = this.parseBlockOrSingle()
    return { k: 'for', init, c, update, body, site: this.siteCounter++, line: kw.line, col: kw.col }
  }

  private parseBlockOrSingle(): Stmt[] {
    if (this.atPunct('{')) {
      this.next()
      const body = this.parseStmtList(false)
      this.expectPunct('}')
      return body
    }
    const s = this.parseStmt(false)
    return s ? [s] : []
  }

  /** 赋值 / 自增自减（不消费分号；供语句与 for 头部共用） */
  private parseSimpleStmt(): AssignStmt | IncDecStmt {
    const start = this.peek()
    // 前缀 ++i / --i
    if (this.atPunct('++') || this.atPunct('--')) {
      const op = this.next().text as '++' | '--'
      const nameTok = this.expectName('变量名')
      return { k: 'incdec', name: nameTok.text, op, line: start.line, col: start.col }
    }
    const target = this.parsePostfix()
    const t = this.peek()
    if (t.kind === 'punct') {
      if (t.text === '=' || t.text === '+=' || t.text === '-=' || t.text === '*=' || t.text === '/=') {
        this.next()
        const value = this.parseExpr()
        if (target.k !== 'var' && target.k !== 'index') {
          this.fail('赋值目标必须是变量或数组元素', start)
        }
        return { k: 'assign', target, op: t.text as AssignStmt['op'], value, line: start.line, col: start.col }
      }
      if (t.text === '%=') this.fail('模拟器暂不支持 %= 运算符', t)
      if (t.text === '++' || t.text === '--') {
        this.next()
        if (target.k !== 'var') this.fail('自增/自减仅支持普通变量（不支持数组元素）', start)
        return { k: 'incdec', name: target.name, op: t.text as '++' | '--', line: start.line, col: start.col }
      }
    }
    if (target.k === 'call') {
      if (target.name in UNSUPPORTED_FUNCS) this.fail(UNSUPPORTED_FUNCS[target.name], start)
      this.fail('表达式语句仅支持赋值、自增自减与 __syncthreads()', start)
    }
    this.fail(`无效的语句：'${start.text} …'（仅支持赋值、自增自减与 __syncthreads()）`, start)
  }

  // ---------- 表达式（优先级从低到高） ----------
  private parseExpr(): Expr {
    const e = this.parseTernary()
    const t = this.peek()
    if (t.kind === 'punct' && BITWISE_OPS.has(t.text)) {
      this.fail(`模拟器暂不支持 位运算符 '${t.text}'`, t)
    }
    return e
  }

  private parseTernary(): Expr {
    const c = this.parseOr()
    if (this.atPunct('?')) {
      const q = this.next()
      const t = this.parseExpr()
      this.expectPunct(':', '三元表达式')
      const e = this.parseTernary()
      return { k: 'cond', c, t, e, f: false, site: this.siteCounter++, line: q.line, col: q.col }
    }
    return c
  }

  private parseOr(): Expr {
    let l = this.parseAnd()
    while (this.atPunct('||')) {
      const op = this.next()
      const r = this.parseAnd()
      l = { k: 'logic', op: '||', l, r, f: false, line: op.line, col: op.col }
    }
    return l
  }

  private parseAnd(): Expr {
    let l = this.parseEq()
    while (this.atPunct('&&')) {
      const op = this.next()
      const r = this.parseEq()
      l = { k: 'logic', op: '&&', l, r, f: false, line: op.line, col: op.col }
    }
    return l
  }

  private parseEq(): Expr {
    let l = this.parseRel()
    while (this.atPunct('==') || this.atPunct('!=')) {
      const op = this.next()
      const r = this.parseRel()
      l = { k: 'bin', op: op.text as '==' | '!=', l, r, f: false, line: op.line, col: op.col }
    }
    return l
  }

  private parseRel(): Expr {
    let l = this.parseAdd()
    while (this.atPunct('<') || this.atPunct('<=') || this.atPunct('>') || this.atPunct('>=')) {
      const op = this.next()
      const r = this.parseAdd()
      l = { k: 'bin', op: op.text as '<' | '<=' | '>' | '>=', l, r, f: false, line: op.line, col: op.col }
    }
    return l
  }

  private parseAdd(): Expr {
    let l = this.parseMul()
    while (this.atPunct('+') || this.atPunct('-')) {
      const op = this.next()
      const r = this.parseMul()
      l = { k: 'bin', op: op.text as '+' | '-', l, r, f: false, line: op.line, col: op.col }
    }
    return l
  }

  private parseMul(): Expr {
    let l = this.parseUnary()
    while (this.atPunct('*') || this.atPunct('/') || this.atPunct('%')) {
      const op = this.next()
      const r = this.parseUnary()
      l = { k: 'bin', op: op.text as '*' | '/' | '%', l, r, f: false, line: op.line, col: op.col }
    }
    return l
  }

  private parseUnary(): Expr {
    const t = this.peek()
    if (t.kind === 'punct') {
      if (t.text === '-') {
        this.next()
        return { k: 'un', op: '-', e: this.parseUnary(), f: false, line: t.line, col: t.col }
      }
      if (t.text === '!') {
        this.next()
        return { k: 'un', op: '!', e: this.parseUnary(), f: false, line: t.line, col: t.col }
      }
      if (t.text === '+') {
        this.next()
        return this.parseUnary()
      }
      if (t.text === '*') this.fail('模拟器暂不支持 指针解引用 *p（指针只能以 p[i] 形式访问）', t)
      if (t.text === '&') this.fail('模拟器暂不支持 取地址运算 &x', t)
      if (t.text === '~') this.fail("模拟器暂不支持 位运算符 '~'", t)
      if (t.text === '++' || t.text === '--') {
        this.fail('自增/自减只能作为独立语句使用（如 i++;），不能嵌在表达式里', t)
      }
    }
    return this.parsePostfix()
  }

  private parsePostfix(): Expr {
    let e = this.parsePrimary()
    for (;;) {
      const t = this.peek()
      if (t.kind !== 'punct') break
      if (t.text === '[') {
        this.next()
        if (e.k !== 'var') {
          if (e.k === 'index') this.fail('模拟器暂不支持 多维下标（如 A[i][j]）', t)
          this.fail('仅支持对指针参数与共享数组进行下标访问', t)
        }
        const idx = this.parseExpr()
        this.expectPunct(']')
        e = { k: 'index', name: e.name, idx, space: 'global', f: false, line: t.line, col: t.col }
        continue
      }
      if (t.text === '(') {
        if (e.k !== 'var') this.fail('无效的函数调用', t)
        this.next()
        const args: Expr[] = []
        if (!this.atPunct(')')) {
          for (;;) {
            args.push(this.parseExpr())
            if (this.eatPunct(',')) continue
            break
          }
        }
        this.expectPunct(')')
        e = { k: 'call', name: e.name, args, f: false, line: e.line, col: e.col }
        continue
      }
      if (t.text === '.') {
        this.next()
        const fieldTok = this.peek()
        if (fieldTok.kind !== 'ident') this.fail("'.' 后需要成员名", fieldTok)
        this.next()
        if (e.k !== 'var') this.fail('模拟器暂不支持 struct / 成员访问', t)
        if (fieldTok.text !== 'x' && fieldTok.text !== 'y' && fieldTok.text !== 'z') {
          this.fail(`不支持的成员 '.${fieldTok.text}'（仅支持 .x .y .z）`, fieldTok)
        }
        e = { k: 'dim', base: e.name, field: fieldTok.text, f: false, line: e.line, col: e.col }
        continue
      }
      if (t.text === '->') this.fail("模拟器暂不支持 struct / '->' 成员访问", t)
      break
    }
    return e
  }

  private parsePrimary(): Expr {
    const t = this.peek()
    if (t.kind === 'num') {
      this.next()
      return { k: 'num', v: t.value ?? 0, f: t.isFloat ?? false, line: t.line, col: t.col }
    }
    if (t.kind === 'punct' && t.text === '(') {
      this.next()
      const e = this.parseExpr()
      this.expectPunct(')')
      return e
    }
    if (t.kind === 'ident') {
      if (t.text === 'true') {
        this.next()
        return { k: 'num', v: 1, f: false, line: t.line, col: t.col }
      }
      if (t.text === 'false') {
        this.next()
        return { k: 'num', v: 0, f: false, line: t.line, col: t.col }
      }
      if (KEYWORDS.has(t.text)) this.fail(`此处不能使用关键字 '${t.text}'`, t)
      this.next()
      return { k: 'var', name: t.text, f: false, line: t.line, col: t.col }
    }
    this.fail(`表达式无效：意外的 '${t.text}'`, t)
  }
}

export function parse(toks: Token[]): KernelAst {
  return new Parser(toks).parseKernel()
}

// ============================================================
// 语义 / 类型检查（就地标注 AST）
// ============================================================
type Sym =
  | { kind: 'scalar'; t: ScalarType }
  | { kind: 'ptr'; elem: ScalarType }
  | { kind: 'shared'; elem: ScalarType; len: number }

class Analyzer {
  private readonly scopes: Map<string, Sym>[] = []
  private loopDepth = 0

  private fail(msg: string, node: NodeBase): never {
    throw new CompileErrorException(msg, node.line, node.col)
  }

  private lookup(name: string): Sym | null {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const s = this.scopes[i].get(name)
      if (s) return s
    }
    return null
  }

  analyze(ast: KernelAst): void {
    const root = new Map<string, Sym>()
    for (const p of ast.params) {
      if (root.has(p.name)) this.fail(`参数 '${p.name}' 重复`, p)
      root.set(p.name, p.isPointer ? { kind: 'ptr', elem: p.elem } : { kind: 'scalar', t: p.elem })
    }
    for (const s of ast.shared) {
      if (root.has(s.name)) this.fail(`共享数组 '${s.name}' 与参数同名`, s)
      root.set(s.name, { kind: 'shared', elem: s.elem, len: s.length })
    }
    this.scopes.push(root)
    this.block(ast.body)
    this.scopes.pop()
  }

  private block(body: Stmt[]): void {
    this.scopes.push(new Map())
    for (const s of body) this.stmt(s)
    this.scopes.pop()
  }

  private stmt(s: Stmt): void {
    switch (s.k) {
      case 'decl': {
        const cur = this.scopes[this.scopes.length - 1]
        for (const d of s.decls) {
          if (d.init) this.expr(d.init)
          if (cur.has(d.name)) this.fail(`变量 '${d.name}' 重复声明`, d)
          cur.set(d.name, { kind: 'scalar', t: s.type })
        }
        return
      }
      case 'assign': {
        this.assignTarget(s.target)
        this.expr(s.value)
        return
      }
      case 'incdec': {
        const sym = this.lookup(s.name)
        if (!sym) this.fail(`未声明的变量 '${s.name}'`, s)
        if (sym.kind !== 'scalar') this.fail(`模拟器暂不支持 指针运算：不能对 '${s.name}' 自增/自减`, s)
        return
      }
      case 'if': {
        this.expr(s.c)
        this.block(s.t)
        if (s.e) this.block(s.e)
        return
      }
      case 'while': {
        this.expr(s.c)
        this.loopDepth++
        this.block(s.body)
        this.loopDepth--
        return
      }
      case 'for': {
        this.scopes.push(new Map())
        if (s.init) this.stmt(s.init)
        if (s.c) this.expr(s.c)
        if (s.update) this.stmt(s.update)
        this.loopDepth++
        this.block(s.body)
        this.loopDepth--
        this.scopes.pop()
        return
      }
      case 'blk':
        this.block(s.body)
        return
      case 'break':
        if (this.loopDepth === 0) this.fail('break 只能出现在循环内', s)
        return
      case 'continue':
        if (this.loopDepth === 0) this.fail('continue 只能出现在循环内', s)
        return
      case 'return':
      case 'sync':
        return
    }
  }

  private assignTarget(t: VarExpr | IndexExpr): void {
    if (t.k === 'var') {
      if (t.name === 'warpSize' || DIM_BUILTINS.has(t.name)) {
        this.fail(`不能给内建变量 '${t.name}' 赋值`, t)
      }
      const sym = this.lookup(t.name)
      if (!sym) this.fail(`未声明的变量 '${t.name}'`, t)
      if (sym.kind !== 'scalar') {
        this.fail(`模拟器暂不支持 指针运算：不能给 '${t.name}' 整体赋值（请用 ${t.name}[i] = …）`, t)
      }
      t.f = sym.t === 'float'
      return
    }
    this.indexExpr(t)
  }

  private indexExpr(e: IndexExpr): void {
    const sym = this.lookup(e.name)
    if (!sym) this.fail(`未声明的变量 '${e.name}'`, e)
    if (sym.kind === 'scalar') this.fail(`'${e.name}' 不是指针参数或共享数组，不能下标访问`, e)
    e.space = sym.kind === 'shared' ? 'shared' : 'global'
    this.expr(e.idx)
    if (e.idx.f) this.fail(`数组下标必须是整数表达式（'${e.name}[...]' 的下标是 float）`, e.idx)
    e.f = sym.elem === 'float'
  }

  private expr(e: Expr): void {
    switch (e.k) {
      case 'num':
        return // f 在词法阶段已定
      case 'var': {
        if (e.name === 'warpSize') {
          e.f = false
          return
        }
        if (DIM_BUILTINS.has(e.name)) {
          this.fail(`'${e.name}' 需要使用 .x / .y / .z 访问（如 ${e.name}.x）`, e)
        }
        const sym = this.lookup(e.name)
        if (!sym) {
          if (e.name in UNSUPPORTED_FUNCS) this.fail(UNSUPPORTED_FUNCS[e.name], e)
          this.fail(`未声明的变量 '${e.name}'`, e)
        }
        if (sym.kind !== 'scalar') {
          this.fail(`模拟器暂不支持 指针运算：'${e.name}' 只能以 ${e.name}[i] 形式访问`, e)
        }
        e.f = sym.t === 'float'
        return
      }
      case 'dim': {
        if (!DIM_BUILTINS.has(e.base)) {
          this.fail(`模拟器暂不支持 struct / 成员访问（'.' 仅用于 threadIdx / blockIdx / blockDim / gridDim）`, e)
        }
        e.f = false
        return
      }
      case 'index':
        this.indexExpr(e)
        return
      case 'un': {
        this.expr(e.e)
        e.f = e.op === '-' ? e.e.f : false
        return
      }
      case 'bin': {
        this.expr(e.l)
        this.expr(e.r)
        switch (e.op) {
          case '+':
          case '-':
          case '*':
          case '/':
            e.f = e.l.f || e.r.f
            return
          case '%':
            if (e.l.f || e.r.f) this.fail("'%' 仅支持整数操作数（模拟器暂不支持浮点取模）", e)
            e.f = false
            return
          default:
            e.f = false // 比较运算结果为 int (0/1)
            return
        }
      }
      case 'logic':
        this.expr(e.l)
        this.expr(e.r)
        e.f = false
        return
      case 'cond':
        this.expr(e.c)
        this.expr(e.t)
        this.expr(e.e)
        e.f = e.t.f || e.e.f
        return
      case 'call': {
        if (e.name === '__syncthreads') {
          this.fail('__syncthreads() 只能作为独立语句调用', e)
        }
        if (e.name in UNSUPPORTED_FUNCS) this.fail(UNSUPPORTED_FUNCS[e.name], e)
        const arity = (n: number) => {
          if (e.args.length !== n) this.fail(`${e.name} 需要 ${n} 个参数（得到 ${e.args.length} 个）`, e)
        }
        for (const a of e.args) this.expr(a)
        if (MATH_FLOAT1.has(e.name)) {
          arity(1)
          e.f = true
          return
        }
        if (e.name === 'powf') {
          arity(2)
          e.f = true
          return
        }
        if (e.name === 'min' || e.name === 'max') {
          arity(2)
          e.f = e.args.some((a) => a.f)
          return
        }
        if (e.name === 'abs') {
          arity(1)
          e.f = e.args[0].f
          return
        }
        this.fail(`未知函数 '${e.name}'（支持：min max abs fabsf sqrtf expf logf powf floorf sinf cosf）`, e)
      }
    }
  }
}

export function analyze(ast: KernelAst): void {
  new Analyzer().analyze(ast)
}
