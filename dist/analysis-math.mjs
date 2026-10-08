// Pure analytical functions. No model loading, device execution or timing estimates.
const prod = a => a.reduce((x, y) => x * y, 1);
export function sumFloor(n, r) {
  const a = Math.floor(n / r), b = n % r;
  return r * a * (a - 1) / 2 + a * (b + 1);
}
export function sumCappedFloor(n, r, cap = Infinity) {
  if (cap === 0) return 0;
  const cutoff = r * cap - 1;
  return n <= cutoff ? sumFloor(n, r) : sumFloor(cutoff, r) + cap * (n - cutoff);
}
export function sumWindow(n, window) {
  return n <= window ? n * (n + 1) / 2 : window * (window + 1) / 2 + (n - window) * window;
}
const integer = (x, lo, hi) => Number.isSafeInteger(x) && x >= lo && x <= hi;

export function deployedCacheEstimate(model, {batch, length, dspark=true}) {
  if(!integer(batch,1,1024)||!integer(length,1,model.context))return {valid:false,errors:['Batch 或缓存长度超出有效整数范围']};
  const s=model.storage;
  const records=s.fullOwners.reduce((n,owner)=>n+Math.floor(length/owner.ratio),0);
  const globalMainBytes=batch*records*s.mainRecordBytes;
  const indexBytes=batch*records*s.indexRecordBytes;
  const swaBytes=batch*s.languageLayers*s.window*s.swaRecordBytes;
  const draftBytes=dspark?batch*s.draftStages*s.window*s.swaRecordBytes:0;
  const stateBytes=batch*s.compressionStateBytesPerRequest;
  return {valid:true,batch,length,dspark,globalMainBytes,indexBytes,swaBytes,draftBytes,stateBytes,
    globalBytes:globalMainBytes+indexBytes,totalBytes:globalMainBytes+indexBytes+swaBytes+draftBytes+stateBytes};
}

export function costEstimate(model, input) {
  const {batch: B, query: Q, history: P, phase, cacheBytes: DTYPE, activationBytes: ACT, weightBytes: WEIGHT} = input;
  const c = model.config, T = P + Q, errors = [];
  if (!integer(B, 1, 1024)) errors.push('batch 须为 1–1024 的整数');
  if (!integer(Q, 1, model.context) || !integer(P, 0, model.context) || T > model.context) errors.push(`history + query 须在 1–${model.context} 内`);
  if (!['prefill', 'decode', 'verify'].includes(phase)) errors.push('请选择有效阶段');
  if (phase === 'decode' && Q !== 1) errors.push('常规 decode 的 query 固定为 1；块验证请选择 verify');
  if (![1, 2, 4].includes(DTYPE) || ![1, 2, 4].includes(ACT) || ![0.5, 1, 2, 4].includes(WEIGHT)) errors.push('精度假设无效');
  if (!['last', 'all'].includes(input.logits)) errors.push('logits 范围无效');
  if (!['expanded', 'latent'].includes(input.path)) errors.push('缓存路径无效');
  if (phase === 'verify' && input.logits !== 'all') errors.push('verify 需要每个 query 的目标 logits');
  if (errors.length) return {valid: false, errors};
  const selected = input.layer && input.layer !== 'all';
  const groups = selected ? model.groups.filter(g => g.nodes.includes(input.layer)).map(g => ({...g, count: 1})) : model.groups;
  if (!groups.length) return {valid: false, errors: ['找不到所选层']};
  const H = c.hidden_size, NH = c.num_attention_heads ?? c.n_heads, V = c.vocab_size;
  let projectionFlops = 0, coreFlops = 0, usefulCoreFlops = 0, indexFlops = 0;
  let cacheElements = 0, visibleCacheElements = 0, indexCacheBytes = 0, compressionStateBytes = 0;
  let projectionIOBytes = 0, scoreBytes = 0, residualBytes = 0, queryBytes = 0, expertIntermediateBytes = 0;
  const rows = [];
  for (const g of groups) {
    let projection = 0, io = 0;
    for (const m of g.linears) {
      const calls = m.callsPerToken;
      projection += 2 * B * Q * prod(m.shape) * calls;
      const [out, inn] = m.shape;
      io += B * Q * (out + inn * (m.inputGroups ?? 1)) * calls * ACT;
    }
    let core, useful, kv, visibleKV, idx = 0, idxBytes = 0, state = 0, score;
    const densePairs = Q * T, causalPairs = Q * P + Q * (Q + 1) / 2;
    if (model.kind === 'v4') {
      const r = g.ratio, D = c.head_dim, W = c.sliding_window ?? c.window_size, C = r ? Math.floor(T / r) : 0;
      const cap = r === 4 ? c.index_topk : Infinity;
      const slots = W + (r ? Math.min(C, cap) : 0);
      const usefulPairs = sumWindow(T, W) - sumWindow(P, W) + (r ? sumCappedFloor(T, r, cap) - sumCappedFloor(P, r, cap) : 0);
      // Slot contraction is a bound; masked/padded slots are not guaranteed kernel work.
      core = 4 * B * NH * D * Q * slots;
      useful = 4 * B * NH * D * usefulPairs;
      kv = B * (W + C) * D;
      visibleKV = B * (Math.min(W, T) + C) * D;
      if (r) {
        const coff = r === 4 ? 2 : 1;
        state = B * 2 * coff * r * coff * D * 4;
        if (r === 4) {
          idx = 2 * B * Q * C * c.index_n_heads * c.index_head_dim;
          idxBytes = B * C * c.index_head_dim * DTYPE;
          state += B * 2 * coff * r * coff * c.index_head_dim * 4;
        }
      }
      score = B * NH * Q * slots * 4;
      queryBytes = Math.max(queryBytes, B * Q * NH * D * ACT);
    } else if (model.kind === 'gqa') {
      const D = c.head_dim ?? H / NH, NK = c.num_key_value_heads;
      core = 4 * B * NH * D * densePairs;
      useful = 4 * B * NH * D * causalPairs;
      kv = visibleKV = 2 * B * T * NK * D;
      score = B * NH * densePairs * 4;
      queryBytes = Math.max(queryBytes, B * Q * NH * D * ACT);
    } else {
      const R = c.kv_lora_rank, DN = c.qk_nope_head_dim, DR = c.qk_rope_head_dim, DV = c.v_head_dim;
      const latent = input.path === 'latent';
      const Dscore = latent ? R + DR : DN + DR, Dvalue = latent ? R : DV;
      core = 2 * B * NH * densePairs * (Dscore + Dvalue);
      // DSA reference still forms dense scores before masking. Selected-pair work is separate.
      const selectedPairs = model.kind === 'dsa' ? sumWindow(T, c.index_topk) - sumWindow(P, c.index_topk) : causalPairs;
      useful = 2 * B * NH * selectedPairs * (Dscore + Dvalue);
      kv = visibleKV = B * T * (latent ? R + DR : NH * (DN + DR + DV));
      if (model.kind === 'dsa') {
        idx = 2 * B * Q * T * c.index_n_heads * c.index_head_dim;
        idxBytes = B * T * (c.index_head_dim + 4); // Reference FP8 key + FP32 scale.
      }
      // Replacing new-token Wkv_b expansion with Q absorption and output restoration
      // has the same 2 * B * Q * NH * R * (DN + DV) multiplication count here.
      score = B * NH * densePairs * 4;
      queryBytes = Math.max(queryBytes, B * Q * NH * (DN + DR) * ACT);
    }
    const n = g.count;
    projectionFlops += projection * n; projectionIOBytes += io * n;
    coreFlops += core * n; usefulCoreFlops += useful * n; indexFlops += idx * n;
    cacheElements += kv * n; visibleCacheElements += visibleKV * n;
    indexCacheBytes += idxBytes * n; compressionStateBytes += state * n;
    scoreBytes = Math.max(scoreBytes, score);
    residualBytes = Math.max(residualBytes, B * Q * (c.hc_mult ?? 1) * H * ACT);
    expertIntermediateBytes = Math.max(expertIntermediateBytes, 2 * B * Q * g.selectedExperts * g.intermediate * ACT);
    rows.push({type: g.type, ratio: g.ratio, ffn: g.ffn, layers: n, projectionFlops: projection * n, coreFlops: core * n,
      usefulCoreFlops: useful * n, indexFlops: idx * n, mainKvBytes: kv * n * DTYPE, indexCacheBytes: idxBytes * n, compressionStateBytes: state * n});
  }
  const logitsTokens = input.logits === 'all' ? Q : 1;
  const headFlops = selected ? 0 : 2 * B * logitsTokens * V * H;
  const headControlFlops = selected ? 0 : 2 * B * Q * model.headControlElements;
  const headIO = selected ? 0 : B * logitsTokens * (H + V) * ACT;
  const logitsBytes = selected ? 0 : B * logitsTokens * V * 4;
  const mainKvBytes = cacheElements * DTYPE, visibleKvBytes = visibleCacheElements * DTYPE;
  const warnings = [];
  if (['v4', 'dsa'].includes(model.kind) && P > 0 && Q > 1) warnings.push('这是多 query 的数学估算；所选 native 增量 demo 仅处理单 token，不据此声明其支持块验证或分块预填充。');
  if (model.defaultLogits==='last' && input.logits==='all' && Q>1) warnings.push('本例按全部 query 的理论输出投影计数；所选 native demo 默认只返回最后 query 的 logits，需要另核对框架的输出分支。');
  return {valid: true, errors: [], input: {...input}, modelId: model.id, totalTokens: T,
    projectionFlops, headFlops, headControlFlops, coreFlops, usefulCoreFlops, indexFlops,
    contractionFlops: projectionFlops + headFlops + headControlFlops + coreFlops + indexFlops,
    usefulContractionFlops: projectionFlops + headFlops + headControlFlops + usefulCoreFlops + indexFlops,
    mainKvBytes, visibleKvBytes, indexCacheBytes, compressionStateBytes,
    sequenceCacheBytes: mainKvBytes + indexCacheBytes + compressionStateBytes,
    pdMainKvBytes: visibleKvBytes, projectionIOBytes: projectionIOBytes + headIO,
    publishedWeightBytes: model.publishedWeightBytes, mainLogicalWeightBytes: model.mainLogicalParameters * WEIGHT,
    declaredTensors: {queryBytes, residualBytes, scoreBytes, expertIntermediateBytes, logitsBytes}, rows,
    warnings, measuredLatency: null, measuredThroughput: null, measuredPeakMemory: null};
}

export function parallelEstimate(model, input) {
  const {tp, dp, batch, query, history, weightBytes, activationBytes, cacheBytes, redundantExperts, remoteFraction, allReduces} = input;
  const c = model.config, experts = c.n_routed_experts ?? c.n_routed_experts_count ?? 0;
  const topK = c.num_experts_per_tok ?? c.n_activated_experts ?? 0;
  const ep = experts ? tp * dp : 1, errors = [];
  if (![tp, dp].every(x => integer(x, 1, 128))) errors.push('TP 与 DP 须为 1–128 的整数');
  if (!integer(redundantExperts, 0, 4096) || (!experts && redundantExperts)) errors.push('冗余专家数量无效');
  if (experts && (experts + redundantExperts) % ep) errors.push('此均分模型要求（专家总数 + 冗余专家总数）能被 EP 整除');
  if (remoteFraction !== 'uniform' && (!Number.isFinite(remoteFraction) || remoteFraction < 0 || remoteFraction > 1)) errors.push('远端路由比例须在 0–1 内');
  if (!integer(allReduces, 0, 8)) errors.push('每层 AllReduce 次数须为 0–8 的整数');
  if (!['replicated', 'uniform-tp'].includes(input.policy)) errors.push('请选择容量假设');
  const cost = costEstimate(model, {batch, query, history, cacheBytes, activationBytes, weightBytes, phase: query === 1 ? 'decode' : 'prefill', logits: 'all', path: 'latent', layer: 'all'});
  if (!cost.valid) errors.push(...cost.errors);
  if (errors.length) return {valid: false, errors};
  const s = model.parallelElements;
  const replicated = s.replicated, linear = s.nonExpertLinears, routed = s.routedExperts;
  const shared = input.policy === 'uniform-tp' ? linear / tp : linear;
  const expertPerRank = experts ? routed * (1 + redundantExperts / experts) / ep : 0;
  const rankElements = replicated + shared + expertPerRank;
  const globalTokens = batch * query * dp;
  const assignments = globalTokens * topK;
  const remote = experts ? (remoteFraction === 'uniform' ? (ep - 1) / ep : remoteFraction) : 0;
  const hidden = c.hidden_size, layers = c.num_hidden_layers;
  const dispatchBytes = assignments * hidden * activationBytes * remote;
  const combineBytes = dispatchBytes;
  const moeLayers = model.groups.filter(g => g.ffn === 'MoE').reduce((sum, g) => sum + g.count, 0);
  const moeSendBytes = (dispatchBytes + combineBytes) * moeLayers;
  const tpRingPerRank = 2 * (tp - 1) / tp * batch * query * hidden * activationBytes * allReduces * layers;
  const tpRingBytes = tpRingPerRank * tp * dp;
  return {valid: true, errors: [], input: {...input}, ep, ranks: tp * dp, globalTokens, assignments,
    remoteFraction: remote, logicalParametersPerRank: rankElements, logicalWeightBytesPerRank: rankElements * weightBytes,
    replicatedElementsPerRank: replicated, nonExpertLinearElementsPerRank: shared, expertElementsPerRank: expertPerRank,
    dispatchSendBytesPerMoeLayer: dispatchBytes, combineSendBytesPerMoeLayer: combineBytes, moeSendBytes,
    uniformMoeSendBytesPerRank: moeSendBytes / ep, tpRingSendBytes: tpRingBytes, tpRingSendBytesPerRank: tpRingPerRank,
    pdMainKvBytesPerReplica: cost.pdMainKvBytes, indexCacheBytesPerReplica: cost.indexCacheBytes,
    measuredCommunicationTime: null, measuredRankPeakMemory: null};
}

export function checkContract(contract, input) {
  const results = [];
  for (const r of contract.rules) {
    let failed = false, unknown = false;
    const v = input[r.field];
    if (r.kind === 'enum') { unknown = v == null; failed = !unknown && !r.values.includes(v); }
    else if (r.kind === 'rank') { unknown = !Array.isArray(v); failed = !unknown && (v.length < r.min || v.length > r.max); }
    else if (r.kind === 'nonempty') { unknown = !Array.isArray(v); failed = !unknown && (!v.length || v.some(x=>x===0)); }
    else if (r.kind === 'swiglu-axis') {
      unknown = !Array.isArray(input.x) || !Number.isInteger(input.dim);
      if (!unknown) {
        const n = input.x.length, dim = input.dim;
        failed = n === 0 || dim < -n || dim >= n || input.x[(dim + n) % n] % 2 !== 0;
      }
    } else if (r.kind === 'suffix') {
      unknown = !Array.isArray(input.x) || !Array.isArray(input.gamma);
      if (!unknown) failed = !input.gamma.length || input.gamma.length > input.x.length || input.gamma.some((x, i) => x !== input.x[input.x.length - input.gamma.length + i]);
    } else if (r.kind === 'matmul-k') {
      unknown = !Array.isArray(input.x) || !Array.isArray(input.w) || input.x.length !== 2 || input.w.length !== 2;
      if (!unknown) failed = input.x[1] !== input.w[0];
    } else unknown = true;
    results.push({ruleId: r.id, status: unknown ? 'unknown' : failed ? 'contradiction' : 'necessary_condition_met', label: r.label, reference: r.reference});
  }
  return {status: results.some(x => x.status === 'contradiction') ? 'contradiction' : 'incomplete_static_check', results,
    message: results.some(x => x.status === 'contradiction') ? '发现与已编码必要条件矛盾的输入。' : '未发现已编码条件矛盾；其他条件、库分派及设备支持仍未由此证明。'};
}

export const softmax = x => {const m = Math.max(...x), e = x.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s);};
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const matmul = (a, b) => a.map(row => b[0].map((_, j) => dot(row, b.map(r => r[j]))));
export const transpose = a => a[0].map((_, j) => a.map(r => r[j]));
const fixed = (rows, cols, seed = 1) => Array.from({length: rows}, (_, i) => Array.from({length: cols}, (_, j) => Math.sin((i + 1) * (j + 2) + seed) * 0.6));
const sigmoid = x => 1 / (1 + Math.exp(-x));

export function mechanism(kind, options = {}) {
  if (kind === 'mla') {
    const R = options.rank ?? 3, C = fixed(4, R), Wk = fixed(2, R, 3), Wv = fixed(2, R, 5), q = [[0.35, -0.2]], ropeQ = 0.2, ropeK = [-0.1, 0.2, 0.1, -0.3];
    const K = matmul(C, transpose(Wk)), V = matmul(C, transpose(Wv));
    const expandedScores = K.map((r, i) => dot(q[0], r) + ropeQ * ropeK[i]);
    const qAbs = matmul(q, Wk)[0], latentScores = C.map((r, i) => dot(qAbs, r) + ropeQ * ropeK[i]);
    const probability = softmax(expandedScores), outExpanded = matmul([probability], V)[0];
    const outLatent = matmul(matmul([probability], C), transpose(Wv))[0];
    return {steps: ['输入潜变量与投影', '展开 K/V', '两条路径打分', '相同 softmax', '输出与缓存比较'], C, Wk, Wv, K, V, expandedScores, latentScores, probability, outExpanded, outLatent,
      maxError: Math.max(...outExpanded.map((v, i) => Math.abs(v - outLatent[i]))), expandedCacheElements: 4 * (2 + 1 + 2), latentCacheElements: 4 * (R + 1)};
  }
  if (kind === 'dsa') {
    const topK = options.topK ?? 3, visible = options.visible ?? 6, keys = fixed(8, 2, 2), q = [[0.4, -0.1], [-0.2, 0.3]], headWeights = [0.7, 0.3];
    const indexScores = keys.map(k => q.reduce((s, h, i) => s + headWeights[i] * Math.max(0, dot(h, k)), 0));
    const selected = indexScores.map((score, i) => ({score, i})).filter(x => x.i < visible).sort((a, b) => b.score - a.score || a.i - b.i).slice(0, topK).map(x => x.i);
    const targetScores = keys.map(k => dot([0.2, 0.5], k));
    const masked = targetScores.map((v, i) => selected.includes(i) ? v : -Infinity);
    return {steps: ['可见历史与因果边界', 'Indexer 的分数', '选择 top-k', '独立 attention 分数加 mask', '聚合选中上下文'], keys, indexScores, targetScores, selected, visible, probability: softmax(masked)};
  }
  if (kind === 'compression') {
    const n = options.tokens ?? 11, r = options.ratio ?? 4, window = 8, values = fixed(n, 2, 4), compressed = [];
    for (let block = 0; block < Math.floor(n / r); block++) {
      const current = values.slice(block * r, (block + 1) * r);
      const previous = r === 4 && block > 0 ? values.slice((block - 1) * r, block * r) : [];
      const source = [...previous, ...current];
      const weights = source.map((v, i) => 0.3 * v[0] + 0.1 * i);
      compressed.push({block, positions: Array.from({length: source.length}, (_, i) => (previous.length ? (block - 1) * r : block * r) + i), weights: softmax(weights), value: matmul([softmax(weights)], source)[0]});
    }
    return {steps: ['原始 token', '完整分块与 overlap', '教学 gating 池化', '尾部保留在状态中', '窗口与压缩前缀并存'], values, compressed,
      remainder: values.slice(Math.floor(n / r) * r), remainderStart: Math.floor(n / r) * r, windowPositions: Array.from({length: Math.min(n, window)}, (_, i) => Math.max(0, n - window) + i), ratio: r};
  }
  if (kind === 'mhc') {
    const iterations = options.iterations ?? 20, residual = fixed(4, 2, 1), pre = [0.2, 0.4, 0.3, 0.5], post = [0.7, 0.8, 0.6, 0.9];
    let comb = fixed(4, 4, 7).map(r => r.map(Math.exp));
    for (let k = 0; k < iterations; k++) {
      comb = comb.map(r => {const s = r.reduce((a, b) => a + b, 0); return r.map(v => v / s);});
      const columns = transpose(comb).map(r => r.reduce((a, b) => a + b, 0));
      comb = comb.map(r => r.map((v, j) => v / columns[j]));
    }
    const input = matmul([pre], residual)[0], sublayer = input.map((v, j) => Math.tanh(v + [0.1, -0.15][j]));
    const mixed = matmul(transpose(comb), residual), output = mixed.map((r, j) => r.map((v, d) => v + post[j] * sublayer[d]));
    return {steps: ['四条残差流', 'pre 折叠为子层输入', '教学子层变换', 'Sinkhorn comb', 'post 还原四流'], residual, pre, post, input, sublayer, comb, output,
      rowSums: comb.map(r => r.reduce((a, b) => a + b, 0)), columnSums: transpose(comb).map(r => r.reduce((a, b) => a + b, 0))};
  }
  if (kind === 'dspark') {
    const count = options.block ?? 5, threshold = options.threshold ?? 0.6, vocab = 5, W1 = fixed(vocab, 2, 2), W2 = fixed(vocab, 2, 4);
    const confidences = Array.from({length: count}, (_, i) => sigmoid(2.5 - 0.55 * i));
    let survival = 1; const prefixSurvival = confidences.map(c => survival *= c);
    const verifyLength = Math.max(1, prefixSurvival.filter(s => s >= threshold).length);
    const baseLogits = fixed(count, vocab, 3), probabilities = [], drafts = [];
    let previous = 1;
    const sample = (p, u) => {let s = 0; return p.findIndex((v, i) => (s += v) >= u || i === p.length - 1);};
    for (let i = 0; i < count; i++) {
      const bias = W2.map(w => dot(w, W1[previous]));
      const p = softmax(baseLogits[i].map((v, j) => v + bias[j]));
      previous = sample(p, (0.27 + i * 0.17) % 1); probabilities.push(p); drafts.push(previous);
    }
    const verification = []; let accepted = 0, rejection = null;
    for (let i = 0; i < verifyLength; i++) {
      const target = softmax(baseLogits[i].map((v, j) => v + 0.8 * Math.cos((j + 1) * (i + 2))));
      const alpha = Math.min(1, target[drafts[i]] / probabilities[i][drafts[i]]), draw = (0.79 + i * 0.08) % 1;
      const ok = draw <= alpha;
      verification.push({position: i, token: drafts[i], target, draft: probabilities[i], alpha, draw, accepted: ok});
      if (!ok) {
        const residual = target.map((v, j) => Math.max(0, v - probabilities[i][j])), total = residual.reduce((a, b) => a + b, 0);
        rejection = {position: i, correctionDistribution: residual.map(v => v / total), correction: sample(residual.map(v => v / total), 0.43)};
        break;
      }
      accepted++;
    }
    const committed = [...drafts.slice(0, accepted)];
    if (rejection) committed.push(rejection.correction);
    else committed.push(sample(softmax(fixed(1, vocab, 9)[0]), 0.43)); // Teaching bonus target token.
    return {steps: ['并行 backbone 的教学 logits', '轻量 Markov 顺序选 token', 'confidence 与前缀生存概率', 'target 的接受/拒绝校正', '仅提交有效前缀与校正/bonus'], baseLogits, W1, W2, drafts, probabilities,
      confidences, prefixSurvival, expectedAcceptedPrediction: prefixSurvival.reduce((a, b) => a + b, 0), verifyLength, verification, rejection, accepted, committed};
  }
  throw Error('Unknown mechanism');
}
