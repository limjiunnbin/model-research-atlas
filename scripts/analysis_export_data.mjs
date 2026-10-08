// Flat scientific tables. Inputs and units remain distinct from computed values.
export function exportTables(d, scenarios, comparison) {
  const costRows=scenarios.costScenarios.map(s=>{
    const i=s.input,r=s.result,m=d.costModels.find(m=>m.id===s.modelId);
    return [s.name,s.modelId,m.revision,i.phase,i.path,i.batch,i.query,i.history,i.logits,i.cacheBytes,i.activationBytes,i.weightBytes,
      r.projectionFlops,r.coreFlops,r.usefulCoreFlops,r.indexFlops,r.headFlops+r.headControlFlops,r.contractionFlops,
      r.mainKvBytes,r.indexCacheBytes,r.compressionStateBytes,r.sequenceCacheBytes,r.mainLogicalWeightBytes,r.publishedWeightBytes,r.projectionIOBytes,r.declaredTensors.scoreBytes,
      r.measuredLatency,r.measuredThroughput,'声明矩阵收缩；逻辑字节与精度假设；主干范围；设备未测'];
  });
  const contractRows=d.cannContracts.flatMap(c=>c.claims.map(r=>[c.id,r.apiDocument,r.context??'API 入口',r.parameter,r.categories.join(' / '),r.statement,
    r.reference.revision,r.reference.line,r.reference.end,r.reference.url,r.reference.rangeSha256,r.tableMapping==='raw_row_with_context'?'合并单元格保留原始列/上下文；未自动映射契约':'固定表行/条件；不证明实际支持']));
  const parallelRows=scenarios.parallelScenarios.map(s=>{
    const i=s.input,r=s.result;
    return [s.name,i.tp,i.dp,r.ep,i.policy,i.batch,i.query,i.history,i.redundantExperts,i.weightBytes,i.activationBytes,r.remoteFraction,i.allReduces,
      r.logicalParametersPerRank,r.logicalWeightBytesPerRank,r.dispatchSendBytesPerMoeLayer,r.combineSendBytesPerMoeLayer,r.moeSendBytes,r.uniformMoeSendBytesPerRank,
      r.tpRingSendBytesPerRank,r.tpRingSendBytes,r.pdMainKvBytesPerReplica,r.indexCacheBytesPerReplica,r.measuredCommunicationTime,r.measuredRankPeakMemory,
      '均分/路由/ring/精度为显式假设；量化 metadata、padding、ABI 与运行布局另核对'];
  });
  const comparisonRows=comparison.models.map(m=>[m.family,m.name,m.branch,m.revision,m.facts.layers?.value??null,m.facts.attention?.value??null,m.facts.parameters?.value??null,
    m.facts.payload?.value??null,m.facts.quantization?.value??null,m.evidenceDepth,typeof m.parameterScope==='string'?m.parameterScope:JSON.stringify(m.parameterScope),m.cache==null?null:JSON.stringify(m.cache),m.source]);
  const mechanismRows=d.mechanisms.map(m=>[m.title,m.id,m.limit,m.references.map(r=>r.symbol).join(';'),m.references.map(r=>r.url).join(';')]);
  return [
    {name:'成本场景',title:'DeepSeek 静态成本场景',note:'每行是独立固定样例。FLOPs 为声明矩阵收缩次数，bytes 为逻辑字节。修改输入请使用网页计算器。',file:'DeepSeek-cost-scenarios.csv',
      headers:['模型','model_id','固定 revision','阶段','MLA 路径','B','Q','history','logits','cache B/元素','act B/元素','weight B/元素','投影 FLOPs','Attention 上界 FLOPs','有效 Attention FLOPs','Indexer FLOPs','Head/折叠 FLOPs','收缩合计 FLOPs','主 KV bytes','Index cache bytes','FP32 状态 bytes','Cache/状态合计 bytes','统一精度主干权重 bytes','实际发布载荷 bytes','投影逻辑 IO bytes','单层 score 逻辑 bytes','实测延迟 ms','实测吞吐 token/s','范围'],rows:costRows,
      widths:[40,32,46,16,18,10,12,14,14,18,18,20,25,28,28,25,26,27,25,26,25,30,31,30,30,30,22,24,85]},
    {name:'并行场景',title:'并行容量与逻辑通信场景',note:'容量假设不代表实际 loader 分片。通信按发送端计一次；时间与峰值显存未测。',file:'DeepSeek-parallel-scenarios.csv',
      headers:['模型','TP','DP','EP','非专家容量假设','B/replica','Q','history','冗余专家总数','weight B/元素','通信 B/元素','远端路由比例','AllReduce 次数/层','每 rank 逻辑参数','每 rank 权重 bytes','Dispatch bytes/MoE层','Combine bytes/MoE层','所有 MoE层发送 bytes','均匀每 rank MoE发送 bytes','每 rank TP ring发送 bytes','所有 TP组 ring发送 bytes','P/D主 KV bytes/replica','Index cache bytes/replica','实测通信 ms','实测峰值显存 bytes','范围'],rows:parallelRows,
      widths:[40,10,10,10,26,14,12,15,19,20,20,22,26,30,30,31,31,32,36,36,36,34,34,22,30,95]},
    {name:'跨家族比较',title:'现有模型条目的统一比较',note:'未知值保持空白。逻辑参数、载荷和辅助模块沿用各版统计口径，未进行速度或能力排名。',file:'跨家族比较.csv',
      headers:['家族','模型','分支','固定 revision','主干层','注意力结构','逻辑参数','权重有效载荷 bytes','精度披露/配置','核验深度','参数统计范围','Cache 表示/公式','官方/固定来源'],rows:comparisonRows,
      widths:[18,42,18,46,14,58,28,31,75,60,105,115,105]},
    {name:'机制要点',title:'机制教学演示与证据边界',note:'演示为小规模合成数据；真实模型维度、权重和服务协议见对应研究。',
      headers:['机制','id','教学边界','参考符号','固定来源'],rows:mechanismRows,widths:[45,24,115,75,120]},
    {name:'算子条件',title:'CANN API 表行与条件',note:'每行保留 API 文档及章节上下文。不同平台/模式的条件不合并成无条件支持。',file:'DeepSeek-CANN-contracts.csv',
      headers:['源码家族','API 文档','章节上下文','参数/条件','类别','原文表行/条件','固定 revision','起始行','结束行','来源','范围 SHA-256','证据范围'],rows:contractRows,
      widths:[34,60,72,44,40,115,46,14,14,120,72,92]},
  ];
}
