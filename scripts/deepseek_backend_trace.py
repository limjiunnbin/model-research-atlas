"""Trace selected open backend registrations and code ranges without pretending C++ is Python AST."""
import hashlib
import json


def build_trace(e):
 from build_deepseek_atlas import DATA,write
 def sid(path):return next(k for k,s in e.sources.items() if s.get('path')==path)
 def span(path,needle,before=2,after=8):
  source=sid(path);lines=e.text(source).splitlines();start=next(i for i,line in enumerate(lines) if needle in line);lo=max(0,start-before);hi=min(len(lines),start+after)
  s=e.sources[source];body='\n'.join(lines[lo:hi])
  return {'sourceId':source,'path':path,'symbolOrStatement':needle,'line':lo+1,'end':hi,'sourceSha256':s['sha256'],'rangeSha256':hashlib.sha256(body.encode()).hexdigest(),'url':s['url']+f'#L{lo+1}-L{hi}','level':'固定 C++ 源码范围；不是 AST 函数体证明'}
 binding='csrc/torch_binding.cpp'
 rows=[
  {'id':'compressor','topic':'compression','flow':['Python Compressor.forward','torch.ops._C_ascend.compressor','PrivateUse1 注册 → vllm_ascend::compressor','EXEC_NPU_CMD(aclnnCompressor)','Compressor host/tiling 与 arch35 kernel 入口'],'conditions':'ratio/coff、FP32 state、norm/rope shapes、量化 scale、slot mapping 与 token update 在 wrapper/host 层分别检查；arch35 源存在不推定设备编译成功。','ranges':[span(binding,'ops.impl("compressor"'),span(binding,'EXEC_NPU_CMD(aclnnCompressor,'),span('csrc/attention/compressor/op_host/compressor_def.cpp','class Compressor'),span('csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp','namespace'),span('csrc/attention/compressor/op_kernel/arch35/compressor_kernel.h','namespace')]},
  {'id':'hc-pre','topic':'mhc','flow':['DeepseekV4DecoderLayer.hc_pre','torch.ops._C_ascend.npu_hc_pre_v2','PrivateUse1 注册/shape 检查','hc_pre host/tiling','公开 AscendC hc_pre 入口'],'conditions':'wrapper 显式限 hc_mult=4、mix=24；fn/base/scale、dtype、pre_mix 与 Sinkhorn 迭代等分开。TileKernels *_asc 是另一独立实现，没有看到该框架调用 TileKernels 的接线。','ranges':[span(binding,'ops.impl("npu_hc_pre_v2"'),span(binding,'hc_mult == HC_PRE_HC_LIMIT'),span('csrc/moe/hc_pre/op_host/hc_pre_def.cpp','class HcPre'),span('csrc/moe/hc_pre/op_kernel/hc_pre.cpp','extern "C"')]},
  {'id':'gmm-swiglu','topic':'moe','flow':['MoE/量化路径按条件选择融合算子','_C_ascend grouped_matmul_swiglu_quant 注册','host dtype/scale/group-list 条件','aclnn wrapper/workspace/launch'],'conditions':'这条 custom-op 与 torch_npu 分步 grouped_matmul + swiglu_group_quant 是不同候选分支；不串成一个执行链。FP4 容器、NZ/ND、group size 和量化 loader 要独立对应。','ranges':[span(binding,'ops.impl("grouped_matmul_swiglu_quant"'),span('csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_def.cpp','class GroupedMatmulSwigluQuant'),span('csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp','aclnnGroupedMatmulSwigluQuant')]},
 ]
 result={'snapshot':'2026-10-01','repo':'vllm-project/vllm-ascend','revision':'a8fcedb03d93e60efceddbfc912406f7fa491d57','paths':rows,'cannAuditPath':'data/families/deepseek/research/cann-operator-audit.json','limits':['公开注册、wrapper、host/tiling 与入口可追溯；未编译、运行或测量','CANN 匹配版本开源来源继续追踪；实际所走 API/库分支、ABI、硬件数值未由源码存在证明','相同名称/源码存在不证明 vLLM-Ascend 调用了 DeepSeek TileKernels']}
 write(DATA/'research/backend-trace.json',result);return result
