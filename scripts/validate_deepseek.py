"""Validate DeepSeek P0/P1 research; this does not validate website data or execute a model."""
import argparse
import ast
import collections
import csv
from decimal import Decimal
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/families/deepseek'
OUT=ROOT/'research/deepseek/round1'


def read(path): return json.loads(path.read_text())


def dim(expression, env):
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) is int: return n.value
        if isinstance(n,ast.Name): return env[n.id]
        if isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add): return a+b
            if isinstance(n.op,ast.Sub): return a-b
            if isinstance(n.op,ast.Mult): return a*b
            if isinstance(n.op,ast.FloorDiv): return a//b
        raise ValueError(f'Invalid dimension: {expression}')
    value=visit(ast.parse(str(expression),mode='eval').body)
    assert isinstance(value,int) and value>=0,(expression,value)
    return value


def broadcast(shapes):
    result=[]
    for axis in itertools.zip_longest(*(reversed(x) for x in shapes),fillvalue=1):
        nonunit={x for x in axis if x!=1}
        assert len(nonunit)<=1,('broadcast mismatch',shapes)
        result.append(max(axis))
    return tuple(reversed(result))


def infer_einsum(equation, shapes):
    inputs,out=equation.split('->')
    axes=inputs.split(','); sizes={}
    assert len(axes)==len(shapes)
    for names,shape in zip(axes,shapes):
        assert len(names)==len(shape),(equation,shape)
        for name,size in zip(names,shape):
            assert name not in sizes or sizes[name]==size,('einsum contraction',equation,name,sizes.get(name),size)
            sizes[name]=size
    assert len(set(out))==len(out)
    return tuple(sizes[name] for name in out)


def check_step(step,env):
    inputs=[tuple(dim(x,env) for x in shape) for shape in step['inputs'].values()]
    outputs=[tuple(dim(x,env) for x in shape) for shape in step['outputs'].values()]
    op=step['op']; first=inputs[0]; target=outputs[0]
    if op=='linear':
        weight=inputs[1]
        assert len(weight)==2 and first[-1]==weight[1]
        assert target==first[:-1]+(weight[0],)
    elif op=='einsum': assert target==infer_einsum(step['einsum'],inputs)
    elif op=='reshape': assert math.prod(first)==math.prod(target)
    elif op in ('add','multiply'): assert target==broadcast(inputs)
    elif op in ('norm','activation','zeros_like','masked_softmax','rope'):
        assert first==target
        if op=='rope': assert first[-1]==env['D_rope'] and first[-1]%2==0
        if op=='masked_softmax':
            if env['phase']=='prefill':
                assert env['start_pos']==0 and env['L_q']==env['L_kv']
                assert target==broadcast([first,(env['L_q'],1,env['L_q'])])
    elif op=='broadcast': assert target==broadcast([first,target])
    elif op in ('split','concat'):
        axis=step.get('axis',-1)%len(first)
        pieces=outputs if op=='split' else inputs
        whole=first if op=='split' else target
        assert all(len(p)==len(whole) for p in pieces)
        assert sum(p[axis] for p in pieces)==whole[axis]
        assert all(all(p[a]==whole[a] for a in range(len(whole)) if a!=axis) for p in pieces)
    elif op=='cache_write':
        assert len(first)==len(target) and first[0]==target[0] and first[2:]==target[2:]
        assert first[1]+env['start_pos']==target[1]
    elif op=='topk':
        k=dim(step['k'],env)
        assert 0<k<=first[-1] and target==first[:-1]+(k,)
    elif op=='group_top2':
        assert first[-1]>=2 and target==first[:-1]
    elif op=='group_mask':
        assert first==(env['N_tok'],env['E']) and inputs[1]==(env['N_tok'],env['G_keep']) and target==first
    elif op=='gather':
        assert inputs[0][0]==inputs[1][0] and target==inputs[1]
    elif op=='dispatch':
        assert first==(env['N_tok'],env['H']) and inputs[1]==(env['N_tok'],env['K'])
        assert target==(env['N_e'],env['H']) and env['N_e']<=env['N_tok']
        assert outputs[1:]==[(env['N_e'],),(env['N_e'],)]
    elif op=='expert_weight_gather':
        assert first==(env['N_tok'],env['K']) and inputs[1:]==[(env['N_e'],),(env['N_e'],)]
        assert target==(env['N_e'],1)
    elif op=='scatter':
        assert first==(env['N_e'],env['H']) and inputs[1]==(env['N_e'],) and inputs[2]==(env['N_tok'],env['H'])
        assert target==(env['N_tok'],env['H'])
    else: raise AssertionError(f'Unvalidated operation kind: {op}')
    return inputs,outputs


def algebra_check():
    """FP64 synthetic matrices: independent expanded vs absorbed MLA algebra, no upstream code."""
    rng=random.Random(20261001)
    heads,dn,dr,rank,dv,length=2,3,2,4,3,5
    def vector(n): return [rng.uniform(-0.7,0.7) for _ in range(n)]
    c=[vector(rank) for _ in range(length)]; pe=[vector(dr) for _ in range(length)]
    score_errors=[]; output_errors=[]
    for _ in range(heads):
        wk=[vector(rank) for _ in range(dn)]; wv=[vector(rank) for _ in range(dv)]
        for _ in range(3):
            q=vector(dn); qr=vector(dr)
            expanded_k=[[sum(wk[d][r]*cs[r] for r in range(rank)) for d in range(dn)] for cs in c]
            expanded_v=[[sum(wv[d][r]*cs[r] for r in range(rank)) for d in range(dv)] for cs in c]
            absorbed_q=[sum(q[d]*wk[d][r] for d in range(dn)) for r in range(rank)]
            a=[sum(q[d]*ks[d] for d in range(dn))+sum(qr[d]*ps[d] for d in range(dr)) for ks,ps in zip(expanded_k,pe)]
            b=[sum(absorbed_q[r]*cs[r] for r in range(rank))+sum(qr[d]*ps[d] for d in range(dr)) for cs,ps in zip(c,pe)]
            score_errors.extend(abs(x-y) for x,y in zip(a,b))
            def softmax(values):
                exps=[math.exp(v-max(values)) for v in values]
                return [v/sum(exps) for v in exps]
            pa,pb=softmax(a),softmax(b)
            naive=[sum(pa[s]*expanded_v[s][d] for s in range(length)) for d in range(dv)]
            latent=[sum(pb[s]*c[s][r] for s in range(length)) for r in range(rank)]
            absorbed=[sum(latent[r]*wv[d][r] for r in range(rank)) for d in range(dv)]
            output_errors.extend(abs(x-y) for x,y in zip(naive,absorbed))
    assert max(score_errors)<1e-12 and max(output_errors)<1e-12
    return {'seed':20261001,'numericType':'Python float / FP64 synthetic algebra',
            'maxScoreAbsError':max(score_errors),'maxOutputAbsError':max(output_errors),
            'limit':'固定合成小矩阵的代数复算；未测试 RoPE 实现、FP8/BF16 舍入、checkpoint 或设备 kernel'}


def routing_check(c):
    rng=random.Random(13); E=c['n_routed_experts']; K=c['num_experts_per_tok']; G=c['n_group']; keep=c['topk_group']
    assert E%G==0 and E//G>=2 and 0<K<=keep*(E//G)
    counts=[0]*E; records=[]
    for _ in range(8):
        scores=[rng.uniform(0.05,0.95) for _ in range(E)]
        bias=[rng.uniform(-0.6,0.6) for _ in range(E)]
        choice=[s+b for s,b in zip(scores,bias)]
        group_scores=[sum(sorted(choice[g*(E//G):(g+1)*(E//G)],reverse=True)[:2]) for g in range(G)]
        groups=sorted(range(G),key=lambda g:group_scores[g],reverse=True)[:keep]
        ids=sorted((e for e in range(E) if e//(E//G) in groups),key=lambda e:choice[e],reverse=True)[:K]
        weights=[scores[e]/sum(scores[i] for i in ids)*c['routed_scaling_factor'] for e in ids]
        assert len(set(ids))==K and all(e//(E//G) in groups for e in ids)
        assert abs(sum(weights)-c['routed_scaling_factor'])<1e-12
        for e in ids: counts[e]+=1
        records.append({'ids':ids,'weightsSum':sum(weights)})
    assert sum(counts)==8*K and max(counts)<=8
    return {'syntheticTokens':8,'assignments':sum(counts),'expectedAssignments':8*K,
            'weightSumAfterScale':c['routed_scaling_factor'],'limit':'配置约束和参考路由数学关系复算，不运行上游 Gate'}


def validate(research=None,check_exports=True):
    d=read(DATA/'research/v3-round1.json'); c=read(ROOT/d['configPath'])
    sources={s['id']:s for s in read(DATA/'sources.json')['sources']}; checks=collections.Counter()
    assert d['kind']=='deepseek_v3_round1_research' and d['schemaVersion']==1
    assert d['revision']==sources['hf-v3-config-json']['revision']
    assert d['referenceRevision']==sources['gh-deepseek-v3-inference-model-py']['revision']
    assert all(not value for value in d['verification'].values())
    for source in sources.values():
        assert len(source['sha256'])==64 and source['bytes']>0
        if source['localPath']:
            raw=(ROOT/source['localPath']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==source['sha256'],source['id']
            checks['trackedInputHashes']+=1
        if research:
            raw=(research/source['cacheFile']).read_bytes()
            assert len(raw)==source['bytes'] and hashlib.sha256(raw).hexdigest()==source['sha256'],source['id']
            checks['upstreamInputHashes']+=1
    demo=read(DATA/'configs/DeepSeek-V3-demo-config.json')
    for f in d['configFields']:
        assert f['value']==c[f['field']] and f['source_id']=='hf-v3-config-json' and f['evidence']=='official'
        if f['demoField']: assert f['value']==demo[f['demoField']]==f['demoValue']
        checks['configFields']+=1
    dense=[i for i in range(c['num_hidden_layers']) if i<c['first_k_dense_replace'] or i%c['moe_layer_freq']!=0]
    moe=[i for i in range(c['num_hidden_layers']) if i not in dense]
    assert d['layerMapping']=={'Dense':dense,'MoE':moe}
    assert sorted(dense+moe)==list(range(61)) and d['mtp']['mtpLayer']==61
    assert d['representatives']==[{'kind':'Dense','sourceLayer':0,'displayLayer':1},{'kind':'MoE','sourceLayer':3,'displayLayer':4}]
    checks['layerMappings']=61
    H=c['hidden_size']; A=c['num_attention_heads']; Q=c['q_lora_rank']; R=c['kv_lora_rank']; D=c['qk_nope_head_dim']; P=c['qk_rope_head_dim']; V=c['v_head_dim']; E=c['n_routed_experts']; I=c['intermediate_size']; J=c['moe_intermediate_size']; S=c['n_shared_experts']; W=c['vocab_size']
    # Independent constructor/parameter formulas, not imported from the generator.
    expected={
        'self_attn.q_a_proj.weight':(Q,H),'self_attn.q_a_layernorm.weight':(Q,),
        'self_attn.q_b_proj.weight':(A*(D+P),Q),'self_attn.kv_a_proj_with_mqa.weight':(R+P,H),
        'self_attn.kv_a_layernorm.weight':(R,),'self_attn.kv_b_proj.weight':(A*(D+V),R),
        'self_attn.o_proj.weight':(H,A*V),'input_layernorm.weight':(H,),'post_attention_layernorm.weight':(H,),
        'mlp.gate_proj.weight':(I,H),'mlp.up_proj.weight':(I,H),'mlp.down_proj.weight':(H,I),
        'mlp.gate.weight':(E,H),'mlp.gate.e_score_correction_bias':(E,),
        'mlp.experts.{e}.gate_proj.weight':(J,H),'mlp.experts.{e}.up_proj.weight':(J,H),'mlp.experts.{e}.down_proj.weight':(H,J),
        'mlp.shared_experts.gate_proj.weight':(S*J,H),'mlp.shared_experts.up_proj.weight':(S*J,H),'mlp.shared_experts.down_proj.weight':(H,S*J),
        'model.embed_tokens.weight':(W,H),'model.norm.weight':(H,),'lm_head.weight':(W,H),
    }
    env={k:v['value'] for k,v in d['symbols'].items() if v['value'] is not None}
    assert len(d['matrices'])==len(expected) and len({m['id'] for m in d['matrices']})==len(expected)
    for m in d['matrices']:
        suffix=m['tensorTemplate'].removeprefix('model.layers.{i}.')
        assert tuple(m['logicalShape'])==expected[suffix]
        assert tuple(dim(x,env) for x in m['shapeExpressions'])==expected[suffix]
        count=E if m['owner']=='routed' else 1
        assert m['count']==count==dim(m['countExpression'],env)
        assert m['logicalParametersEach']==math.prod(expected[suffix])
        assert m['logicalParameters']==math.prod(expected[suffix])*count
        assert m['storedShape'] is None and m['storedDtype'] is None and m['payloadBytes'] is None
        if m['declaredScaleInIndex']: assert m['expectedScaleShape']==[(x+127)//128 for x in expected[suffix]]
        checks['matrixShapesAndCounts']+=1
    attn=H*Q+Q+A*(D+P)*Q+H*(R+P)+R+A*(D+V)*R+H*A*V
    expert=3*H*J; router=E*H+E; dense_layer=attn+2*H+3*H*I
    moe_layer=attn+2*H+router+(E+S)*expert; global_params=2*W*H+H
    independent={'attention':attn,'blockNorm':2*H,'denseFFN':3*H*I,'singleRoutedExpert':expert,
                 'allRoutedExperts':E*expert,'selectedRoutedExpertProxy':c['num_experts_per_tok']*expert,
                 'sharedExperts':S*expert,'routerIncludingCorrectionBias':router,'denseLayer':dense_layer,
                 'moeLayer':moe_layer,'global':global_params,'mainUniqueLogical':global_params+len(dense)*dense_layer+len(moe)*moe_layer}
    for key,value in independent.items():
        assert d['parameterTotals'][key]==value,(key,d['parameterTotals'][key],value)
        checks['parameterFormulaChecks']+=1
    assert independent['mainUniqueLogical']==671_026_419_200
    assert not c['tie_word_embeddings'] and d['mtp']['exactUniqueParameters'] is None
    assert d['mtp']['projectionLogicalShape']==[H,2*H] and d['mtp']['projectionLogicalParameters']==2*H*H
    for pid,p in d['proofs'].items():
        assert p['source_id'] in sources and p['level']=='reference_source' and p['line']<=p['end']
        assert sources[p['source_id']]['url'] in p['url'] and len(p['bodySha256'])==64
        if research:
            text=(research/sources[p['source_id']]['cacheFile']).read_text()
            matches={}
            for node in ast.parse(text).body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): matches[node.name]=node
                if isinstance(node,ast.ClassDef):
                    matches.update({node.name+'.'+fn.name:fn for fn in node.body if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef))})
            fn=matches[p['symbol']]
            assert (p['line'],p['end'])==(fn.lineno,fn.end_lineno)
            body='\n'.join(text.splitlines()[fn.lineno-1:fn.end_lineno])
            assert hashlib.sha256(body.encode()).hexdigest()==p['bodySha256']
            checks['sourceFunctionHashes']+=1
    assert len({s['id'] for s in d['steps']})==len(d['steps'])
    for scenario in d['scenarios']:
        scenario_env=env|scenario|{'N_tok':scenario['B']*scenario['L_q']}
        assert scenario['start_pos']+scenario['L_q']==scenario['L_kv']
        assert scenario['L_q']==1 if scenario['phase']=='decode' else scenario['start_pos']==0
        for kind,path in itertools.product(('Dense','MoE'),('naive','absorb')):
            selected=[s for s in d['steps'] if s['layerKind'] in ('both',kind) and s['branch'] in ('both',path)]
            assert selected[-1]['id']=='ffn_residual'
            state={'x':(scenario['B'],scenario['L_q'],H)}
            for s in selected:
                assert s['proof_id'] in d['proofs'] and all(sid in sources for sid in s['source_ids'])
                assert s['backend'] is None and s['backendEvidence']=='unknown' and s['duration'] is None
                input_shapes,output_shapes=check_step(s,scenario_env)
                for name,shape in zip(s['inputs'],input_shapes):
                    if name in state:
                        assert state[name]==shape,('cross-step shape mismatch',s['id'],name,state[name],shape)
                        checks['crossStepShapeChecks']+=1
                    else:
                        assert name.startswith('W_') or name=='correction_bias',('input not produced',s['id'],name)
                state.update(zip(s['outputs'],output_shapes))
                assert state.get('layer_out',state['x'])==(scenario['B'],scenario['L_q'],H)
                checks['scenarioOperatorShapeChecks']+=1
    assert d['cache']['naiveElements']==A*(D+P+V) and d['cache']['absorbElements']==R+P
    cache_cases=[]
    for scenario in d['scenarios']:
        row={'phase':scenario['phase'],'itemsizeAssumption':2,'layers':c['num_hidden_layers']}
        for path,elements in [('naive',A*(D+P+V)),('absorb',R+P)]:
            row[path+'ActiveElements']=scenario['B']*scenario['L_kv']*c['num_hidden_layers']*elements
            row[path+'ActiveBytesBF16Assumption']=2*row[path+'ActiveElements']
        cache_cases.append(row); checks['cacheFormulaCases']+=1
    summary=read(DATA/'research/v3-weight-index-summary.json')
    assert summary['sourceSha256']==sources['hf-v3-model-safetensors-index-json']['sha256']
    assert sum(summary['namesByLayer'].values())==summary['tensorNames']==91_991
    assert summary['shards']==163 and set(summary['namesByLayer'])==set(map(str,range(62)))|{'global'}
    if research:
        ix=read(research/sources['hf-v3-model-safetensors-index-json']['cacheFile'])
        assert ix['metadata']==summary['indexDeclaredMetadata']
        assert len(ix['weight_map'])==summary['tensorNames']
        assert len(set(ix['weight_map'].values()))==summary['shards']
        assert all(ix['weight_map'][k]==v for k,v in summary['representativeNameToShard'].items())
        checks['indexRepresentativeNames']=len(summary['representativeNameToShard'])
    versions=read(DATA/'versions.json'); selected=read(DATA/'research/selected-checkpoints.json')
    assert len(versions['models'])==17 and len({m['modelId'] for m in versions['models']})==17
    revisions={m['modelId']:m['revision'] for m in selected['selected']}
    for m in versions['models']:
        assert m['revision']==revisions[m['modelId']] and len(m['revision'])==40 and m['source_id'] in sources
        assert m['configVerified']==(m['modelId']==d['modelId'])
        if m['kind']=='distill': assert not m['baseModel'].startswith('deepseek-ai/')
        checks['versionRelationships']+=1
    assert all(m['kind']=='inference_mode' for m in versions['modes'])
    export_result=[]
    if check_exports:
        packet=read(OUT/'tables.json')
        assert packet['sourceSha256']==hashlib.sha256((DATA/'research/v3-round1.json').read_bytes()).hexdigest()
        for entry in packet['tables']:
            p=OUT/entry['filename']; raw=p.read_bytes(); assert raw.startswith(b'\xef\xbb\xbf')
            with p.open(newline='',encoding='utf-8-sig') as handle: rows=list(csv.reader(handle))
            assert rows[0]==entry['headers'] and len(rows)==len(entry['rows'])+1
            for number,(actual,expected_row) in enumerate(zip(rows[1:],entry['rows']),2):
                assert len(actual)==len(expected_row)
                for header,value,expected in zip(entry['headers'],actual,expected_row):
                    if type(expected) in (int,float):
                        # CSV is textual; 0.000001 and 1e-06 are the same typed source number.
                        assert Decimal(value)==Decimal(str(expected)),('CSV numeric mismatch',p,number,header)
                    else:
                        expected_text='' if expected is None else str(expected).lower() if isinstance(expected,bool) else expected
                        assert value==expected_text,('CSV text mismatch',p,number,header)
            if entry['filename']=='v3-representative-operators.csv':
                expected_keys=[(r['sourceLayer'],r['displayLayer'],r['kind'],sc['phase'],branch,s['id'])
                               for r in d['representatives'] for sc in d['scenarios'] for branch in ('naive','absorb')
                               for s in d['steps'] if s['layerKind'] in ('both',r['kind']) and s['branch'] in ('both',branch)]
                assert [tuple(row[1:7]) for row in entry['rows']]==expected_keys
                for row in entry['rows']:
                    s=next(s for s in d['steps'] if s['id']==row[6]); proof=d['proofs'][s['proof_id']]
                    assert row[13]==s['relation'] and row[14]==proof['url'] and row[19]=='unknown' and row[21] is None
            elif entry['filename']=='v3-parameters.csv':
                assert {row[1]:row[2] for row in entry['rows'] if row[0]=='summary'}=={k:v for k,v in d['parameterTotals'].items() if isinstance(v,int)}
                assert {row[1]:row[8] for row in entry['rows'] if row[0]=='matrix'}=={m['id']:m['logicalParameters'] for m in d['matrices']}
            else: raise AssertionError(f'Unexpected export: {entry["filename"]}')
            checks['csvDataRows']+=len(rows)-1
            export_result.append({'path':str(p.relative_to(ROOT)),'rows':len(rows)-1,'columns':len(rows[0]),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    report={'status':'passed','date':'2026-10-01','modelRevision':d['revision'],'referenceRevision':d['referenceRevision'],
            'checks':dict(checks),'mainUniqueLogical':independent['mainUniqueLogical'],
            'sourceCodeReverified':bool(research),'exportsChecked':check_exports,
            'mlaAlgebra':algebra_check(),'routingArithmetic':routing_check(c),'cacheCases':cache_cases,'exports':export_result,
            'limits':['静态配置/shape/源码哈希/索引名称与独立数学复算','未执行上游模型代码或加载 checkpoint','未读取权重文件头','未运行 GPU/NPU 或性能实验','当前验证器仅覆盖首轮研究契约；P5a 后另接网站验证分派']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'checks':report['checks'],'mainUniqueLogical':independent['mainUniqueLogical'],'mlaAlgebra':report['mlaAlgebra']},ensure_ascii=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root',type=Path)
    parser.add_argument('--skip-exports',action='store_true',help='Validate research before CSV export; completion remains exportsChecked=false')
    args=parser.parse_args()
    validate(args.research_root,not args.skip_exports)
