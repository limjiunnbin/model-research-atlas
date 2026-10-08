# 新增模型与家族

页面不按模型名称写死内容。新增资料修改 JSON 和素材即可；既有页面读取同一接口。

## 增加一个家族

1. 创建 `data/families/<family-id>/`，参照现有家族写 `family.json` 与 `report.json`。
2. 在 `data/catalog.json` 的 `families` 添加 `id/name/publisher/description/path/updated/modelCount/figureCount`。
3. 家族文件包含 `overview`、`historyNote`、`auditNote`、`limitations`、`technology`、`scenarios`、`models`、`figures`、`downloads` 等内容。研究主页、时间线说明、技术标签都从这里读取，不需要修改页面。
4. 素材放 `dist/assets/<family-id>/`，所有路径相对 `dist/`；数据路径为 `data/...`。完整报告章节可含可信本地 HTML，但不要放外部脚本、事件属性或未经清洗的第三方 HTML。
5. 运行构建，检查新增家族、各版详情、比较与下载。

## 增加一个版本

在家族 `models` 中添加唯一 `id`，保留准确名称。使用现有 `facts` 键；不适用或未知为 `null`，对应 `evidence: "unknown"`。每个已知事实附可核对 `source`。`releaseDate` 不确定时保留 `null`。没有权重审计时 `auditPath` 为 `null`，不要把其他版本的载荷当成本版测量值。

```json
{
  "layers": {
    "value": null,
    "evidence": "unknown",
    "source": null
  }
}
```

`summary` 是简短结论；`tags` 控制应用筛选，`branch` 控制分支筛选。`sections` 对应本家族报告的章节 ID。技术主题的 `section` 同样引用该 ID。

## 添加 3D / 二维结构

设置 `architecturePath` 指向独立 JSON。渲染器读取 `nodes[]`，无需修改页面：

- `group`: `decoder` / `vision` / 其他组件名称。
- `number`: 层从 1 开始；全局组件为 0。`id` 必须唯一。
- `type`: 显示真实注意力类型；KDA/MLA/DSA/CSA/HCA/SWA/GQA/Vision 使用对应颜色，其他类型采用中性色。
- `parameters`: 本层总逻辑参数；`activeLinearParameters`: 同口径激活线性权重代理；`payloadBytes`: 实际权重载荷。未知仍为 `null`。
- `modules[]`: 含 `id/title/parameters/count/selectedCount/representative/matrices`。专家多时提供一个模板并写明实际数量；不要复制数万模型对象。
- `matrices[]`: `tensor_template/stored_dtype/stored_shape/logical_shape/logical_parameters_each/count`。量化元数据的逻辑参数计数为 0，但不能丢掉存储记录。
- `cache`: 字符串键到说明的映射，仅提供该模型适用的缓存；节点用 `cacheKey` 或 type 的小写键选择。MLA/DSA、CSA/HCA/SWA、GQA、视觉状态分列，不复制相邻架构假设。
- `config`: 已核对的隐藏宽、专家数、top-k、共享数、上下文、精度。
- `scope/source`: 核验深度与来源。只有配置时可提供层型，但不填未经审计的矩阵、载荷与参数。

同一个节点会同时驱动 3D 选层、前后导航、检查面板及二维回退。未提供架构数据时自动显示不可用说明。

存储未审计时 `stored_dtype/stored_shape/payloadBytes` 为 `null`，逻辑矩阵可在固定配置加参考参数声明的推导证据下提供。`parameterScope` 可说明 MTP、共享别名与非训练表的特殊口径；`implementationTopic` 可在节点或模块上声明，优先于默认专题映射。模型的 `auditPath` 可以引用既有 groups 审计格式或 DeepSeek 全分片 header 摘要，详情消费者分别显示两者，不伪造共同字段。`documentCenterUrl` 可将家族侧栏下载入口指向该家族的独立研究包。

## 核验

```sh
python3 scripts/build.py
node --check dist/app.js
node --check dist/structure.js
```

构建会拒绝不存在的文件、重复 ID、空值冒充已知证据、层号不连续、视觉层数不符、审计层模块参数之和不一致。每次更新还应在浏览器检查一条完整的“家族 → 版本 → 层 → 模块 → 矩阵”路径、窄屏排版、搜索与下载。预览不会自动发布。


## 添加实现与硬件研究

家族可选 `hardwarePath` 指向独立 JSON（参见 schema 的 hardware）。modules 必须声明适用 model IDs、数据流、源码观察、硬件解释、边界与 sources 引用；platforms 按具体代际/型号分列。optimizations 保存证据类别、适用范围、指标和风险；尚未 profiling 的项不能写成已验证瓶颈。

源码用固定 commit URL 与 SHA-256，滚动网页用访问日；版本漂移单独指出。`implementationLinks` 将模型与专题绑定；3D 按 KDA/MLA/vision 及 routed/latent/residual 模块选择专题，新增架构需扩展映射。`downloads` 中报告和CSV路径由数据指定，不硬编码家族。运行 build 会从研究JSON重新生成这两份附件。
# 可选逐层计算数据

家族可设置 `computePath` 指向计算 JSON，模型可设置同名字段指向逐层页面。每个真实组件必须有唯一 `id`、1-based `layer`（全局为 0）、`template` 和参数数目。共享模板需保留完整组件映射；专家用 `{e}`，层用 `{i}`，不得混用。

每一步记录输入/输出/中间及缓存形状、阶段、精度、权重、参考实现和四后端证据。设备接口必须附固定 revision、函数、行号和分支条件；未知接口显式标记。不得将 AST 调用集合当作执行顺序，或将模块调用误标为单步设备内核。

当前 `scripts/validate_compute.py` 是 Kimi 专用审计验证器；新增家族应提供对应验证器，并调整 `build.py` 的计算验证分派。

DeepSeek 使用 `scripts/validate_deepseek_atlas.py` 与 `validate_deepseek_exports.py`；build 根据 family ID 分派，未注册验证器的 compute 家族直接报错。其可选 `phaseOptions/pathOptions` 驱动阶段与参考分支筛选，`weightEvidence` 说明每版真实核验深度。组件的 `sourceIndex` 是模板替换用的源码索引，不总是 display layer−1；MTP 或全局组件必须显式给出。参考 API 的 `level/condition/proofs` 表达模块与分派上下文，不把调用集合标为 kernel 顺序。hardware.versionNotes 提供适用 `models` 与说明，模块必须有可读 evidence 与 refs；新实现专题应在浏览器验证当前版本过滤与 3D 跳转。

DSpark 使用独立 `group=dspark`，三个组件的 `sourceIndex=0/1/2` 对应实际 `mtp.*` 存储；不能由 `num_nextn_predict_layers=1` 推导只有一个组件，也不能套用普通 MTP 的 eh_proj/enorm/hnorm。每个版本的 stored shape/dtype/payload 必须绑定自己的完整 header，并保留 FP4 逻辑维度与 I8 容器、scale、非训练表和共享存储副本的区别。参数声明、prefill 的 context cache 写入、decode 的 draft backbone/Markov/confidence 与服务 verification 是不同范围，模板步骤不等于每阶段都执行全部矩阵。
