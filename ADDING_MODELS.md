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
- `type`: 显示真实注意力类型；已知 KDA/MLA/GQA/Vision 使用对应颜色，其他类型采用中性色。
- `parameters`: 本层总逻辑参数；`activeLinearParameters`: 同口径激活线性权重代理；`payloadBytes`: 实际权重载荷。未知仍为 `null`。
- `modules[]`: 含 `id/title/parameters/count/selectedCount/representative/matrices`。专家多时提供一个模板并写明实际数量；不要复制数万模型对象。
- `matrices[]`: `tensor_template/stored_dtype/stored_shape/logical_shape/logical_parameters_each/count`。量化元数据的逻辑参数计数为 0，但不能丢掉存储记录。
- `cache`: 对该模型适用的 MLA/KDA/视觉缓存假设；不要把 Kimi 的 576 元素缓存直接复制给其他架构。
- `config`: 已核对的隐藏宽、专家数、top-k、共享数、上下文、精度。
- `scope/source`: 核验深度与来源。只有配置时可提供层型，但不填未经审计的矩阵、载荷与参数。

同一个节点会同时驱动 3D 选层、前后导航、检查面板及二维回退。未提供架构数据时自动显示不可用说明。

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
