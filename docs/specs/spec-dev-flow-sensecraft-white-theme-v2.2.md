# SenseCraft Local Voice 界面全局白底风格与视觉简化规范 (v2.2)

> **目标**：
> 1. **对齐 SenseCraft 白底设计规范**：将 Dashboard 从深灰暗色风格（`#0f1419` / `#1a1f2e`）完整迁移至 SenseCraft 官方浅色设计体系（`sensecraft-ui` 策略 c），采用统一白底（`--sc-color-bg-layout: #f8fafc`、`--sc-color-bg-container: #ffffff`、`--sc-color-border: #e2e8f0`）、SenseCraft 品牌绿（`#8CC020`）、深色易读前景色（`--on-accent: #111827`）、图表专用高对比绿（`--chart-primary: #357f11`）、标准四态徽标及层级清晰的排版系统。
> 2. **界面视觉与信息层级简化**：消除沉重深色块与杂乱描边，重塑顶部工具栏、对话气泡、侧边监控卡片与设置抽屉，实现通透、低认知负荷的现代控制台体验。
> 3. **无障碍与工程契约合规**：严格遵守 WCAG AA 标准（正常文本 ≥ 4.5:1，控件边界与数据表达图形 ≥ 3:1，聚焦指示器 ≥ 3:1）；100% 保持既有 169 个 DOM 节点 ID；完整保持 Translator V7.1-V7.6 状态机契约与 MCP 红点联动。
>
> **遵循准则**：`dev-flow`、`taste`（变更有界、按使用者心智模型建模、真实优先、通用优先于 Hack、零破坏性）。

---

## 1. 架构目标与视觉重塑

### 1.1 现状分析与用户诉求
- **现状**：当前界面为深灰暗色操作界面（背景 `#0f1419`，面板 `#1a1f2e`，边框 `#252b3a`），暗色调沉重，与 Seeed Studio / SenseCraft 其它系列方案（白底、通透、简洁）风格脱节。
- **用户明确要求**：“帮我优化整个界面的，整个页面可以简化一点。然后应该是用 3 克以往那种白底那种风格嘛，现在还是灰底的。，让 claude 完整优化下”。
- **“3 克”指向**：用户语音输入将 “SenseCraft / Seeed” 识别为“3 克”，指代 Seeed SenseCraft 官方浅色白底控制台风格（`@suharvest/ui-kit` / `tokens.css`）。

### 1.2 SenseCraft 浅色设计系统 Token 映射与精确对比度收敛

严格对齐 `@suharvest/ui-kit` 设计规范并强化 WCAG 2.1 AA 合规要求（正常文本 ≥ 4.5:1，数据表达图形/控件边界/焦点环 ≥ 3:1）：

| Token 变量 | 浅色基准值 | 实测对比度 (vs 白) | 实测对比度 (vs `#f1f5f9`) | 页面语义与应用位置 |
|---|---|---|---|---|
| `--sc-color-bg-layout` | `#f8fafc` | - | - | 全局外层底色（`html, body`、主区域背景） |
| `--sc-color-bg-container` | `#ffffff` | - | - | 容器、侧边卡片、工具栏、抽屉与对话气泡主体纯白底 |
| `--sc-color-panel-2` | `#f1f5f9` | - | - | 抽屉内嵌块、状态药丸底色、次级面板浅灰背景 |
| `--sc-color-border` | `#e2e8f0` | 1.17:1 | 1.08:1 | 装饰性分割线、卡片边框、工具栏底边（非控件边界） |
| `--control-border` | `#64748b` | **4.76:1** | **4.37:1** | **表单控件边框**（input, select, textarea，严格满足 ≥ 3:1 控件边界要求） |
| `--sc-color-primary` | `#8CC020` | 1.40:1 | 1.29:1 | SenseCraft 品牌主色（主按钮底色、激活指示条） |
| `--sc-color-primary-hover` | `#7aad1c` | 1.63:1 | 1.50:1 | 主按钮 Hover 状态 |
| `--sc-color-primary-active` | `#689a18` | 2.05:1 | 1.88:1 | 主按钮 Active 点击态 |
| `--sc-color-primary-dark` | `#5B7D15` | **4.72:1** | **4.33:1** | 高对比绿（强调文字、重音边框） |
| `--on-accent` | `#111827` | **8.17:1** (vs `#8CC020`) | - | **主按钮文本色**（禁止使用纯白 `#ffffff`，纯白对比度仅 2.17:1 严重不合规） |
| `--focus-ring` | `#357f11` | **5.01:1** | **4.57:1** | **焦点指示器**（outline 或 box-shadow，严格满足 ≥ 3:1） |
| `--chart-primary` | `#357f11` | **5.01:1** | **4.57:1** | **Canvas 图表主色**（折线、说话中 VAD 电平柱，满足非文本 ≥ 3:1） |
| `--chart-idle` | `#64748b` | **4.76:1** | **4.37:1** | **Canvas 空闲 RMS 电平柱**（静音/低电平柱，满足非文本 ≥ 3:1） |
| `--chart-threshold` | `#475569` | **7.58:1** | **6.92:1** | **Canvas VAD 阈值参考线**（虚线样式，满足非文本 ≥ 3:1） |
| `--chart-grid` | `#e2e8f0` | 1.17:1 | 1.08:1 | 装饰性图表参考水平基线（非数据表达） |
| `--sc-color-text` | `#1f2937` | **12.63:1** | **11.60:1** | 主文字颜色（极高清晰度） |
| `--sc-color-text-secondary`| `#475569` | **7.58:1** | **6.92:1** | **次要文字、标签说明**（在白底与 `#f1f5f9` 面板上均严格 ≥ 4.5:1） |
| `--placeholder` | `#64748b` | **4.76:1** | **4.37:1** | **输入占位符文本**（严格遵守 WCAG AA 4.5:1 要求，禁止浅灰不可读） |
| `--success` | `#166534` | **7.13:1** | **6.51:1** | **成功语义文本/图标**（`ev-on_assistant_done`, `dot.open`） |
| `--warn` | `#92400e` | **7.09:1** | **6.47:1** | **警告语义文本/图标**（`ev-on_user_stop_intent`, 待重启提示） |
| `--error` | `#b91c1c` | **6.47:1** | **5.91:1** | **错误语义文本/图标**（`ev-on_error`, `dot.closed`） |
| `--state-idle` | `#475569` | **7.58:1** | **6.92:1** | 空闲状态文本/指示灯 |
| `--state-listening` | `#0369a1` | **5.32:1** | **4.85:1** | 聆听中状态文本/指示灯 |
| `--state-thinking` | `#6d28d9` | **6.47:1** | **5.90:1** | 思考中状态文本/指示灯 |
| `--state-speaking` | `#166534` | **7.13:1** | **6.51:1** | 发音中状态文本/指示灯 |
| `--state-barged` | `#92400e` | **7.09:1** | **6.47:1** | 打断状态文本/指示灯 |

### 1.3 核心区域视觉简化方案

1. **顶部工具栏 (`#toolbar`)**：
   - 背景设为纯白 `#ffffff`，底边为细分割线 `1px solid var(--sc-color-border)`，微阴影 `0 1px 2px rgba(0,0,0,0.03)`；
   - 状态指示灯 `#statePill`：白底圆角药丸，细边框，状态点在 IDLE 时使用柔和灰（`#475569`），在 LISTENING 时使用深蓝（`#0369a1`），SPEAKING 时使用深绿（`#166534`），呼吸动画柔和自然；
   - 模式选择器 `#modeBtn` 与设置按钮 `#modeSettingsBtn`：浅灰底色悬浮提亮，视觉轻盈；
   - 桌面端高度严格维持 `<= 54px` 单行，全视口（1920px 至 375px）零横向溢出。
2. **对话面板 (`#tab-chat`)**：
   - 助手回复气泡：纯白卡片 `#ffffff`，搭配柔和微阴影 `0 1px 3px rgba(0,0,0,0.04)`，细边框 `1px solid #e2e8f0`，深灰主文字 `#1f2937`；代码块与引用采用 `#f8fafc` 浅灰衬底；
   - 用户提问气泡：采用 SenseCraft 品牌轻绿色调衬底（`background: #effce8; border: 1px solid #cde69c; color: #1f2937;`），与助手气泡形成清晰优雅对比；
   - 空状态引导文案：采用 Slate 600（`#475569`，对比度 7.58:1）提示。
3. **右侧监控与状态卡片 (`#right`)**：
   - 所有卡片统一采用纯白卡片（`background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);`）；
   - 卡片标题：深灰加粗 `#1f2937`，辅助文字 `#475569`；
   - 音量波形与延迟图表（Canvas）：
     - 空白装饰基线采用浅灰边框色 `#e2e8f0`；
     - 延迟趋势折线采用高对比深绿（`--chart-primary: #357f11`，对比度 5.01:1 ≥ 3:1）；
     - 麦克风有效说话电平柱（≥ 阈值）采用高对比深绿 `#357f11`（对比度 5.01:1 ≥ 3:1）；
     - 麦克风低电平/静音柱（< 阈值）采用中灰 `#64748b`（对比度 4.76:1 ≥ 3:1，表达实测能量数据）；
     - 麦克风阈值参考线采用深灰虚线 `#475569`（对比度 7.58:1 ≥ 3:1，线宽 1.5px，虚线 `[3, 3]`）。
4. **设置抽屉 (`#modeSettingsPanel`) 与 MCP 模态框**：
   - 抽屉背景改为纯白 `#ffffff`，顶部 Header 背景 `#f8fafc`；
   - Tab 导航栏：白底背景，非激活项文字 `#475569`，激活项为深灰加粗 `#1f2937`，底部 2px 指示条为 SenseCraft 品牌绿 `#8CC020`；
   - 输入框、下拉框与文本域：纯白背景 `#ffffff`，控件边框 `1px solid var(--control-border)`（`#64748b`，对比度 4.76:1 ≥ 3:1），聚焦时呈现高对比深绿外圈指示器（`outline: 2px solid var(--focus-ring); outline-offset: 2px;`，对比度 5.01:1 ≥ 3:1）；
   - 主操作按钮（保存、发送）：背景 `#8CC020`，文本色 `#111827`（对比度 8.17:1），字重 600。

---

## 2. 变更范围与硬性约束清单 (Manifest & Contracts)

### 2.1 变更文件白名单与锚点定义 (File Manifest & Anchors)
严格限定在以下 3 个静态前端文件，**严禁改动任何 Python 后端代码或 API 路由**：

1. `agent/ovs_agent/plugins/static/dashboard.html`：
   - 顶部工具栏与状态药丸：`dashboard.html:10-52`
   - 对话流与多 Tab 容器：`dashboard.html:53-79`
   - 右侧监控卡片与 Canvas：`dashboard.html:80-148`
   - 设置抽屉、Tab 导航与配置项：`dashboard.html:149-316`
   - 声音克隆浮层：`dashboard.html:317-340`
   - MCP 服务管理模态框：`dashboard.html:341-396`

2. `agent/ovs_agent/plugins/static/dashboard.css`：
   - 官方 Token 与主题变量系统：`dashboard.css:1-50`
   - 页面全局基础样式（body, selection）：`dashboard.css:54-65`
   - 顶部工具栏、状态药丸与断点：`dashboard.css:66-160`
   - 按钮、输入控件边界与焦点环：`dashboard.css:161-241`
   - 消息气泡视觉简化：`dashboard.css:242-260`
   - 事件流与历史流语义色彩：`dashboard.css:261-287`
   - 侧边栏卡片与 Canvas 图表容器：`dashboard.css:288-380`
   - 设置抽屉、Tab 切换与弹窗样式：`dashboard.css:600-846`

3. `agent/ovs_agent/plugins/static/dashboard.js`：
   - 设置抽屉 Tab 状态机与意图保护：`dashboard.js:198-315`
   - Canvas RMS 柱状图与参考线绘制：`dashboard.js:1215-1241`
   - Canvas Sparkline 延迟折线绘制：`dashboard.js:1263-1299`
   - MCP 待生效红点徽标联动：`dashboard.js:2112-2125`

### 2.2 核心契约与 169 个受保护 ID 零破坏契约
- **169 个 DOM 节点 ID 保护**：依据 `.cache/review/html_ids_inventory.json` 清单，`dashboard.html` 中全部 169 个唯一 ID 必须 100% 完整保留在 DOM 树中，数量严格等于 1。禁止删除、重命名或遗漏任何一个 ID。
- **配置与设置控件保留**：`setPipelineMode`, `setSleepTimeout`, `setStopWords`, `setExitWords`, `setWakeReply`, `setExitReply`, `btnAgentSettingsSave`, `btnAgentSettingsReload`, `wwPhrases`, `btnWwSave`, `btnWwValidate`, `btnWwTone`, `wwBadge`, `wwHint`, `btnRestartAgent`, `btnRestartAgentMenu`, `modeRestartTag` 等 100% 保持。
- **核心容器与控制 ID**：`msMode`, `msClose`, `btnMcpRefresh`, `setTranslatorTgtLang`, `btnMenu` 等 100% 保持。
- **Tab 与事件过滤结构**：`#msTabs`, `#msTabPendingBadge-mcp`, `#filters`（以及 7 个 `data-f` input）100% 保持。
- **交互状态机与无障碍**：严禁破坏已建立的 `desiredTab` 意图状态机、Roving Tabindex 与 WAI-ARIA 键盘无障碍流转。

---

## 3. 样式重构规则 (`dashboard.css`)

### 3.1 CSS 变量系统设计规范
```css
:root {
  /* SenseCraft 官方设计 Token */
  --sc-color-primary: #8CC020;
  --sc-color-primary-hover: #7aad1c;
  --sc-color-primary-active: #689a18;
  --sc-color-primary-dark: #5B7D15;
  --sc-color-bg-container: #ffffff;
  --sc-color-bg-layout: #f8fafc;
  --sc-color-border: #e2e8f0;
  --sc-color-text: #1f2937;
  --sc-color-text-secondary: #475569; /* Slate 600，白底 7.58:1，#f1f5f9 上 6.92:1 >= 4.5:1 */

  /* 仪表盘核心映射变量 */
  --bg: var(--sc-color-bg-layout);
  --panel: var(--sc-color-bg-container);
  --panel-2: #f1f5f9;
  --border: var(--sc-color-border);
  --control-border: #64748b; /* 表单控件边框，4.76:1 vs 白，>= 3:1 */
  --text: var(--sc-color-text);
  --muted: var(--sc-color-text-secondary);
  --muted-dim: #64748b;
  --accent: var(--sc-color-primary);
  --accent-hover: var(--sc-color-primary-hover);
  --accent-active: var(--sc-color-primary-active);
  --accent-dark: var(--sc-color-primary-dark);
  --on-accent: #111827; /* 严格保证在 #8CC020 上 8.17:1 对比度 */
  --focus-ring: #357f11; /* 焦点指示器，在白底上 5.01:1，>= 3:1 */
  --chart-primary: #357f11; /* 图表与有效 VAD 柱状图主色，在白底上 5.01:1，>= 3:1 */
  --chart-idle: #64748b; /* 空闲 RMS 电平柱，在白底上 4.76:1，>= 3:1 */
  --chart-threshold: #475569; /* VAD 阈值参考线，在白底上 7.58:1，>= 3:1 */
  --chart-grid: #e2e8f0;

  /* 语义化文本安全色彩（浅底高对比，严格 >= 4.5:1） */
  --error: #b91c1c;   /* Red 700: 6.47:1 vs 白, 5.91:1 vs #f1f5f9 */
  --success: #166534; /* Green 800: 7.13:1 vs 白, 6.51:1 vs #f1f5f9 */
  --warn: #92400e;    /* Amber 800: 7.09:1 vs 白, 6.47:1 vs #f1f5f9 */

  --state-idle: #475569;      /* 7.58:1 vs 白 */
  --state-listening: #0369a1; /* 5.32:1 vs 白 */
  --state-thinking: #6d28d9;  /* 6.47:1 vs 白 */
  --state-speaking: #166534;  /* 7.13:1 vs 白 */
  --state-barged: #92400e;    /* 7.09:1 vs 白 */

  --radius: 8px;
  --radius-sm: 4px;
  --radius-lg: 12px;
  --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
  --accent-soft: #effce8;
  --accent-line: #cde69c;
}
```

### 3.2 控件与无障碍聚焦强化
- **表单输入框、下拉框与文本域**：
  ```css
  input, select, textarea {
    background: #ffffff;
    border: 1px solid var(--control-border); /* 4.76:1 满足 3:1 控件边界要求 */
    color: var(--text);
  }
  input::placeholder, textarea::placeholder {
    color: #64748b; /* 4.76:1 严格满足文本 4.5:1 要求 */
    opacity: 1;
  }
  input:focus-visible, select:focus-visible, textarea:focus-visible, button:focus-visible, .tab:focus-visible, .ms-tab:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  ```
- **主操作按钮（Primary）**：
  ```css
  .ctl-btn.primary, #btnSend {
    background: var(--accent);
    color: var(--on-accent); /* #111827，对比度 8.17:1 */
    font-weight: 600;
  }
  ```

---

## 4. 图表与动态绘制浅色对齐 (`dashboard.js`)

在 `dashboard.js` 中将硬编码的 Canvas 绘制色值对齐浅色主题，确保全部数据表达图形与参考线满足 WCAG 2.1 1.4.11 ≥ 3:1 非文本对比度要求：
1. **波形图 `drawRms()` (`dashboard.js:1215-1241`)**：
   - 空闲/静音电平柱（< 阈值）：采用中灰 `#64748b`（对比度 4.76:1 ≥ 3:1，表达实测能量数据，严禁使用淡灰 `#cbd5e1` 导致低于 3:1）；
   - 说话有效电平柱（≥ 阈值）：采用高对比深绿 `#357f11`（对比度 5.01:1 ≥ 3:1，严禁浅色 `#8CC020` 导致低于 3:1）；
   - 阈值参考线：采用高对比深灰虚线 `ctx.strokeStyle = "#475569"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);`（对比度 7.58:1 ≥ 3:1，严禁使用透明度衰减色导致实际对比度不足 3:1）。
2. **延迟趋势图 `drawSpark()` (`dashboard.js:1263-1299`)**：
   - 空数据装饰基线：采用浅灰边框色 `#e2e8f0`（纯装饰性指示）；
   - 活跃折线颜色：采用高对比深绿 `#357f11`（对比度 5.01:1 ≥ 3:1）。

---

## 5. 验收与验证矩阵 (Verification Matrix)

| 编号 | 检查项 | 验证方式与断言标准 |
|---|---|---|
| **V1** | **全部 169 个 DOM 核心 ID 完整保护** | 读取 `.cache/review/html_ids_inventory.json`，在浏览器 DOM 中逐个断言：全部 169 个 ID **均存在且数量严格等于 1**。 |
| **V2** | **全界面白底浅色主题视觉表现** | 检查 `body`、`#toolbar`、`.card`、`.mode-settings-panel` 背景色：必须全部为浅色/白底（`#ffffff`、`#f8fafc`），严禁残留 `#0f1419` 或 `#1a1f2e` 等深黑底色。 |
| **V3** | **严格 WCAG AA 对比度与控件/图形可识别性合规** | 1. 主文字 `#1f2937` vs 白色背景对比度 12.63:1（≥ 4.5:1）；<br>2. 提示与次级文字 `#475569` 在白底（7.58:1）与面板 `#f1f5f9`（6.92:1）上对比度均严格 ≥ 4.5:1；<br>3. 输入框占位符 `#64748b` 对比度 4.76:1（严格遵守 WCAG 文本对比度 ≥ 4.5:1 标准）；<br>4. 主按钮（`#8CC020` 底色）上的文本色必须为 `#111827`，对比度达到 8.17:1（严禁白字）；<br>5. 交互控件输入边界（`--control-border: #64748b`）对比度 4.76:1（≥ 3:1）；<br>6. 聚焦指示器（`--focus-ring: #357f11`）对比度 5.01:1（≥ 3:1）；<br>7. Canvas 有效 VAD 柱状图主色采用 `#357f11`，对比度 5.01:1（≥ 3:1）；<br>8. Canvas 空闲 RMS 电平柱采用 `#64748b`，对比度 4.76:1（≥ 3:1）；<br>9. Canvas 阈值参考线采用 `#475569` 虚线，对比度 7.58:1（≥ 3:1）；<br>10. Canvas 延迟折线采用 `#357f11`，对比度 5.01:1（≥ 3:1）；<br>11. 语义化文本颜色（`--success: #166534`, `--warn: #92400e`, `--error: #b91c1c`）在白底与 `#f1f5f9` 上对比度均严格 ≥ 4.5:1。 |
| **V4** | **顶部工具栏防遮挡与全视口自适应保持** | 验证在 1920/1440/1280/1024/900/899/768/375px 八个视口下单行高度与换行表现，文档横向溢出严格为 0px，状态指示灯与设置按钮完全可见，桌面端工具栏高度 `<= 54px`。 |
| **V5** | **Tab 状态机与 Roving Tabindex 功能完好** | 切换 4 个 Tab、键盘方向键轮转、刷新持久化记忆功能 100% 保持；任何时刻仅 1 个可见 Tab `aria-selected="true"` 且 `tabindex="0"`。 |
| **V6** | **图表动态绘制适配** | 延迟折线和波形图在白底背景下清晰可见，无反白或隐形现象。 |
| **V7** | **Translator 后端生命周期与意图模型闭环 (V7.1-V7.6 回归)** | 1. **V7.1 初始探针恢复**：初始 `desiredTab = 'translator'`，探针成功返回 available 时，自动平滑激活并呈现 translator Tab；<br>2. **V7.2 用户意图绝对保护**：探针挂起期间用户点击 `voice` Tab，探针成功后断言保持停留在 `voice`，不被篡改强跳；<br>3. **V7.3 探针失败保持意图**：探针失败时断言 `localStorage` 保持原本的 `desiredTab`，再次打开抽屉重试成功后恢复呈现；<br>4. **V7.4 noop 保护非翻译配置**：启动存储为 `voice` 或 `mcp` 时探针返回 `noop`，断言其存储值与激活状态完全保持，绝不被篡改覆写为 `prompt`；<br>5. **V7.5 刷新失败降级与重试恢复**：在 translator Tab 中点击刷新发生失败，断言 `#msTab-translator` 隐藏，界面平滑降级至 `prompt` 面板且焦点安全转移至 prompt Tab 或关闭按钮，再次探测成功后平滑恢复展示 translator 面板；<br>6. **V7.6 noop 降级与焦点安全转移**：在 translator Tab 激活状态下刷新返回 noop，断言 `#msTab-translator` 隐藏，立即激活 prompt Tab（`aria-selected=true`, `tabindex=0`），prompt 面板可见，translator 面板隐藏，焦点安全转移至 prompt Tab 按钮或抽屉关闭按钮，且持久化 intent 被重置为 `'prompt'`。 |
| **V8** | **MCP 待生效红点联动** | 设置 `mcpState.pending = true` 时，`#msTabPendingBadge-mcp` 显示红点；重启或重置为 false 时，红点即刻隐藏。 |
| **V9** | **控制台健康监控** | 页面加载与全交互流程中 `pageerror` 与 `console.error` 计数严格为 0。 |
| **V10** | **后端 API 回归通过** | 在 `agent/` 目录下执行 `uv run pytest tests/test_dashboard_api.py tests/test_mcp_dashboard_api.py -v`，所收集的全部测试用例 100% 通过。 |
