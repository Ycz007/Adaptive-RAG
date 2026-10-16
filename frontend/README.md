# Adaptive RAG 客服工作台

Vue 3 + TypeScript + Vite + Pinia + Vue Router。本目录是独立前端，不修改后端行为。

## 启动

需要 Node.js 20.19+ 或 22.12+。在仓库根目录启动后端：

```powershell
uvicorn app.asgi:app --reload --port 8000
```

新终端进入本目录：

```powershell
cd frontend
npm install
npm run dev
```

打开终端打印的地址，默认 http://127.0.0.1:5173。Vite 将 `/api` 代理到 `http://127.0.0.1:8000`。后端位于其他主机时可设置 `API_PROXY_TARGET` 再启动 Vite。

## 功能

- 工作台：普通/流式回答、最终答案替换、引用、回答详情、复制、停止接收。
- 业务确认：收到 pending_action 后核对参数并确认，使用原问题、原 session_id 与 confirm_action_id 发起新请求；中断时标记结果待核实，不提供重复执行按钮。
- 会话：浏览器本地存储、标题与消息搜索、8 条分页、继续会话、删除二次确认。无服务端会话列表接口，因此删除仅影响本地数据。
- 设置：API 基础地址、用户 ID、流式开关、带超时的健康检查、实时校验。
- 侧栏收起、移动端导航抽屉、移动端会话卡片、Toast、骨架屏、键盘焦点与弹窗焦点约束。

新会话使用当前设置，已有会话固定原 API 地址和 user_id。后端生成提示词目前没有回注历史消息，因此保留同一个 session_id 不代表模型拥有完整多轮记忆。

默认不会填充假消息或伪造指标。后端未启动时页面保持可用，连接和发送会显示真实错误。订单示例 ORD10001 / ORD10003 需要后端已初始化演示数据并且 user_id 有权查询；普通启动不保证存在种子数据。

## 验证与部署

```powershell
npm test
npm run build
```

`npm run build` 先运行 TypeScript 检查，再输出 `dist/`。将 dist 交给静态服务器托管，生产服务器须把 `/api` 反向代理到 FastAPI；Vite 的开发代理不会打包进产物。使用 Hash 路由，无需为页面路由单独配置 SPA 回退。若设置跨域完整 API 地址，后端需要配置允许的 CORS 来源。

Inter 通过 Google Fonts 加载；无法访问时自动使用系统字体。消息按纯文本渲染，不执行 HTML。

注意：后端确认 ID 与原始提案的持久化绑定仍需完善；前端确认界面不能替代服务端授权检查。SSE token 是待检查的临时内容，以最后的 answer 为准。停止接收无法保证服务端停止执行业务操作。
