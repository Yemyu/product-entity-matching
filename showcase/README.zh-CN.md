# 双语静态研究展示

[English](README.md) | **简体中文**

打开[中文页面](zh-CN/index.html)或[英文页面](en/index.html)。每种语言均有完整的
概览、数据与方法、实验结果、复现四页。语言按钮跳到当前页的另一语言版本；
启用 JavaScript 时还保留段落锚点、AP/F1 选择、教学场景与当前展开的模型／配置面板。关闭 JavaScript
仍能阅读全部正文、图表和表格，两张指标图与三个教学场景都会显示。交互
脚本提供场景和指标切换、段落定位与命令复制。数据与方法页的“模型说明”和
“最终配置”在按钮下方展开可阅读面板，再次点击收起；焦点位于面板内时也可
按 Escape 收起。未启用 JavaScript 时，两块正文直接显示，按钮跳转到对应标题。


概览页先介绍匹配任务与合成字段示例，再说明评分流程、保存结果和代码入口。
方法页先说明数据来源与项目用途，再解释归一化和模型；token 预算与训练参数
可展开查看。技术文档链接打开 GitHub 渲染的仓库文档，并明确标记 **GitHub**。
结果页内可直接阅读完整局限；CSV、JSON 入口明确用于下载机器可读文件。

先按 [README 快速开始](../README.zh-CN.md#快速开始)创建项目环境，再从仓库根目录运行：

macOS / Linux：

```bash
.venv/bin/python scripts/build_showcase.py
.venv/bin/python scripts/build_showcase.py --check
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

Windows / PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts/build_showcase.py
.\.venv\Scripts\python.exe scripts/build_showcase.py --check
.\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1
```

第一条命令生成八份 HTML 和 `assets/fixed-results.csv`。检查命令只在内存中
生成并逐字节比较，不写文件。本地预览地址为
`http://127.0.0.1:8000/showcase/zh-CN/index.html`。

生成器在构建时读取本地 `results/fixed-results.json`、
`reproducibility/FINAL_MODEL.json` 和 `reproducibility/FEATURES.json`，不获取
远端服务、不运行神经模型，也不改变这三份来源。三个虚构商品对经公开的
规范化和特征函数生成教学表格；浏览器只切换已生成的表格，不计算模型分数。
生成器不拟合 IDF、不读取真实商品或标签。CSV
保留原始未舍入的数值；
页面展示值按各处注明的精度格式化。

正文源位于 `content/en.json` 和 `content/zh-CN.json`；共享页面结构位于
`templates/page.html`，本地样式位于 `assets/site.css`。修改正文或结构后重新
构建。中英文字应同步更新，数字应继续由聚合结果与模型配置生成。

## GitHub Pages

[Pages 工作流](../.github/workflows/pages.yml)在推送到 `main` 时运行公开测试，
核对生成页面并发布。只使用 Python 标准库，不安装或运行模型依赖。仓库
Pages 设置的发布来源应为 **GitHub Actions**。根网址进入中文概览，语言按钮
切换到对应英文页面。`.venv/bin/python scripts/build_pages.py`（Windows 使用
`.\.venv\Scripts\python.exe scripts/build_pages.py`）将已核验的公开清单复制到
`dist/pages`，保留网页引用的文档与文件。

[NOTICE.md](NOTICE.md)说明 Academic/Nerfies 来源，以及网站 CC BY-SA 4.0 与
模型代码 MIT 的独立适用范围；[UPSTREAM.json](UPSTREAM.json)记录固定版本
模板的原字节长度和哈希。Bulma v0.9.1 使用本地文件并保留其独立 MIT 许可。

网站展示保存的聚合结果和合成字段示例。实际预测通过模型命令入口运行，并需在本地提供商品数据、快照和训练权重。Notebook 讲解入口位于仓库 README 顶部。
