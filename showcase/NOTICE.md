# Website attribution and license scope / 网站来源与许可范围

[🌐 English](#english) | [🇨🇳 简体中文说明](#zh-cn)

<a id="english"></a>

## English

This site adapts the hero, publication resource links, centered content layout,
sections, footer and selected styling of the **Academic Project Page Template**
by Eliahu Horwitz. That template was originally adapted from **Nerfies** by
Keunhong Park and the Nerfies authors.

- Academic template: <https://github.com/eliahuhorwitz/Academic-project-page-template>
- Exact source commit: `d38af1ccae1ce82c3404d2820c4c646afd409f81`
- Original page: <https://eliahuhorwitz.github.io/Academic-project-page-template/>
- Nerfies page: <https://nerfies.github.io/>
- Nerfies website source: <https://github.com/nerfies/nerfies.github.io>
- Website license: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0): <https://creativecommons.org/licenses/by-sa/4.0/>
- Original English legal text: [LICENSE](LICENSE)
- Verified source lengths, SHA-256 and Git blob hashes: [UPSTREAM.json](UPSTREAM.json)

The original template requests a link back in the footer. Both the Academic
template and Nerfies links, the license link, and an indication of changes are
retained on every content page. Changes include the four-page bilingual
navigation, a report sidebar, local system fonts, a warm brown and ivory paper palette with gold labels, text, diagrams,
aggregate charts, result table and reproduction controls. Placeholder authors,
paper links, sample images, videos, analytics, external fonts, jQuery, carousels
and PDF-viewer dependencies have been removed.

The adapted website pages, shared page template, website text, CSS, diagrams,
site interaction code and generator in `scripts/build_showcase.py` are provided
under **CC BY-SA 4.0**. The separate model package and its existing project code
retain the repository's [MIT license](../LICENSE). This website license does not
relicense product data, labels, trained weights or third-party models, and none
of those artifacts is redistributed here.

Bulma v0.9.1 is included as the original local layout CSS. It has its own **MIT**
license; its file header and [original license](assets/vendor/BULMA-LICENSE) are
retained. No Font Awesome, Academicons, external font, video or carousel asset
is included. `UPSTREAM.json` records the lengths and hashes of the pinned
template files. `bulma.min.css` is retained byte for byte.

The information hierarchy also draws on the [VitePress documentation](https://vitepress.dev/guide/what-is-vitepress) and [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html). Their text and CSS are not copied.

<a id="zh-cn"></a>

## 简体中文说明

[🌐 English](#english) | **🇨🇳 简体中文说明**

本站基于 Academic Project Page Template 的标题、资源按钮、居中内容区、分节、
页脚及部分样式改编，保留其来源、Nerfies来源、许可链接和改动说明。网站派生
部分及生成器适用 CC BY-SA 4.0；模型包及原项目代码继续适用独立的 MIT 许可。
本页中文说明用于解释范围，不替代 LICENSE 中的英文许可原文。数据与第三方
模型遵循各自条款；本站没有分发商品记录、标签、模型快照或训练权重。

页面改成中英四页，加入页内侧栏，使用本地系统字体和暖棕与米白纸张配色、金色标签，并加入
自制教学示例、方法图、聚合结果
图表及复现入口。示例商品没有经过模型预测。保留的 Bulma CSS 另适用其 MIT
许可，文件头及原许可文本均保留。
固定模板文件的长度与哈希记录于 `UPSTREAM.json`；Bulma CSS 按原字节保留。

信息层级还参考了上述 VitePress 文档和 scikit-learn 用户指南；未复制其正文或CSS。
