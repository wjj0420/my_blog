---
title: 欢迎来到我的博客
date: 2026-08-04
updated: 2026-08-14
categories: [随笔]
tags: [欢迎, 开始]
index_img: /img/covers/welcome.svg
---

你好呀，欢迎来到我的博客 👋

这里会用来记录我的**学习笔记**和**日常生活**：

- 📚 学习：技术文章、读书笔记、学习心得
- 🌱 生活：日常随笔、旅行记录、成长感悟

---

## ✍️ 如何写一篇新文章

### 方式一：命令行创建（适合纯文本文章）

在项目根目录运行：

``` bash
npx hexo new post 文章标题
```

然后编辑 `source/_posts/` 下生成的 Markdown 文件，文章会自动出现在首页。

### 方式二：Typora + PicGo 写文章（推荐，含图片）

1. 在 Typora 中正常写作，插入图片时会自动调用 **PicGo** 上传到云端图床，md 里会写入 `https://picgocloud.com/...` 这样的外链；
2. 写完后把文件保存到 `source/_posts/` 目录下（文件名建议用英文，页面标题由 front-matter 的 `title` 决定）；
3. 文章最前面需要加上 front-matter，至少包含 `title` 和 `date`，示例：

``` yaml
---
title: 文章标题
date: 2026-08-14
categories: [学习]
tags: [标签1, 标签2]
---
```

> ⚠️ 提示：外链图片依赖图床服务，图床失效图片就会丢失。如果担心，可以执行下面的「图片本地化」步骤，把图片下载到仓库里。

## 📥 图片本地化（可选，推荐）

把文章里的外链图片下载到本站仓库，由自己的站点直接提供，不再依赖第三方图床：

``` bash
node scripts/localize-images.mjs             # 处理全部文章
node scripts/localize-images.mjs 文件名.md    # 只处理指定文章
```

脚本会自动：

- 下载外链图片到该文章同名的资源目录（如 `source/_posts/文章名/`）；
- 把 md 里的 `https://...` 链接改写为相对文件名（如 `![](xxx.png)`），Typora 和网站都能正常显示；
- 已处理过的图片自动跳过，可放心重复运行。

## 🖼️ 文章封面图（可选）

首页卡片默认使用统一的渐变封面。想给文章配专属封面：

1. 在 `source/img/covers/` 下放一张 `xxx.svg`（可参照现有的模板修改文字和配色）；
2. 在文章 front-matter 中加一行：

``` yaml
index_img: /img/covers/xxx.svg
```

## 🖥️ 本地预览

``` bash
npx hexo server
```

浏览器打开 http://localhost:4000 就能看到效果。

## 🚀 发布上线

写完文章后，提交代码并推送到 GitHub，Vercel 会自动重新部署：

``` bash
git add .
git commit -m "新文章"
git push
```

完整流程回顾：**Typora 写作 → （可选）图片本地化 → git 提交推送 → Vercel 自动部署** ✅
