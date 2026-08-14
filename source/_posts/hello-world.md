---
title: 欢迎来到我的博客
date: 2026-08-04
updated: 2026-08-04
categories: [随笔]
tags: [欢迎, 开始]
index_img: /img/covers/welcome.svg
---

你好呀，欢迎来到我的博客 👋

这里会用来记录我的**学习笔记**和**日常生活**：

- 📚 学习：技术文章、读书笔记、学习心得
- 🌱 生活：日常随笔、旅行记录、成长感悟

## 如何写一篇新文章

在项目根目录运行：

``` bash
npx hexo new post 文章标题
```

然后编辑 `source/_posts/` 下生成的 Markdown 文件，文章会自动出现在首页。

## 本地预览

``` bash
npx hexo server
```

浏览器打开 http://localhost:4000 就能看到效果。

## 发布上线

写完文章后，提交代码并推送到 GitHub，Vercel 会自动重新部署：

``` bash
git add .
git commit -m "新文章"
git push
```
