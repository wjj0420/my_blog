#!/usr/bin/env node
/**
 * 将文章中的外链图片下载到本地并改写引用（Typora 友好版）
 *
 * 背景：你用 Typora + PicGo 写文章，图片会以 https://... 外链形式写入 md。
 * 外链图床如果失效，文章图片就会丢失。运行本脚本可把外链图片
 * 下载到该文章同名的资源目录（如 source/_posts/文章名/），
 * 并把 md 里的链接改写成相对文件名（如 ![](xxx.png)）。
 *
 * 这样：
 *  - Typora 本地预览：图片和 md 同目录，可正常显示；
 *  - 网站部署：Hexo 的 post_asset_folder 会把资源目录复制到文章 URL 下。
 *
 * 用法：
 *   node scripts/localize-images.mjs            # 处理全部文章
 *   node scripts/localize-images.mjs 文件名.md   # 只处理指定文章
 *
 * 特性：幂等（已下载过的图片会跳过并复用）；自动识别图片真实格式。
 */
import { readdirSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join, dirname, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const postsDir = join(root, 'source', '_posts');

// 匹配 ![alt](https://...)
const imgRe = /!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g;

async function download(url, postName) {
  const postImgDir = join(postsDir, postName);
  mkdirSync(postImgDir, { recursive: true });
  const existing = new Set(readdirSync(postImgDir));

  // 从 URL 推导基础文件名
  const pathname = new URL(url).pathname;
  const base = decodeURIComponent(basename(pathname)).replace(/[\\/:*?"<>|]/g, '_');
  let name = base || ('image-' + Date.now());

  const res = await fetch(url);
  if (!res.ok) {
    console.warn('  ⚠ 下载失败 (' + res.status + '): ' + url);
    return null;
  }

  const buf = Buffer.from(await res.arrayBuffer());
  const ct = res.headers.get('content-type') || '';

  // 判断真实扩展名：优先 Content-Type，其次文件头魔数，最后回退 URL 扩展名
  let ext;
  if (ct.includes('png')) ext = 'png';
  else if (ct.includes('jpeg') || ct.includes('jpg')) ext = 'jpg';
  else if (ct.includes('webp')) ext = 'webp';
  else if (ct.includes('gif')) ext = 'gif';
  else if (buf.length > 3 && buf[0] === 0x89 && buf[1] === 0x50) ext = 'png';
  else if (buf.length > 3 && buf[0] === 0xff && buf[1] === 0xd8) ext = 'jpg';
  else ext = extname(name).replace('.', '') || 'bin';

  if (extname(name).toLowerCase() !== '.' + ext) {
    name = name.replace(/\.[^.]+$/, '') + '.' + ext;
  }

  if (existing.has(name)) {
    return name; // 已下载过，复用
  }

  writeFileSync(join(postImgDir, name), buf);
  console.log('  ↓ ' + postName + '/' + name + ' (' + (buf.length / 1024).toFixed(0) + ' KB)');
  return name;
}

async function main() {
  const args = process.argv.slice(2);
  let files = args.length
    ? args.filter((f) => f.endsWith('.md'))
    : readdirSync(postsDir).filter((f) => f.endsWith('.md'));

  if (!files.length) {
    console.log('未找到要处理的 md 文件');
    return;
  }

  let totalDownloaded = 0;
  let totalChanged = 0;

  for (const file of files) {
    const fp = join(postsDir, file);
    if (!existsSync(fp)) {
      console.warn('跳过（不存在）: ' + file);
      continue;
    }
    let md = readFileSync(fp, 'utf-8');
    const matches = [...md.matchAll(imgRe)];
    if (!matches.length) {
      console.log('· ' + file + ' （无外链图片，跳过）');
      continue;
    }

    const postName = file.endsWith('.md') ? file.slice(0, -3) : file;
    let changed = false;
    for (const m of matches) {
      const [full, alt, url] = m;
      const name = await download(url, postName);
      if (!name) continue;
      // 相对引用：带文章目录前缀（如 文章名/图片.png），Typora 和网站都能显示
      const rel = postName + '/' + name;
      if (full !== '![' + alt + '](' + rel + ')') {
        md = md.replace(full, '![' + alt + '](' + rel + ')');
        changed = true;
        totalDownloaded++;
      }
    }
    if (changed) {
      writeFileSync(fp, md);
      totalChanged++;
      console.log('✔ ' + file + ' 已改写');
    }
  }

  console.log('\n完成：下载/复用 ' + totalDownloaded + ' 张图片，改写 ' + totalChanged + ' 篇文章');
  console.log('提示：检查效果后 git add -A && git commit && git push 即可部署');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
