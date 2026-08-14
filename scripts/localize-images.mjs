#!/usr/bin/env node
/**
 * 将文章中的外链图片下载到本地并改写引用
 *
 * 背景：你用 Typora + PicGo 写文章，图片会以 https://... 外链形式写入 md。
 * 外链图床如果失效，文章图片就会丢失。运行本脚本可把外链图片
 * 下载到 source/img/posts/ 并把 md 里的链接改成本地路径。
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
const imgDir = join(root, 'source', 'img', 'posts');
mkdirSync(imgDir, { recursive: true });

// 匹配 ![alt](https://...)
const imgRe = /!\[([^\]]*)\]\((https?:\/\/[^)\s]+)\)/g;

// 已存在的文件名 -> 直接复用，避免重复下载
const existing = new Set(readdirSync(imgDir));

async function download(url) {
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

  writeFileSync(join(imgDir, name), buf);
  existing.add(name);
  console.log('  ↓ ' + name + ' (' + (buf.length / 1024).toFixed(0) + ' KB)');
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

    let changed = false;
    for (const m of matches) {
      const [full, alt, url] = m;
      const name = await download(url);
      if (!name) continue;
      const local = '/img/posts/' + name;
      if (full !== '![' + alt + '](' + local + ')') {
        md = md.replace(full, '![' + alt + '](' + local + ')');
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
