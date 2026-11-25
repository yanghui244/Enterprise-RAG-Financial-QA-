"""命令行入口：解析 PDF 并重建向量索引。

用法：
    python ingest.py                 # 增量解析（命中缓存）+ 重建索引
    python ingest.py --reparse       # 强制重新解析 PDF
    python ingest.py --parse-only    # 只解析，不调用 Embedding（无需 API Key）
"""

from __future__ import annotations

import argparse
import json
import sys

from config import settings, setup_logging
from core.index import build_index, list_pdfs, parse_all


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="金融研报 RAG 索引构建")
    parser.add_argument("--reparse", action="store_true", help="忽略解析缓存，强制重新解析 PDF")
    parser.add_argument("--parse-only", action="store_true", help="只解析 PDF，不构建向量索引")
    args = parser.parse_args()

    pdfs = list_pdfs()
    if not pdfs:
        print(f"未在 {settings.pdf_dir} 找到 PDF 文件", file=sys.stderr)
        return 1
    print(f"发现 {len(pdfs)} 份研报：{[p.name for p in pdfs]}")

    try:
        if args.parse_only:
            docs, failures = parse_all(reparse=args.reparse)
            for doc in docs:
                print(
                    f"- {doc.doc_id}: {len(doc.pages)} 页 / {doc.total_chars} 字符 / 解析器 {doc.parser}"
                )
            for failure in failures:
                print(f"- 解析失败 {failure['file']}: {failure['error']}", file=sys.stderr)
            return 0

        manifest = build_index(reparse=args.reparse)
    except Exception as exc:  # noqa: BLE001 - CLI 需要明确报错
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"索引已写入：{settings.storage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
