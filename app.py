"""Streamlit 前端：研报问答 + 问题清单批量评测。"""

from __future__ import annotations

import json
import logging

import streamlit as st

from config import has_api_key, settings, setup_logging
from core.index import build_index, load_index, load_manifest
from core.rag import FinancialRAG, Source

logger = logging.getLogger(__name__)
setup_logging()

st.set_page_config(page_title="金融研报 RAG", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _load_rag(version: float) -> FinancialRAG | None:
    index = load_index()
    if index is None:
        return None
    manifest = load_manifest() or {}
    return FinancialRAG(index, manifest.get("documents", {}))


def load_questions() -> list[dict]:
    if not settings.question_file.exists():
        return []
    try:
        payload = json.loads(settings.question_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        st.error(f"question.json 解析失败：{exc}")
        return []
    return payload.get("questions", [])


def source_to_dict(source: Source | dict) -> dict:
    if isinstance(source, Source):
        return {
            "doc_id": source.doc_id,
            "company": source.company,
            "year": source.year,
            "page": source.page,
            "section": source.section,
            "score": source.score,
            "snippet": source.snippet,
        }
    return source


# --------------------------------------------------------------------------- #
# 展示组件
# --------------------------------------------------------------------------- #
def render_sources(sources: list[Source | dict]) -> None:
    if not sources:
        st.caption("未检索到相关资料，本回答为“不知道”。")
        return
    sources = [source_to_dict(s) for s in sources]
    st.caption(
        "引用来源："
        + "；".join(f"{s['doc_id']} 第{s['page']}页（{s['score']:.2f}）" for s in sources[:5])
    )
    with st.expander("查看引用原文"):
        for i, source in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {source['doc_id']} · 第{source['page']}页 · "
                f"{source['section']}**（相关度 {source['score']:.3f}）"
            )
            st.text(source["snippet"])


def run_query(
    rag: FinancialRAG,
    question: str,
    kind: str = "string",
    choices: list[str] | None = None,
    docs: list[str] | None = None,
    years: list[int] | None = None,
) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    try:
        with st.chat_message("assistant"):
            if kind == "string":
                generator, sources, nodes = rag.stream(
                    question, kind=kind, docs=docs, years=years
                )
                if not nodes:
                    st.write("不知道")
                    answer_text = "不知道"
                else:
                    chunked = st.write_stream(generator)
                    answer_text = chunked if isinstance(chunked, str) else "".join(map(str, chunked))
                    answer_text = answer_text.strip() or "不知道"
                render_sources(sources)
            else:
                with st.spinner("检索并抽取结构化答案..."):
                    result = rag.answer(
                        question, kind=kind, choices=choices, docs=docs, years=years
                    )
                st.write(f"抽取结果：{result.answer}")
                render_sources(result.sources)
                answer_text = result.answer
                sources = result.sources
    except Exception as exc:  # noqa: BLE001 - 前端必须兜底，避免整页崩溃
        logger.exception("问答失败")
        st.error(f"回答失败：{exc}")
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": [source_to_dict(s) for s in sources],
            "kind": kind,
            "question": question,
        }
    )


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
def render_sidebar(manifest: dict | None) -> tuple[list[str], list[int]]:
    docs_meta = (manifest or {}).get("documents", {})

    with st.sidebar:
        st.header("知识库")
        if manifest:
            st.success(f"索引就绪：{len(docs_meta)} 份研报 / {manifest.get('chunks', 0)} 个片段")
            st.caption(f"构建时间：{manifest.get('built_at', '-')}")
            failed = manifest.get("failed_documents") or []
            if failed:
                st.warning("解析失败：" + "、".join(item["file"] for item in failed))
        else:
            st.warning("尚未构建索引")
            st.caption(f"研报目录：{settings.pdf_dir}")

        reparse = st.checkbox("强制重新解析 PDF", value=False, help="忽略缓存，重新调用 MinerU")
        if st.button("构建 / 重建索引", type="primary", width="stretch"):
            if not has_api_key():
                st.error("请先在 .env 中配置 OPENAI_API_KEY")
            else:
                with st.status("正在构建索引...", expanded=True) as status:
                    st.write("PDF 解析 → 切分 → 向量化，扫描件可能需要较长时间")
                    try:
                        result = build_index(reparse=reparse)
                        status.update(label="索引构建完成", state="complete")
                        st.write(f"{len(result['documents'])} 份研报 / {result['chunks']} 个片段")
                        _load_rag.clear()
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        status.update(label="索引构建失败", state="error")
                        st.exception(exc)

        doc_ids = sorted(docs_meta)

        def _label(doc_id: str) -> str:
            meta = docs_meta.get(doc_id, {})
            return f"{meta.get('company', doc_id)} {meta.get('year', '')}（{doc_id}）"

        selected_docs = st.multiselect(
            "检索范围（留空自动路由）",
            doc_ids,
            format_func=_label,
            help="留空时根据问题中的公司名/年份自动选择研报",
        )
        all_years = sorted({int(meta.get("year", 0)) for meta in docs_meta.values() if meta.get("year")})
        selected_years = st.multiselect("年份（留空不限）", all_years)

        st.divider()
        st.caption(
            f"LLM：{settings.llm_model}\n\n"
            f"Embedding：{settings.embed_model}\n\n"
            f"召回：top_k={settings.top_k}"
        )
        if not has_api_key():
            st.error("未检测到 OPENAI_API_KEY，请在 .env 中配置后重启。")
        if "messages" in st.session_state and st.button("清空对话", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    return selected_docs, selected_years


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
def render_qa_tab(
    rag: FinancialRAG | None,
    questions: list[dict],
    selected_docs: list[str],
    selected_years: list[int],
) -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    if rag is None:
        st.info("请先在左侧构建索引，再进行提问。")
        return

    filtered = [
        q
        for q in questions
        if not selected_docs or set(q.get("scope", {}).get("docs", [])) & set(selected_docs)
    ]
    with st.expander(f"问题清单（{len(filtered)} 条，点击直接提问）", expanded=False):
        if filtered:
            index = st.selectbox(
                "选择问题",
                options=list(range(len(filtered))),
                format_func=lambda i: f"[{filtered[i]['kind']}] {filtered[i]['question']}",
            )
            if st.button("提问", key="preset_ask"):
                st.session_state.pending = filtered[index]
        else:
            st.caption("当前筛选范围内没有问题清单条目。")

    typed = st.chat_input(
        "输入问题，例如：宁德时代2025年营业收入是多少？",
        disabled=not has_api_key(),
    )

    preset = st.session_state.pop("pending", None)
    if preset:
        scope = preset.get("scope", {})
        run_query(
            rag,
            preset["question"],
            kind=preset.get("kind", "string"),
            choices=preset.get("choices"),
            docs=selected_docs or scope.get("docs"),
            years=selected_years or scope.get("years"),
        )
    elif typed:
        run_query(
            rag,
            typed,
            docs=selected_docs or None,
            years=selected_years or None,
        )


def render_question_tab(rag: FinancialRAG | None, questions: list[dict]) -> None:
    st.caption(
        "按 question.json 中的 kind 约束输出：string 为开放式纯文本；"
        "number / boolean / enum 为结构化值。资料无依据时统一返回“不知道”。"
    )
    if not questions:
        st.warning("未找到 question.json 或其中没有问题。")
        return

    st.dataframe(
        [
            {
                "id": q["id"],
                "公司": q.get("company", ""),
                "研报": "、".join(q.get("scope", {}).get("docs", [])),
                "kind": q.get("kind", "string"),
                "问题": q["question"],
            }
            for q in questions
        ],
        width="stretch",
        hide_index=True,
    )

    by_id = {q["id"]: q for q in questions}
    picked = st.multiselect("选择要批量运行的问题", list(by_id), default=[])
    if st.button("批量运行", type="primary", disabled=not picked or rag is None):
        progress = st.progress(0.0, text="准备中...")
        results: list[dict] = []
        for i, qid in enumerate(picked):
            question = by_id[qid]
            scope = question.get("scope", {})
            try:
                result = rag.answer(
                    question["question"],
                    kind=question.get("kind", "string"),
                    choices=question.get("choices"),
                    docs=scope.get("docs"),
                    years=scope.get("years"),
                )
                answer = result.answer
                sources = [source_to_dict(s) for s in result.sources]
            except Exception as exc:  # noqa: BLE001
                logger.exception("批量运行失败：%s", qid)
                answer, sources = f"ERROR: {exc}", []
            results.append(
                {
                    "id": qid,
                    "company": question.get("company", ""),
                    "kind": question.get("kind", "string"),
                    "question": question["question"],
                    "answer": answer,
                    "sources": [
                        {k: s[k] for k in ("doc_id", "page", "section", "score")}
                        for s in sources[:3]
                    ],
                }
            )
            progress.progress(
                (i + 1) / len(picked), text=f"已完成 {i + 1}/{len(picked)}：{qid}"
            )
        st.session_state.batch_results = results
        progress.empty()

    results = st.session_state.get("batch_results") or []
    if results:
        st.dataframe(
            [
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "问题": r["question"],
                    "回答": r["answer"],
                }
                for r in results
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "导出结果 JSON",
            data=json.dumps(results, ensure_ascii=False, indent=2),
            file_name="rag_answers.json",
            mime="application/json",
        )


def main() -> None:
    st.title("金融研报 RAG")
    st.caption("宁德时代 / 中芯国际 年报问答 · 严格依据原文，无依据直接回答“不知道”")

    manifest = load_manifest()
    version = float(manifest.get("built_at_ts", 0)) if manifest else 0.0
    try:
        rag = _load_rag(version) if manifest else None
    except Exception as exc:  # noqa: BLE001
        logger.exception("索引加载失败")
        st.error(f"索引加载失败：{exc}")
        rag = None

    selected_docs, selected_years = render_sidebar(manifest)
    questions = load_questions()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []

    tab_qa, tab_questions = st.tabs(["研报问答", "问题清单"])
    with tab_qa:
        render_qa_tab(rag, questions, selected_docs, selected_years)
    with tab_questions:
        render_question_tab(rag, questions)


if __name__ == "__main__":
    main()
