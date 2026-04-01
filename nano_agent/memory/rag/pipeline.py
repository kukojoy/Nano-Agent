from typing import List, Dict, Optional, Any
import os
import hashlib
from ..embedding import get_text_embedder, get_dimension
from ..storage.qdrant_store import QdrantVectorStore

# === 文档处理器 ===
def _get_markitdown_instance():
    """获取一个 MarkItDown 实例, 用于文档处理"""
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        print("[WARNING] MarkItDown not available. Install with: pip install markitdown")
        return None

def _fallback_text_reader(path: str) -> str:
    """MarkItDown 不可用时的回退机制
    使用简单的文本读取, 尝试不同的编码以最大化兼容性
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        try:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                return f.read()
        except Exception:
            return ""

# === 文档加载与智能分块 ===  
def load_and_chunk_texts(
    paths: List[str], 
    chunk_size: int = 800, 
    chunk_overlap: int = 100, 
    namespace: Optional[str] = None, 
    source_label: str = "rag"
) -> List[Dict]:
    """统一的文档加载与分块函数
    
    支持多种格式, 内部使用 MarkItDown 将文件转换为 markdown 后进行智能分块
    """
    print(f"[RAG] 统一加载器启动: files={len(paths)} chunk_size={chunk_size} overlap={chunk_overlap} ns={namespace or 'default'}")
    chunks: List[Dict] = []
    seen_hashes = set()

    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue

        print(f"[RAG] Processing: {path}")
        ext = (os.path.splitext(path)[1] or '').lower()

        # 使用 MarkItDown 将文件转换为 .md 格式
        markdown_text = _convert_to_markdown(path)
        if not markdown_text.strip():
            print(f"[WARNING] No content extracted from: {path}")
            continue

        lang = _detect_lang(markdown_text)
        doc_id = hashlib.md5(f"{path}|{len(markdown_text)}".encode('utf-8')).hexdigest()

        # 分段 -> 分块
        para = _split_paragraphs_with_headings(markdown_text)
        token_chunks = _chunk_paragraphs(para, chunk_tokens=max(1, chunk_size), overlap_tokens=max(0, chunk_overlap))

        for ch in token_chunks:
            content = ch["content"]
            start = ch.get("start", 0)
            end = ch.get("end", start + len(content))
            norm = content.strip()
            if not norm:
                continue
                
            content_hash = hashlib.md5(norm.encode('utf-8')).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            chunk_id = hashlib.md5(f"{doc_id}|{start}|{end}|{content_hash}".encode('utf-8')).hexdigest()
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source_path": path,
                    "file_ext": ext,
                    "doc_id": doc_id,
                    "lang": lang,
                    "start": start,
                    "end": end,
                    "content_hash": content_hash,
                    "namespace": namespace or "default",
                    "source": source_label,
                    "external": True,
                    "heading_path": ch.get("heading_path"),
                    "format": "markdown",  # Mark all content as markdown-processed
                },
            })
            
    print(f"[RAG] Universal loader done: total_chunks={len(chunks)}")
    return chunks

def _convert_to_markdown(path: str) -> str:
    """使用 MarkItDown 将文件转换为 markdown 格式"""
    if not os.path.exists(path):
        return ""
    
    # 对 PDF 文件使用增强处理
    ext = (os.path.splitext(path)[1] or '').lower()
    if ext == '.pdf':
        return _enhanced_pdf_processing(path)
    
    # 其他格式使用原有 MarkItDown
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    
    try:
        result = md_instance.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown failed for {path}: {e}")
        return _fallback_text_reader(path)

def _enhanced_pdf_processing(path: str) -> str:
    """对 PDF 文件增强处理 (后处理清理), 以获得更好的文本提取质量"""
    print(f"[RAG] Using enhanced PDF processing for: {path}")
    
    # 使用原有 MarkItDown 提取
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    
    try:
        result = md_instance.convert(path)
        raw_text = getattr(result, "text_content", None)
        if not raw_text or not raw_text.strip():
            return ""
        
        # 后处理: 清理和重组文本
        cleaned_text = _post_process_pdf_text(raw_text)
        print(f"[RAG] PDF post-processing completed: {len(raw_text)} -> {len(cleaned_text)} chars")
        return cleaned_text
        
    except Exception as e:
        print(f"[WARNING] Enhanced PDF processing failed for {path}: {e}")
        return _fallback_text_reader(path)

def _post_process_pdf_text(text: str) -> str:
    """对 PDF 文本进行后处理
    
    包括清理噪音, 合并短行, 重组段落等, 以提高文本质量和可用性
    """
    import re
    
    # 1. 按行分割并清理
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 移除单个字符的行 (通常是噪音)
        if len(line) <= 2 and not line.isdigit():
            continue
            
        # 移除明显的页眉页脚噪音
        if re.match(r'^\d+$', line):  # 纯数字行 (页码)
            continue
        if line.lower() in ['github', 'project', 'forks', 'stars', 'language']:
            continue
            
        cleaned_lines.append(line)
    
    # 2. 智能合并短行
    merged_lines = []
    i = 0
    
    while i < len(cleaned_lines):
        current_line = cleaned_lines[i]
        
        # 如果当前行很短, 尝试与下一行合并
        if len(current_line) < 60 and i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i + 1]
            
            # 合并条件: 都是内容, 不是标题
            if (not current_line.endswith('：') and 
                not current_line.endswith(':') and
                not current_line.startswith('#') and
                not next_line.startswith('#') and
                len(next_line) < 120):
                
                merged_line = current_line + " " + next_line
                merged_lines.append(merged_line)
                i += 2  # 跳过下一行
                continue
        
        merged_lines.append(current_line)
        i += 1
    
    # 3. 重新组织段落
    paragraphs = []
    current_paragraph = []
    
    for line in merged_lines:
        # 检查是否是新段落的开始
        if (line.startswith('#') or  # 标题
            line.endswith('：') or   # 中文冒号结尾
            line.endswith(':') or    # 英文冒号结尾
            len(line) > 150 or       # 长句通常是段落开始
            not current_paragraph):  # 第一行
            
            # 保存当前段落
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            
            paragraphs.append(line)
        else:
            current_paragraph.append(line)
    
    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return '\n\n'.join(paragraphs)

def _detect_lang(sample: str) -> str:
    """检测文本语言"""
    try:
        from langdetect import detect
        return detect(sample[:1000]) if sample else "unknown"
    except Exception:
        return "unknown"

def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """按 markdown 结构分段, 同时保留标题路径信息"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0
    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })
    for ln in lines:
        raw = ln
        if raw.strip().startswith("#"):
            # heading line
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level-1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue
        # paragraph accumulation
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    return paragraphs

def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    """对 paragraphs 进行基于 token 的分块, 同时保留 heading_path 信息
    
    Args:
        paragraphs: 包含 content 和 heading_path 的段落列表
        chunk_tokens: 每块的目标 token 数
        overlap_tokens: 块之间的重叠 token 数
    
    Returns:
        包含 content, start, end, heading_path 的块列表
    """
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            # emit current chunk
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })
            # build overlap by keeping tail tokens
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
        chunks.append({
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        })
    return chunks

def _approx_token_len(text: str) -> int:
    """近似估计文本的 token 数量: CJK字符按1 token, 其他按空白分词"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens

def _is_cjk(ch: str) -> bool:
    """判断字符是否为 CJK (中日韩统一表意文字)"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF
    )

# === 分块索引生成 ===
def index_chunks(
    store = None, 
    chunks: List[Dict] = None, 
    cache_db: Optional[str] = None, 
    batch_size: int = 64,
    rag_namespace: str = "default"
) -> None:
    """使用统一 embedding 和 Qdrant 将 markdown chunks 索引化"""
    if not chunks:
        print("[RAG] No chunks to index")
        return
    
    # 获取全局embedder
    embedder = get_text_embedder()
    dimension = get_dimension(384)
    
    # 如果没有传入数据库实例, 创建一个默认的 QdrantVectorStore 实例 (使用默认配置)
    if store is None:
        store = _create_default_vector_store(dimension)
        print(f"[RAG] Created default Qdrant store with dimension {dimension}")
    
    # 预处理 chunk 的 content 以适应 embedder
    processed_texts = []
    for c in chunks:
        raw_content = c["content"]
        processed_content = _preprocess_markdown_for_embedding(raw_content)
        processed_texts.append(processed_content)
    
    print(f"[RAG] Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")
    
    # 批量生成 embedding
    vecs: List[List[float]] = []
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i: i + batch_size]
        try:
            part_vecs = embedder.encode(part)
            
            # Normalize to List[List[float]]
            if not isinstance(part_vecs, list):
                # 单个 numpy 数组转为列表中的列表
                if hasattr(part_vecs, "tolist"):
                    part_vecs = [part_vecs.tolist()]
                else:
                    part_vecs = [list(part_vecs)]
            else:
                # 检查是否是嵌套列表
                if part_vecs and not isinstance(part_vecs[0], (list, tuple)) and hasattr(part_vecs[0], "__len__"):
                    # numpy数组列表 -> 转换每个数组
                    normalized_vecs = []
                    for v in part_vecs:
                        if hasattr(v, "tolist"):
                            normalized_vecs.append(v.tolist())
                        else:
                            normalized_vecs.append(list(v))
                    part_vecs = normalized_vecs
                elif part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                    # 单个向量被误判为列表，实际应该包装成[[...]]
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]
            
            for v in part_vecs:
                try:
                    # 确保向量是 float 列表
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v_norm = [float(x) for x in v]
                    if len(v_norm) != dimension:
                        print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                        # 用零向量填充或截断
                        if len(v_norm) < dimension:
                            v_norm.extend([0.0] * (dimension - len(v_norm)))
                        else:
                            v_norm = v_norm[:dimension]
                    vecs.append(v_norm)
                except Exception as e:
                    print(f"[WARNING] 向量转换失败: {e}, 使用零向量")
                    vecs.append([0.0] * dimension)
                
        except Exception as e:
            print(f"[WARNING] Batch {i} encoding failed: {e}")
            print(f"[RAG] Retrying batch {i} with smaller chunks...")
            
            # 重试: 将批次分解为更小的块
            success = False
            for j in range(0, len(part), 8):  # 更小的批次
                small_part = part[j: j + 8]
                try:
                    import time
                    time.sleep(2)  # 等待2秒避免频率限制
                    
                    small_vecs = embedder.encode(small_part)
                    # Normalize to List[List[float]]
                    if isinstance(small_vecs, list) and small_vecs and not isinstance(small_vecs[0], list):
                        small_vecs = [small_vecs]
                    
                    for v in small_vecs:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        try:
                            v_norm = [float(x) for x in v]
                            if len(v_norm) != dimension:
                                print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                                if len(v_norm) < dimension:
                                    v_norm.extend([0.0] * (dimension - len(v_norm)))
                                else:
                                    v_norm = v_norm[:dimension]
                            vecs.append(v_norm)
                            success = True
                        except Exception as e2:
                            print(f"[WARNING] 小批次向量转换失败: {e2}")
                            vecs.append([0.0] * dimension)
                except Exception as e2:
                    print(f"[WARNING] 小批次 {j // 8} 仍然失败: {e2}")
                    # 为这个小批次创建零向量
                    for _ in range(len(small_part)):
                        vecs.append([0.0] * dimension)
            
            if not success:
                print(f"[ERROR] 批次 {i} 完全失败, 使用零向量")
        
        print(f"[RAG] Embedding progress: {min(i + batch_size, len(processed_texts))}/{len(processed_texts)}")
    
    # Prepare metadata with RAG tags
    metas: List[Dict] = []
    ids: List[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],  # Keep original markdown content
            "data_source": "rag_pipeline",  # RAG identification tag
            "rag_namespace": rag_namespace,
            "is_rag_data": True,  # Clear RAG data marker
        }
        # Merge chunk metadata
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])
    
    print(f"[RAG] Qdrant upsert start: n={len(vecs)}")
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if success:
        print(f"[RAG] Qdrant upsert done: {len(vecs)} vectors indexed")
    else:
        print(f"[RAG] Qdrant upsert failed")
        raise RuntimeError("Failed to index vectors to Qdrant")

def _create_default_vector_store(dimension: int = None) -> QdrantVectorStore:
    """Create default Qdrant vector store with RAG-optimized settings"""
    if dimension is None:
        dimension = get_dimension(384)
    
    # Check for Qdrant configuration
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    # 使用连接管理器
    from ..storage.qdrant_store import QdrantConnectionManager
    return QdrantConnectionManager.get_instance(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name="nano_agent_rag_vectors",
        vector_size=dimension,
        distance="cosine"
    )

def _preprocess_markdown_for_embedding(text: str) -> str:
    """
    Preprocess markdown text for better embedding quality.
    Removes excessive markup while preserving semantic content.
    """
    import re
    
    # Remove markdown headers symbols but keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown links but keep the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove markdown emphasis markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # italic
    text = re.sub(r'`([^`]+)`', r'\1', text)        # inline code
    
    # Remove markdown code blocks but keep content
    text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

# === 向量搜索 ===
def search_vectors(
    store = None, 
    query: str = "", 
    top_k: int = 8, 
    rag_namespace: Optional[str] = None, 
    only_rag_data: bool = True, 
    score_threshold: Optional[float] = None
) -> List[Dict]:
    """Search RAG vectors global embedder and Qdrant"""
    if not query:
        return []
    
    # Create default store if not provided
    if store is None:
        store = _create_default_vector_store()
    
    # Embed query with unified embedder
    qv = embed_query(query)
    
    # Build filter for RAG data
    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace
    
    try:
        return store.search_similar(
            query_vector=qv, 
            limit=top_k, 
            score_threshold=score_threshold, 
            where=where
        )
    except Exception as e:
        print(f"[WARNING] RAG search failed: {e}")
        return []

def embed_query(query: str) -> List[float]:
    """为用户查询生成嵌入向量, 使用全局embedder"""
    embedder = get_text_embedder()
    dimension = get_dimension(384)
    try:
        vec = embedder.encode(query)
        
        # Normalize to List[float]
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        
        # 处理嵌套列表情况
        if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
            vec = vec[0]  # Extract first vector if nested
        
        # 转换为float列表
        result = [float(x) for x in vec]
        
        # 检查维度
        if len(result) != dimension:
            print(f"[WARNING] Query向量维度异常: 期望{dimension}, 实际{len(result)}")
            # 用零向量填充或截断
            if len(result) < dimension:
                result.extend([0.0] * (dimension - len(result)))
            else:
                result = result[:dimension]
        
        return result
    except Exception as e:
        print(f"[WARNING] Query embedding failed: {e}")
        # Return zero vector as fallback
        return [0.0] * dimension

def search_vectors_expanded(
    store = None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    only_rag_data: bool = True,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    """Search with query expansion using unified embedding and Qdrant"""
    if not query:
        return []
    
    # Create default store if not provided
    if store is None:
        store = _create_default_vector_store()
    
    # expansions
    expansions: List[str] = [query]
    
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions))
    if enable_hyde:
        hyde_text = _prompt_hyde(query)
        if hyde_text:
            expansions.append(hyde_text)

    # unique and trim
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq[: max(1, len(uniq))]

    # distribute pool per expansion
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    # Build filter for RAG data
    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    # collect hits across expansions
    agg: Dict[str, Dict] = {}
    for q in expansions:
        qv = embed_query(q)
        hits = store.search_similar(query_vector=qv, limit=per, score_threshold=score_threshold, where=where)
        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))
            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h
    # return top by score
    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]

def _prompt_mqe(query: str, n: int) -> List[str]:
    """使用 MQE (多查询扩展) 技术生成多个查询变体, 以提高召回率"""
    try:
        from ...core.llm import NanoAgentLLM
        llm = NanoAgentLLM()
        prompt = [
            {"role": "system", "content": "你是检索查询扩展助手. 生成语义等价或互补的多样化查询. 使用中文, 简短, 避免标点."},
            {"role": "user", "content": f"原始查询: {query}\n请给出{n}个不同表述的查询, 每行一个."}
        ]
        text = llm.invoke(prompt)
        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]

def _prompt_hyde(query: str) -> Optional[str]:
    """使用 hyde 增强提示词, 生成一个可能的答案段落来丰富查询语义, 以提高检索相关性"""
    try:
        from ...core.llm import NanoAgentLLM
        llm = NanoAgentLLM()
        prompt = [
            {"role": "system", "content": "根据用户问题, 先写一段可能的答案性段落, 用于向量检索的查询文档 (不要分析过程)."},
            {"role": "user", "content": f"问题: {query}\n请直接写一段中等长度, 客观, 包含关键术语的段落."}
        ]
        return llm.invoke(prompt)
    except Exception:
        return None
    
# === RAG pipeline 工厂函数 ===
def create_rag_pipeline(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "nano_agent_rag_vectors",
    rag_namespace: str = "default"
) -> Dict[str, Any]:
    """使用统一的 embedding 和 Qdrant 数据库构建一个完整的 RAG pipeline.
    
    Returns:
        字典, 包含数据库实例, namespace以及实用函数 (add_documents/search/search_advanced/get_stats)
    """

    dimension = get_dimension()

    store = QdrantVectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_size=dimension,
        distance="cosine"
    )

    def add_documents(file_paths: List[str], chunk_size: int = 800, chunk_overlap: int = 100) -> int:
        """RAG pipeline 中的添加文档功能
        
        Args:
            file_path: 文档文件路径
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
        
        Return:
            分块数量
        """

        # 分块
        chunks = load_and_chunk_texts(
            paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag"
        )

        # 建立索引
        index_chunks(
            store=store,
            chunks=chunks,
            rag_namespace=rag_namespace
        )

        return len(chunks)
    
    def search(query: str, top_k: int = 8, score_threshold: Optional[float] = None):
        """搜索 RAG 知识库"""
        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            score_threshold=score_threshold
        )

    def search_advanced(
        query: str, 
        top_k: int = 8, 
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None
    ):
        """高级检索"""
        return search_vectors_expanded(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            score_threshold=score_threshold
        )
    
    def get_stats():
        """获取 pipeline 统计信息 (实际上是数据库的统计信息)"""
        return store.get_collection_stats()

    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats
    }

# === 备用 ===
""" TODO: 目前框架内部没有使用到以下函数 """

def _is_markitdown_supported_format(path: str) -> bool:
    pass

def build_graph_from_chunks(neo4j, chunks: List[Dict]) -> None:
    pass

def rerank_with_cross_encoder(
    query: str, 
    items: List[Dict], 
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
    top_k: int = 10
) -> List[Dict]:
    pass

def _try_load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    pass

def compute_graph_signals_from_pool(vector_hits: List[Dict], same_doc_weight: float = 1.0, proximity_weight: float = 1.0, proximity_window_chars: int = 1600) -> Dict[str, float]:
    pass

def rank(
    vector_hits: List[Dict], 
    graph_signals: Optional[Dict[str, float]] = None, 
    w_vector: float = 0.7, 
    w_graph: float = 0.3
) -> List[Dict]:
    pass

def merge_snippets(ranked_items: List[Dict], max_chars: int = 1200) -> str:
    pass

def expand_neighbors_from_pool(
        selected: List[Dict], 
        pool: List[Dict], 
        neighbors: int = 1,
        max_additions: int = 5
) -> List[Dict]:
    pass

def merge_snippets_grouped(
    ranked_items: List[Dict], 
    max_chars: int = 1200, 
    include_citations: bool = True
) -> str:
    pass

def compress_ranked_items(
        ranked_items: List[Dict], 
        enable_compression: bool = True, 
        max_per_doc: int = 2, 
        join_gap: int = 200
) -> List[Dict]:
    pass

def tldr_summarize(text: str, bullets: int = 3) -> Optional[str]:
    pass