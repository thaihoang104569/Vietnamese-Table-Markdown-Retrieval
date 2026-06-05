# Vietnamese Table Markdown Retrieval & RAG Pipeline

> **Một pipeline Retrieval-Augmented Generation (RAG) hiện đại dành riêng cho việc truy xuất thông tin từ bảng Markdown tiếng Việt**, được xây dựng trên bộ dữ liệu [GreenNode/GreenNode-Table-Markdown-Retrieval-VN](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN).

---

## 📋 Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Kiến trúc Pipeline](#kiến-trúc-pipeline)
- [Bộ dữ liệu](#bộ-dữ-liệu)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Cấu hình tham số](#cấu-hình-tham-số)
- [Chạy trên Kaggle](#chạy-trên-kaggle)
- [Đánh giá hiệu suất](#đánh-giá-hiệu-suất)
- [Mô tả chi tiết các module](#mô-tả-chi-tiết-các-module)

---

## Tổng quan dự án

Dự án này giải quyết bài toán **Vietnamese Table Markdown Retrieval** — tìm kiếm và trả lời câu hỏi từ các tài liệu chứa bảng biểu dạng Markdown tiếng Việt. Đây là bài toán phi-trivial vì:

- Thông tin nằm trong **cấu trúc bán cấu trúc** (bảng Markdown), không phải văn bản thuần túy.
- Câu hỏi thường hỏi về **giá trị cụ thể trong ô bảng** (số liệu, mã hiệu, tên model, v.v.).
- Tiếng Việt đòi hỏi **phân tách từ chuyên biệt** (word segmentation) do từ ghép viết tách rời.

Pipeline được thiết kế theo mô hình **Hybrid Search + Reranking + RAG** hiện đại, có thể chạy trên:
- 💻 **Máy cục bộ** (CPU — tốc độ chậm hơn, chỉ phù hợp với indexing và retrieval).
- ☁️ **Kaggle Free Tier** (GPU T4 — khuyến nghị để chạy Dense Embedding và LLM Generator).

---

## Kiến trúc Pipeline

```
                            ┌─────────────────────────────────────────────┐
                            │               RAG PIPELINE                  │
                            │                                             │
  [Câu hỏi tiếng Việt]      │   ┌─────────┐      ┌─────────────┐        │
         │                  │   │  BM25   │─────►│             │        │
         ├─────────────────►│   │ (Lexical)│      │     RRF     │        │
         │                  │   └─────────┘  ┌──►│   Fusion    │        │
         │                  │                │   │             │        │
         │                  │   ┌─────────┐  │   └──────┬──────┘        │
         └─────────────────►│   │  Dense  │──┘          │               │
                            │   │  FAISS  │         ┌───▼────────┐      │
                            │   │(Semantic)│         │ Cross-Enc  │      │
                            │   └─────────┘         │  Reranker  │      │
                            │                        └───┬────────┘      │
                            │                            │               │
                            │                     ┌──────▼──────┐       │
                            │                     │  Qwen 2.5   │       │
                            │                     │  7B-Instruct│       │
                            │                     │  (4-bit Q)  │       │
                            │                     └──────┬───────┘      │
                            └────────────────────────────┼──────────────┘
                                                         │
                                               [Câu trả lời tiếng Việt]
```

### Luồng xử lý chi tiết

| Bước | Module | Mô tả |
|------|--------|-------|
| 1 | `bm25_retriever.py` | Phân tách từ tiếng Việt → BM25 → Top 50 documents (khớp từ khóa chính xác) |
| 2 | `dense_retriever.py` | Mã hóa câu hỏi bằng `GreenNode/M3-GN-VN` → FAISS Inner Product → Top 50 documents (ngữ nghĩa) |
| 3 | `fusion.py` | Gộp kết quả 2 bước trên bằng Reciprocal Rank Fusion (RRF, k=60) → Top 20 candidates |
| 4 | `reranker.py` | Cross-Encoder `bge-reranker-v2-m3` chấm điểm từng cặp (query, doc) → Top 5 documents |
| 5 | `generator.py` | Xây dựng prompt tiếng Việt + gọi `Qwen2.5-7B-Instruct` (4-bit) → Sinh câu trả lời hoàn chỉnh |

---

## Bộ dữ liệu

**Nguồn:** [`GreenNode/GreenNode-Table-Markdown-Retrieval-VN`](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN)  
**Chuẩn benchmark:** [MTEB](https://github.com/embeddings-benchmark/mteb) — Massive Text Embedding Benchmark  
**Lĩnh vực:** Tài chính, Bách khoa, Non-fiction  
**Ngôn ngữ:** Tiếng Việt (monolingual)

### Thống kê dữ liệu

| Thành phần | Số lượng (raw) | Số lượng (processed) | Mô tả |
|---|---|---|---|
| **Corpus** | 44,678 | 44,678 | Tài liệu chứa 2–5 bảng Markdown + mô tả |
| **Queries (tổng)** | 178,897 | 178,886 | Câu hỏi tiếng Việt, 11 câu rỗng đã bị lọc |
| **Qrels Train** | 143,106 | 143,097 | Ánh xạ query↔document cho tập huấn luyện |
| **Qrels Test** | 35,791 | 35,789 | Ánh xạ query↔document cho tập kiểm tra |

### Đặc điểm dữ liệu

- **Corpus**: Mỗi tài liệu trung bình ~267 từ, tối đa ~744 từ — phù hợp với context window mô hình nhúng, **không cần chunk**.
- **Queries**: Trung bình ~20 từ/câu hỏi, hỏi về giá trị cụ thể trong bảng.
- **Qrels**: Mỗi document trong train set có trung bình **4.0 câu hỏi** liên quan; không có overlap giữa train và test.
- **Cấu trúc bảng**: Phần lớn tài liệu chứa 2–5 bảng Markdown kèm đoạn mô tả.

### Ví dụ dữ liệu

**Corpus document:**
```
Dưới đây là bảng thông số kỹ thuật cho biến dòng:
| Model | Dòng vào (A) | Dòng ra (A) | Tần số (Hz) | Điện áp (V) | Công suất (W) |
|-------|--------------|-------------|-------------|--------------|---------------|
| M1    | 10           | 5           | 50          | 220          | 1100          |
| M4    | 25           | 12.5        | 60          | 240          | 3000          |
...
```

**Query:** `"Model nào có công suất 3000W?"`  
**Expected answer:** `M4`

---

## Cấu trúc thư mục

```
IR/
├── .venv/                          # Python virtual environment
└── src/
    ├── Data/
    │   ├── raw/                    # Dữ liệu thô từ Hugging Face
    │   │   ├── corpus.jsonl        # 44,678 tài liệu (~72 MB)
    │   │   ├── queries.jsonl       # 178,897 câu hỏi (~31 MB)
    │   │   └── qrels/
    │   │       ├── train.jsonl     # 143,106 ánh xạ train
    │   │       └── test.jsonl      # 35,791 ánh xạ test
    │   └── processed/              # Dữ liệu đã xử lý (output của process_data.py)
    │       ├── corpus.jsonl        # Corpus sạch (không có trường title)
    │       ├── queries.jsonl       # Queries sạch (đã lọc câu rỗng)
    │       ├── qrels/
    │       │   ├── train.jsonl
    │       │   └── test.jsonl
    │       ├── bm25_index.pkl      # BM25 index (tự tạo khi chạy lần đầu)
    │       ├── dense_index.faiss   # FAISS vector index (tự tạo khi chạy lần đầu)
    │       └── dense_doc_ids.json  # Document ID mapping cho FAISS
    │
    ├── load_data.py        # Tải dữ liệu thô từ Hugging Face Hub
    ├── process_data.py     # Làm sạch & tiền xử lý dữ liệu
    ├── config.py           # Tất cả tham số cấu hình
    ├── tokenizer.py        # Phân tách từ tiếng Việt (underthesea), table-aware
    ├── bm25_retriever.py   # BM25 Lexical Retriever
    ├── dense_retriever.py  # Dense Vector Retriever (FAISS + SentenceTransformer)
    ├── fusion.py           # Reciprocal Rank Fusion (RRF)
    ├── reranker.py         # Cross-Encoder Reranker
    ├── generator.py        # Local LLM Generator (Qwen2.5-7B, 4-bit quantized)
    ├── pipeline.py         # RAG Pipeline tích hợp
    ├── evaluate.py         # Đánh giá hiệu suất (NDCG@10, Recall@K, MAP@10)
    ├── sample_run.py       # Script kiểm thử nhanh trên mini-corpus (CPU friendly)
    └── requirements.txt    # Danh sách thư viện
```

---

## Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị (Kaggle T4) |
|---|---|---|
| **Python** | 3.10+ | 3.10+ |
| **RAM** | 16 GB | 16 GB |
| **GPU VRAM** | Không bắt buộc (CPU) | 16 GB (T4 GPU) |
| **Dung lượng đĩa** | ~15 GB (data + model weights) | ~20 GB |

> **Lưu ý:** Việc mã hóa 44,678 tài liệu (Dense Indexing) và chạy LLM Generator đòi hỏi GPU. Trên CPU, chỉ nên dùng BM25 retrieval và bỏ qua bước Dense + LLM.

---

## Hướng dẫn cài đặt

### 1. Clone repository và tạo môi trường ảo

```bash
cd IR
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r src/requirements.txt
```

Các thư viện chính:

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `sentence-transformers` | ≥5.0 | Mã hóa văn bản thành vector nhúng |
| `faiss-cpu` | ≥1.8 | Lưu trữ và tìm kiếm vector (in-memory) |
| `rank-bm25` | ≥0.2 | Thuật toán BM25 Okapi |
| `underthesea` | ≥6.0 | Phân tách từ tiếng Việt |
| `transformers` | ≥4.40 | Load và chạy mô hình LLM |
| `bitsandbytes` | ≥0.43 | Quantization 4-bit cho LLM |
| `accelerate` | ≥0.30 | Quản lý thiết bị GPU/CPU cho HuggingFace |

### 3. Tải dữ liệu

```bash
python -m src.load_data --output-dir src/Data
```

Lệnh này sẽ tải dữ liệu từ Hugging Face Hub về thư mục `src/Data/raw/`.

### 4. Xử lý dữ liệu

```bash
python -m src.process_data
```

Bước này thực hiện:
- Lọc bỏ 11 câu hỏi trống/chỉ chứa khoảng trắng.
- Loại bỏ trường `title` (toàn bộ có giá trị `"None"`, không hữu ích).
- Đồng bộ hóa tệp `qrels` với các câu hỏi hợp lệ.
- Lưu kết quả vào `src/Data/processed/`.

---

## Hướng dẫn sử dụng

### Kiểm thử nhanh trên CPU (Mini Sample)

Chạy pipeline trên 100 tài liệu đầu tiên — không cần GPU, hoàn thành trong ~2 phút:

```bash
python -m src.sample_run
```

Script này sẽ:
1. Xây dựng BM25 index và Dense FAISS index mini.
2. Chạy một câu truy vấn mẫu qua đầy đủ các bước Retrieval (BM25 → Dense → RRF → Rerank).
3. In kết quả từng bước và Mock LLM Prompt.
4. Tự dọn dẹp file tạm sau khi chạy.

### Đánh giá Retrieval trên tập Test (Không LLM)

```bash
python -m src.evaluate --sample-size 500
```

Đánh giá pipeline truy xuất trên 500 câu hỏi ngẫu nhiên từ tập test và in ra các chỉ số NDCG@10, Recall@K, MAP@10.

### Đánh giá đầy đủ RAG với LLM Generator (Cần GPU)

```bash
python -m src.evaluate --with-llm --sample-size 200
```

Bổ sung thêm bước sinh câu trả lời từ `Qwen2.5-7B-Instruct` (4-bit quantized). In ra 3 ví dụ đầu tiên gồm câu hỏi, tài liệu được truy xuất và câu trả lời sinh ra.

### Sử dụng Pipeline trong Code

```python
from src.pipeline import RAGPipeline

# Khởi tạo pipeline (bỏ qua LLM nếu không có GPU)
pipeline = RAGPipeline(include_llm=False)
pipeline.initialize()  # Tự động build/load index

# Chỉ truy xuất tài liệu
results = pipeline.retrieve_only("Model nào có công suất 3000W?")
for doc in results:
    print(doc["corpus-id"], doc["score"])

# Truy xuất + sinh câu trả lời (cần GPU + include_llm=True)
output = pipeline.query("Model nào có công suất 3000W?")
print(output["answer"])
print(output["retrieved_docs"])
```

---

## Cấu hình tham số

Tất cả tham số được quản lý tập trung trong [`src/config.py`](src/config.py):

```python
class Config:
    # Models
    DENSE_MODEL_NAME  = "GreenNode/M3-GN-VN"       # Embedding model (fine-tuned trên dataset này)
    RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # Cross-Encoder reranker
    LLM_MODEL_NAME    = "Qwen/Qwen2.5-7B-Instruct" # LLM generator (quantized 4-bit)

    # Retrieval
    TOP_K_RETRIEVE = 50   # Số candidates từ BM25 và Dense mỗi loại
    TOP_K_RRF      = 20   # Số candidates sau bước RRF fusion
    TOP_K_RERANK   = 5    # Số candidates cuối sau Reranker (đưa vào LLM)

    # BM25
    BM25_K1 = 1.5
    BM25_B  = 0.75

    # RRF
    RRF_K = 60            # Hằng số RRF (càng lớn, càng ưu tiên top rank)

    # LLM Generation
    LLM_MAX_NEW_TOKENS = 300
    LLM_TEMPERATURE    = 0.1  # Greedy-ish decoding cho câu trả lời thực tế
```

---

## Chạy trên Kaggle

Dự án được tối ưu hóa để chạy trên **Kaggle Notebooks (T4 GPU — miễn phí)**.

### Bước 1: Upload dự án lên Kaggle

Nén thư mục `src/` và upload lên Kaggle Dataset, hoặc sử dụng Kaggle API.

### Bước 2: Tạo Notebook mới với GPU T4

Trong Kaggle Notebook, chọn **Settings → Accelerator → GPU T4 x2**.

### Bước 3: Chạy các lệnh

```python
# Cài đặt thư viện
!pip install sentence-transformers faiss-gpu rank-bm25 underthesea transformers bitsandbytes accelerate

# (Dữ liệu đã có sẵn trong /kaggle/input/...)

# Xử lý dữ liệu
!python -m src.process_data

# Chạy đánh giá đầy đủ với LLM
!python -m src.evaluate --with-llm --sample-size 500
```

> **Lưu ý:** Trên Kaggle, thay `faiss-cpu` bằng `faiss-gpu` để tận dụng GPU cho indexing nhanh hơn.

---

## Đánh giá hiệu suất

Các chỉ số được tính trên tập kiểm tra (Test split) của bộ dữ liệu:

| Chỉ số | Mô tả |
|---|---|
| **NDCG@10** | Normalized Discounted Cumulative Gain (chỉ số chính của MTEB) |
| **MAP@10** | Mean Average Precision tại K=10 |
| **Recall@1** | Tỷ lệ tìm thấy đúng document liên quan ở vị trí đầu tiên |
| **Recall@5** | Tỷ lệ tìm thấy đúng document liên quan trong Top 5 |
| **Recall@10** | Tỷ lệ tìm thấy đúng document liên quan trong Top 10 |

Chạy đánh giá:

```bash
python -m src.evaluate --sample-size 0   # 0 = đánh giá toàn bộ test set
```

---

## Mô tả chi tiết các module

### `load_data.py` — Tải dữ liệu thô
Tải toàn bộ dataset từ Hugging Face Hub về máy cục bộ sử dụng `snapshot_download`. Không thực hiện bất kỳ xử lý nào trên dữ liệu.

### `process_data.py` — Tiền xử lý dữ liệu
- Lọc bỏ các câu hỏi rỗng hoặc chỉ chứa khoảng trắng.
- Loại bỏ trường `title` (100% có giá trị `"None"`).
- Đồng bộ hóa tệp `qrels` với các câu hỏi hợp lệ.

### `tokenizer.py` — Phân tách từ tiếng Việt (Table-Aware)
Sử dụng thư viện `underthesea` để gộp các từ ghép tiếng Việt (e.g., `điện thoại` → `điện_thoại`). Được thiết kế đặc biệt để xử lý từng ô trong bảng Markdown mà không làm vỡ cấu trúc bảng. Chỉ áp dụng cho BM25 — mô hình nhúng dùng raw text.

### `bm25_retriever.py` — Tìm kiếm từ khóa (Lexical Search)
- **Build**: Phân tách từ toàn bộ corpus, fit `BM25Okapi`, lưu index vào file `.pkl`.
- **Load**: Tải lại index đã build để tránh rebuild.
- **Retrieve**: Tính BM25 score, trả về Top-K documents.

### `dense_retriever.py` — Tìm kiếm ngữ nghĩa (Dense Vector Search)
- **Build**: Mã hóa corpus thành vector nhúng với `GreenNode/M3-GN-VN`, xây dựng `FAISS IndexFlatIP` (Inner Product ~ Cosine Similarity).
- **Load**: Tải FAISS index và document ID mapping từ file.
- **Retrieve**: Mã hóa câu hỏi, tìm kiếm top-K nearest vectors.

### `fusion.py` — Reciprocal Rank Fusion (RRF)
Gộp kết quả từ BM25 và Dense retriever theo công thức:
$$RRF(d) = \sum_{m} \frac{1}{k + r_m(d)}$$
Phương pháp này không cần huấn luyện thêm và ổn định hơn score-based fusion.

### `reranker.py` — Cross-Encoder Reranker
Nhận danh sách candidates từ RRF, tính điểm similarity thực tế bằng Cross-Encoder `BAAI/bge-reranker-v2-m3` (xử lý từng cặp query-document), trả về Top-K kết quả chính xác nhất.

### `generator.py` — LLM Generator
- Load `Qwen/Qwen2.5-7B-Instruct` với quantization **NF4 4-bit** (`bitsandbytes`) để phù hợp với VRAM T4.
- Xây dựng prompt tiếng Việt bao gồm tài liệu ngữ cảnh và câu hỏi.
- Sinh câu trả lời với `temperature=0.1` (deterministic) để đảm bảo độ chính xác thực tế.

### `pipeline.py` — RAG Pipeline
- `initialize()`: Tự động build hoặc load BM25/Dense index, load Reranker và LLM.
- `retrieve_only(query)`: Chạy BM25 → Dense → RRF → Rerank, trả về Top-K tài liệu.
- `query(query)`: Chạy toàn bộ pipeline bao gồm sinh câu trả lời từ LLM.

### `evaluate.py` — Đánh giá hiệu suất
Chạy pipeline trên tập test và tính toán NDCG@10, MAP@10, Recall@1/5/10. Hỗ trợ `--sample-size` để đánh giá nhanh trên tập con.

### `sample_run.py` — Kiểm thử nhanh (CPU)
Script tích hợp test end-to-end trên 100 tài liệu mẫu, không cần GPU. Dùng để kiểm tra imports và logic pipeline trước khi chạy toàn bộ trên Kaggle.

---

## Tài liệu tham khảo

- [GN-TRVN: A Benchmark for Vietnamese Table Markdown Retrieval Task](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN)
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316)
- [BGE-M3: Multi-Lingual, Multi-Functionality, Multi-Granularity](https://arxiv.org/abs/2402.03216)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://dl.acm.org/doi/10.1145/1571941.1572114)
