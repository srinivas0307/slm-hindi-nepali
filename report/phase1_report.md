## Large Language Model and Agents - Individual Project

### Phase 1 : Data Collection and Tokenizer Construction

### High Resource Language: Hindi

For the high-resource language, I selected **Hindi**. Hindi is widely spoken and has a massive digital footprint, making it easy and ideal to get quality data and reaching the target of 500 million tokens.

#### 1. Manual Data Collection
* **Wikipedia Hindi:** 
  * **Extracted Tokens:** 41,793,201
* **Wikisource Hindi:**
  * **Extracted Tokens:** 51,441,132
* **Wiktionary Hindi:**
  * **Extracted Tokens:** 4,042,118
* **Wikibooks Hindi:**
  * **Extracted Tokens:** 2,169,174
* **Wikiquote Hindi:**
  * **Extracted Tokens:** 484,758
* **Scraping Hindi:**
  * **Extracted Tokens:** 186,630

#### 2. Public Data Collection
* **IndicCorp v2 (Hindi):** To reach the target, I used the downloaded *IndicCorp v2* dataset. This is a huge collection of text primarily consisting of news articles, blogs, and public web pages. This public dataset provides the sheer volume and everyday vocabulary necessary for the model to generalize well across common topics.
  * **Extracted Tokens:** 485,521,928

**Raw Data (Google Drive):** [Hindi Raw Dataset](https://drive.google.com/drive/folders/1WY1NF3ANJv1sy-3VNYX9pliG313ckX6R?usp=sharing)

#### Data Cleaning

The combined raw corpus of **~9.13M documents** (from all sources) was passed through a multi-stage cleaning pipeline running on Kaggle. The cleaning steps are grouped into three categories.

**A. Quality Filtering (Heuristic-Based)**
These are fast rule-based checks applied to every document before any heavy processing.
* **Short Document Filtering:** Removed documents that were too short (< 50 chars) to provide any meaningful training signal.
* **Language Identification:** Used the FastText language detection model (`lid.176.bin`) to filter out any document that was not confidently identified as Hindi (threshold 0.5). This removed content from other languages that accidentally crept in through mixed-language web pages.
* **Post-LangID Latin Strip:** Regex removal of Latin words ≥ 3 chars, then short-doc re-check.

**B. Content Quality Filtering**
These checks target the internal quality of each document's text.
* **Unicode Normalization (NFC):** Canonical decomposition + composition.
* **Markup Stripping:** Removed HTML tags, wiki templates, refs, comments, categories, headings, and URLs.
* **Control Character Removal:** Stripped `\x00-\x08`, `\x0b`, `\x0c`, `\x0e-\x1f`, `\x7f-\x9f`.
* **Devanagari Normalization:** Removed ZWJ, ZWNJ, ZWSP, LRM, RLM, BOM.
* **Whitespace Normalization:** Converted tabs to space, multi-spaces to single, and triple-newlines to double.

**C. Deduplication**
Once the per-document quality filters were applied, two deduplication stages were performed to remove repeated content across documents.
* **Exact Deduplication:** Removed any document whose text was an exact character-for-character match (using SHA-256 hash) with another document already seen.
* **Near-Duplicate Deduplication (MinHash LSH):** Used MinHash signatures and Locality Sensitive Hashing (LSH, Jaccard threshold 0.8, 64 effective permutations) to detect and remove near-duplicate documents.

**Final Statistics After Cleaning:**
| Metric | Value |
|--------|-------|
| Input Documents | ~9.13M |
| Output Documents | 7,745,675 |
| Total Removed | ~1.38M |
| Final Token Count | 585,640,696 |
| — Manual (Wiki + Scraping) | 100,118,768 tokens |
| — Public (IndicCorp) | 485,521,928 tokens |

**Train / Val / Test Split (96% / 2% / 2%):**
| Split | Documents | Tokens |
|-------|-----------|--------|
| Train | 7,435,848 | 562,216,799 |
| Val   | 154,913   | 11,701,875 |
| Test  | 154,914   | 11,722,022 |

**Cleaned Data (Google Drive):** [Hindi Cleaned Dataset](https://drive.google.com/drive/folders/1Sy6n7GJ9b8NkpcOOrirtaOGq1cF0ZQwS?usp=sharing)  
**Splits Data (Google Drive):** [Hindi Splits](https://drive.google.com/drive/folders/1lk482TkUqp3D8jz2JK6ngtQrGZ7Eibbw?usp=sharing)

#### Tokenization

A SentencePiece Unigram tokenization algorithm was trained and evaluated on the cleaned Hindi dataset. The algorithm was trained on a sampled training corpus (1,000,000 sentences). Five vocabulary sizes were tested: **8,000 / 16,000 / 24,000 / 32,000 / 48,000**.

The tokenizer was evaluated on the **validation set** using two metrics:
* **Fertility:** The average number of subword tokens produced per word. A lower fertility means the tokenizer is more efficient (recognises longer, more meaningful chunks).
* **Unknown Rate (%):** The percentage of tokens that are mapped to the `[UNK]` (unknown) token. A lower rate is better, as it means the tokenizer has good coverage of the vocabulary.

**Algorithm: Unigram**
Unigram uses a probabilistic model to find the subword vocabulary that maximises the likelihood of the training corpus. It is better suited for morphologically rich languages like Hindi.

| Vocab Size | Exact Fertility | Unknown Rate (%) |
|------------|-----------------|------------------|
| 8,000      | 1.380           | 0.1508           |
| 16,000     | 1.285           | 0.1620           |
| 24,000     | 1.248           | 0.1668           |
| 32,000     | 1.229           | 0.1694           |
| 48,000     | 1.209           | 0.1721           |

### Final Tokenizer Selection & Rationale

**Selected Tokenizer:** Unigram with an 8,000 Vocabulary Size

**Rationale:**
1. **Unknown Rate Optimization:** The primary selection criteria was finding the smallest size achieving an UNK rate < 0.1%. Since none achieved this (due to `byte_fallback=False`), the size with the absolute lowest UNK rate was chosen. Interestingly, UNK rate increases with larger vocab sizes because larger vocabs have lower character coverage utilization.
2. **Parameter Budget Constraints:** A smaller vocabulary saves parameters in the embedding layer. The 8K model strikes the best balance.
3. **Tokenization Efficiency:** Unigram-8,000 achieves a reasonable fertility of 1.380.

#### Validation Set Statistics (Unigram 8,000)
* **Average Characters per Token:** 3.0044
* **Vocab Coverage:** 95.93%
* **Top 5 Frequent Tokens:** `।`, `▁के`, `▁में`, `▁है`, `,`
* **Tokenization Examples:**
  1. **Original:** `उपयोग....`
     * **Tokens:** `['▁उपयोग', '.']`
  2. **Original:** `मुख्यमंत्री निवास में...`
     * **Tokens:** `['▁मुख्यमंत्री', '▁निवास', '▁में', '▁आज', '▁', 'ती', 'जा', '-', 'पो', 'रा']`

---

### Low Resource Language: Nepali

For the low-resource language, I selected **Nepali**. Nepali has a significantly smaller digital footprint compared to Hindi, making data collection more challenging and requiring multiple targeted sources to reach the token target.

#### 1. Manual Data Collection
* **Wikipedia Nepali:** 
  * **Extracted Tokens:** 5,730,251
* **Wiktionary Nepali:**
  * **Extracted Tokens:** 1,271,639
* **Wikibooks Nepali:**
  * **Extracted Tokens:** 319,953
* **OCR Nepali:**
  * **Extracted Tokens:** 347,449

#### 2. Web-Scraped Data Collection
Since Nepali has limited public corpora, custom scrapers for major Nepali news websites were built to supplement the data.
* **Scraping (Setopati, BBC Nepali):** 
  * **Extracted Tokens:** 2,498,421
* **Scraping (Custom):** 
  * **Extracted Tokens:** 45,860,732

#### 3. Public Data Collection
* **IndicCorp v2 (Nepali):** Used the downloaded Nepali subset of IndicCorp v2. This was by far the largest single source.
  * **Extracted Tokens:** 455,413,217

**Raw Data (Google Drive):** [Nepali Raw Dataset](https://drive.google.com/drive/folders/1HzfrB5oEoGhBO_pJwf8dyabl5sp8DC2j?usp=sharing)

#### Data Cleaning

The combined raw corpus of **~14.96M documents** was passed through the same multi-stage cleaning pipeline used for Hindi.

**A. Quality Filtering (Heuristic-Based)**
* **Short Document Filtering:** Removed documents that were too short (< 50 chars) to be useful.
* **Language Identification:** FastText language detection filtered out non-Nepali documents (threshold lowered to 0.3 for low-resource Nepali).
* **Post-LangID Latin Strip:** Regex removal of Latin words ≥ 3 chars, then short-doc re-check.

**B. Content Quality Filtering**
* **Unicode Normalization (NFC):** Canonical decomposition + composition.
* **Markup Stripping:** Removed HTML tags, wiki templates, refs, comments, categories, headings, and URLs.
* **Control Character Removal:** Stripped control characters.
* **Devanagari Normalization:** Removed ZWJ, ZWNJ, ZWSP, LRM, RLM, BOM.
* **Whitespace Normalization:** Converted tabs to space, multi-spaces to single, and triple-newlines to double.

**C. Deduplication**
* **Exact Deduplication:** Removed character-for-character duplicate documents (merged with combine step to save disk).
* **Near-Duplicate Deduplication (MinHash LSH):** Used MinHash LSH to detect and remove near-duplicate documents. 

**Final Statistics After Cleaning:**
| Metric | Value |
|--------|-------|
| Input Documents | ~14.96M |
| Output Documents | 11,958,680 |
| Total Removed | ~3.00M |
| Final Token Count | 511,441,662 |
| — Manual (Wiki + Scraping + OCR) | 56,028,445 tokens |
| — Public (IndicCorp) | 455,413,217 tokens |

**Train / Val / Test Split (96% / 2% / 2%):**
| Split | Documents | Tokens |
|-------|-----------|--------|
| Train | 11,480,332 | 491,008,773 |
| Val   | 239,173    | 10,232,865 |
| Test  | 239,175    | 10,200,024 |

**Cleaned Data (Google Drive):** [Nepali Cleaned Dataset](https://drive.google.com/drive/folders/1RimXAox4T4FvWhD8cziWafZy509oaW0c?usp=sharing)  
**Splits Data (Google Drive):** [Nepali Splits](https://drive.google.com/drive/folders/15JrX813U_GPfn9sR1tU6551cfknqnAVO?usp=sharing)

#### Tokenization

The SentencePiece Unigram algorithm was trained and evaluated on the cleaned Nepali dataset, using identical hyperparameters and four vocabulary sizes: **4,000 / 8,000 / 16,000 / 24,000**. The evaluation was performed on the Nepali validation set.

**Algorithm: Unigram**

| Vocab Size | Exact Fertility | Unknown Rate (%) |
|------------|-----------------|------------------|
| 4,000      | 1.698           | 0.1098           |
| 8,000      | 1.505           | 0.1506           |
| 16,000     | 1.359           | 0.1669           |
| 24,000     | 1.299           | 0.1746           |

### Final Tokenizer Selection & Rationale

**Selected Tokenizer:** Unigram with a 4,000 Vocabulary Size

**Rationale:**
1. **Unknown Rate Optimization:** The 4,000 vocab size achieved the lowest UNK rate (0.1098%), closest to the < 0.1% target. Similar to Hindi, UNK rate increases with vocab size due to the `byte_fallback=False` setting.
2. **Parameter Budget Constraints:** A 4K model is a very reasonable choice for a lower-resource language, keeping embedding parameters low. 
3. **Tokenization Efficiency:** Nepali fertility values (1.70 at 4K) are higher than Hindi (1.38 at 8K), reflecting that Nepali is morphologically richer, which means Nepali text requires ~23% more tokens per word. 

#### Validation Set Statistics (Unigram 4,000)
* **Average Characters per Token:** 3.2270
* **Vocab Coverage:** 101.15%
* **Top 5 Frequent Tokens:** `को`, `मा`, `▁।`, `,`, `ले`
* **Tokenization Examples:**
  1. **Original:** `नाम`
     * **Tokens:** `['▁नाम']`
  2. **Original:** `‘सेवा र कामका बारेमा...`
     * **Tokens:** `['▁‘', 'सेवा', '▁र', '▁काम', 'का', '▁बारेमा', '▁बुझाउन', '▁केही', '▁कठिन', '▁भएको']`
  3. **Original:** `कोरोना महामारीका कारण...`
     * **Tokens:** `['▁कोरोना', '▁महामारी', 'का', '▁कारण', '▁लकडाउन', '▁भएकाले', '▁उक्त', '▁कम्पनीको', '▁सबै', '▁शाखा']`
