import re
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# Preprocessing
# =========================================================
def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# Trie
# =========================================================
@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _find_prefix_node(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    # ----- ITERATIF -----
    def suggest_iterative(self, prefix: str, limit: int = 10) -> List[str]:
        node = self._find_prefix_node(prefix)
        if not node:
            return []

        res = []
        stack: List[Tuple[TrieNode, str]] = [(node, "")]

        while stack and len(res) < limit:
            curr, path = stack.pop()
            if curr.is_end:
                res.append(prefix + path)
                if len(res) >= limit:
                    break
            for ch, nxt in curr.children.items():
                stack.append((nxt, path + ch))
        return res

    # ----- REKURSIF (DIBATASI) -----
    def suggest_recursive(self, prefix: str, limit: int = 10, max_depth: int = 200) -> List[str]:
        node = self._find_prefix_node(prefix)
        if not node:
            return []

        res = []

        def dfs(curr: TrieNode, path: str, depth: int):
            if depth > max_depth or len(res) >= limit:
                return
            if curr.is_end:
                res.append(prefix + path)
                if len(res) >= limit:
                    return
            for ch, nxt in curr.children.items():
                dfs(nxt, path + ch, depth + 1)
                if len(res) >= limit:
                    return

        dfs(node, "", 0)
        return res


def build_trie(words: List[str]) -> Trie:
    t = Trie()
    for w in words:
        if w:
            t.insert(w)
    return t


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="Trie Autocomplete AKA", layout="wide")
st.title("Autocomplete Trie — Iteratif vs Rekursif (Tubes AKA)")

st.markdown("""
Aplikasi ini **membandingkan algoritma Trie autocomplete** menggunakan:
- Traversal **Iteratif**
- Traversal **Rekursif**

Perbandingan dilakukan **dalam satu eksekusi terkontrol** untuk menjaga stabilitas
dan keadilan pengukuran waktu.
""")

# Upload dataset
uploaded = st.file_uploader("Upload CSV (kolom wajib: product_name)", type=["csv"])
if not uploaded:
    st.info("Upload dataset untuk mulai.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    if "product_name" not in df.columns:
        raise ValueError("Kolom 'product_name' tidak ditemukan.")
    df["clean_name"] = df["product_name"].astype(str).apply(preprocess_text)
    df = df[df["clean_name"].str.len() > 0].reset_index(drop=True)
    return df

df = load_data(uploaded)
words_all = df["clean_name"].tolist()

st.write(f"Total data valid: **{len(words_all)}**")

with st.expander("Preview data"):
    st.dataframe(df[["product_name", "clean_name"]].head(20), width="stretch")

# =========================================================
# Demo Autocomplete (AMAN)
# =========================================================
st.subheader("Perbandingan Autocomplete (Satu Prefix)")

MAX_DEMO = min(len(words_all), 1000)
demo_n = st.slider("Ukuran data (N)", 10, MAX_DEMO, min(500, MAX_DEMO))

if "trie_demo" not in st.session_state or st.session_state.get("trie_n") != demo_n:
    with st.spinner("Membangun Trie..."):
        st.session_state.trie_demo = build_trie(words_all[:demo_n])
        st.session_state.trie_n = demo_n

trie = st.session_state.trie_demo

prefix_input = st.text_input("Masukkan prefix", value="win")
prefix = preprocess_text(prefix_input)

k = st.slider("Jumlah suggestion (top-k)", 1, 20, 10)

# ==========================
# SATU TOMBOL PERBANDINGAN
# ==========================
if st.button("Bandingkan Iteratif vs Rekursif"):
    # Iteratif
    t0 = time.perf_counter()
    sug_it = trie.suggest_iterative(prefix, limit=k)
    t_it = (time.perf_counter() - t0) * 1000

    # Rekursif (sekali, aman)
    try:
        t1 = time.perf_counter()
        sug_rec = trie.suggest_recursive(prefix, limit=k, max_depth=200)
        t_rec = (time.perf_counter() - t1) * 1000
        rec_ok = True
    except RecursionError:
        sug_rec = []
        t_rec = None
        rec_ok = False

    st.subheader("Hasil Perbandingan")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Iteratif (ms)", f"{t_it:.3f}")
        st.write("Suggestions:")
        st.write(sug_it if sug_it else "-")

    with col2:
        if rec_ok:
            st.metric("Rekursif (ms)", f"{t_rec:.3f}")
            st.write("Suggestions:")
            st.write(sug_rec if sug_rec else "-")
        else:
            st.error("Traversal rekursif melebihi batas kedalaman.")

    st.markdown("""
**Catatan Akademik:**
- Kedua algoritma memiliki kelas kompleksitas yang sama secara asimtotik.
- Perbedaan waktu disebabkan oleh overhead stack rekursif vs stack manual.
""")


st.divider()

# =========================================================
# Benchmark (Batch, Aman)
# =========================================================
st.subheader("Benchmark Runtime (Batch)")

sizes_text = st.text_input("Ukuran N (pisahkan koma)", value="10,50,100,500,1000")
prefix_tests = st.slider("Jumlah prefix uji per N", 10, 200, 50)

def parse_sizes(s: str) -> List[int]:
    out = []
    for p in s.split(","):
        if p.strip().isdigit():
            out.append(int(p.strip()))
    return sorted(set(out))

sizes = parse_sizes(sizes_text)

if st.button("Jalankan Benchmark"):
    rows = []

    for N in sizes:
        if N > len(words_all):
            continue

        trie_bm = build_trie(words_all[:N])

        prefixes = random.sample(
            [w[:3] for w in words_all[:N] if len(w) >= 3],
            min(prefix_tests, N)
        )

        # Iteratif
        t0 = time.perf_counter()
        for p in prefixes:
            trie_bm.suggest_iterative(p)
        t_it = (time.perf_counter() - t0) * 1000

        # Rekursif
        t1 = time.perf_counter()
        for p in prefixes:
            trie_bm.suggest_recursive(p, max_depth=200)
        t_rec = (time.perf_counter() - t1) * 1000

        rows.append({
            "N": N,
            "Iteratif_ms": t_it,
            "Rekursif_ms": t_rec
        })

    res = pd.DataFrame(rows)
    st.dataframe(res, width="stretch")

    fig = plt.figure()
    plt.plot(res["N"], res["Iteratif_ms"], marker="o", label="Iteratif")
    plt.plot(res["N"], res["Rekursif_ms"], marker="o", label="Rekursif")
    plt.xlabel("Ukuran Input (N)")
    plt.ylabel("Waktu (ms)")
    plt.title("Perbandingan Runtime Autocomplete Trie")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download hasil benchmark",
        data=res.to_csv(index=False).encode(),
        file_name="hasil_benchmark_trie.csv",
        mime="text/csv"
    )
