import re
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# 1) Preprocessing
# =========================================================
def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================================================
# 2) Trie
# =========================================================
@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
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

    # ---------- REKURSIF (AMAN) ----------
    def suggest_recursive(self, prefix: str, limit: int = 10, max_depth: int = 200) -> List[str]:
        """
        DFS rekursif dengan pembatasan kedalaman (anti RecursionError).
        Secara konsep tetap rekursif (DFS).
        """
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        results: List[str] = []

        def dfs(curr: TrieNode, path: str, depth: int):
            if depth > max_depth or len(results) >= limit:
                return
            if curr.is_end:
                results.append(prefix + path)
                if len(results) >= limit:
                    return
            for ch, nxt in curr.children.items():
                dfs(nxt, path + ch, depth + 1)
                if len(results) >= limit:
                    return

        dfs(node, "", 0)
        return results

    # ---------- ITERATIF ----------
    def suggest_iterative(self, prefix: str, limit: int = 10) -> List[str]:
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        results: List[str] = []
        stack: List[Tuple[TrieNode, str]] = [(node, "")]

        while stack and len(results) < limit:
            curr, path = stack.pop()
            if curr.is_end:
                results.append(prefix + path)
                if len(results) >= limit:
                    break
            for ch, nxt in curr.children.items():
                stack.append((nxt, path + ch))

        return results


def build_trie(words: List[str]) -> Trie:
    t = Trie()
    for w in words:
        if w:
            t.insert(w)
    return t


# =========================================================
# 3) Benchmark helpers
# =========================================================
def generate_prefixes(words: List[str], n_prefix: int, min_len: int, max_len: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    candidates = [w for w in words if len(w) >= min_len]
    if not candidates:
        return []
    out = []
    for _ in range(n_prefix):
        w = rng.choice(candidates)
        L = rng.randint(min_len, min(max_len, len(w)))
        out.append(w[:L])
    return out


def time_function(fn: Callable[[], None], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def benchmark_for_sizes(
    words_all: List[str],
    sizes: List[int],
    prefix_tests: int,
    suggestion_limit: int,
    prefix_min_len: int,
    prefix_max_len: int,
    repeat: int,
    seed: int
) -> pd.DataFrame:
    rows = []
    for N in sizes:
        if N <= 0 or N > len(words_all):
            break

        words = words_all[:N]

        t_build = time_function(lambda: build_trie(words), repeat=repeat)
        trie = build_trie(words)

        prefixes = generate_prefixes(words, prefix_tests, prefix_min_len, prefix_max_len, seed)

        def work_rec():
            for p in prefixes:
                trie.suggest_recursive(p, limit=suggestion_limit)

        def work_it():
            for p in prefixes:
                trie.suggest_iterative(p, limit=suggestion_limit)

        t_rec = time_function(work_rec, repeat=repeat)
        t_it = time_function(work_it, repeat=repeat)

        rows.append({
            "N": N,
            "build_ms": t_build * 1000,
            "autocomplete_recursive_ms": t_rec * 1000,
            "autocomplete_iterative_ms": t_it * 1000,
            "prefix_tests": len(prefixes),
            "suggestion_limit": suggestion_limit
        })
    return pd.DataFrame(rows)


# =========================================================
# 4) Streamlit UI
# =========================================================
st.set_page_config(page_title="Trie Autocomplete AKA", layout="wide")
st.title("Autocomplete Trie — Iteratif vs Rekursif (Tubes AKA)")
st.markdown(
    "Aplikasi ini membangun **Trie** dari `product_name`, menampilkan **autocomplete**, "
    "dan membandingkan efisiensi traversal **rekursif vs iteratif**."
)

# Sidebar
st.sidebar.header("Pengaturan")
mode = st.sidebar.radio("Mode Traversal", ["Rekursif", "Iteratif"])
suggestion_limit = st.sidebar.slider("Top-k suggestions", 1, 50, 10)

st.sidebar.divider()
st.sidebar.subheader("Benchmark")
sizes_text = st.sidebar.text_input("Ukuran N (pisahkan koma)", value="10,50,100,500,1000,2000,5000")
prefix_tests = st.sidebar.slider("Jumlah prefix uji per N", 10, 300, 50, step=10)
prefix_min_len = st.sidebar.slider("Panjang prefix min", 1, 10, 2)
prefix_max_len = st.sidebar.slider("Panjang prefix max", 2, 20, 6)
repeat = st.sidebar.slider("Repeat timing", 1, 7, 3)
seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 42)

uploaded = st.file_uploader("Upload CSV (kolom wajib: product_name)", type=["csv"])
if not uploaded:
    st.info("Upload file CSV untuk mulai.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_and_preprocess(file) -> pd.DataFrame:
    df0 = pd.read_csv(file)
    if "product_name" not in df0.columns:
        raise ValueError("Kolom 'product_name' tidak ditemukan.")
    df0["clean_name"] = df0["product_name"].astype(str).apply(preprocess_text)
    df0 = df0[df0["clean_name"].str.len() > 0].reset_index(drop=True)
    return df0

try:
    df = load_and_preprocess(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

words_all = df["clean_name"].tolist()
st.write(f"Total data valid: **{len(words_all)}**")

with st.expander("Preview data (20 baris)", expanded=False):
    st.dataframe(df[["product_name", "clean_name"]].head(20), width="stretch")

# ---------------- Demo Autocomplete (STABIL) ----------------
st.subheader("Demo Autocomplete (Single Query)")
max_demo = min(len(words_all), 1500)   # BATASI DEMO (AMAN UNTUK REKURSIF)
demo_n = st.slider("Ukuran data demo (N)", 10, max_demo, min(500, max_demo))

# Bangun Trie hanya saat N berubah
if "trie_demo" not in st.session_state or st.session_state.get("trie_demo_n") != demo_n:
    with st.spinner(f"Membangun Trie untuk N={demo_n} ..."):
        st.session_state.trie_demo = build_trie(words_all[:demo_n])
        st.session_state.trie_demo_n = demo_n

trie_demo: Trie = st.session_state.trie_demo

prefix_in = st.text_input("Ketik prefix", value="win")
prefix = preprocess_text(prefix_in)

if st.button("Cari Suggestions"):
    try:
        start = time.perf_counter()
        if mode == "Rekursif":
            sug = trie_demo.suggest_recursive(prefix, limit=suggestion_limit, max_depth=200)
        else:
            sug = trie_demo.suggest_iterative(prefix, limit=suggestion_limit)
        elapsed_ms = (time.perf_counter() - start) * 1000

        st.metric("Waktu eksekusi (ms)", f"{elapsed_ms:.3f}")
        st.write("Suggestions:")
        st.write(sug if sug else "Tidak ada suggestion.")
    except RecursionError:
        st.error("Traversal rekursif melebihi batas. Kurangi N atau prefix.")

st.divider()

# ---------------- Benchmark ----------------
st.subheader("Benchmark (N vs Waktu)")

def parse_sizes(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return sorted(set([x for x in out if x > 0]))

sizes = parse_sizes(sizes_text)
if not sizes:
    st.error("Ukuran N tidak valid.")
    st.stop()

if st.button("Jalankan Benchmark"):
    with st.spinner("Menjalankan benchmark..."):
        res = benchmark_for_sizes(
            words_all=words_all,
            sizes=sizes,
            prefix_tests=prefix_tests,
            suggestion_limit=suggestion_limit,
            prefix_min_len=prefix_min_len,
            prefix_max_len=prefix_max_len,
            repeat=repeat,
            seed=seed,
        )

    if res.empty:
        st.warning("Hasil kosong. Coba perkecil N.")
        st.stop()

    st.dataframe(res, width="stretch")

    fig = plt.figure()
    plt.plot(res["N"], res["build_ms"], marker="o", label="Build Trie (ms)")
    plt.plot(res["N"], res["autocomplete_recursive_ms"], marker="o", label="Autocomplete Rekursif (ms)")
    plt.plot(res["N"], res["autocomplete_iterative_ms"], marker="o", label="Autocomplete Iteratif (ms)")
    plt.xlabel("Ukuran input (N)")
    plt.ylabel("Waktu (ms)")
    plt.title("Benchmark Trie: Build + Autocomplete")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.download_button(
        "Download hasil benchmark (CSV)",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="hasil_benchmark_trie.csv",
        mime="text/csv"
    )
else:
    st.info("Klik **Jalankan Benchmark** untuk melihat tabel & grafik.")
