import re
import time
import random
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================
# PREPROCESS
# =========================
def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# TRIE
# =========================
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

    # Iteratif
    def suggest_iterative(self, prefix: str, limit: int = 10) -> List[str]:
        node = self._find_prefix_node(prefix)
        if node is None:
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

    # Rekursif (aman)
    def suggest_recursive(self, prefix: str, limit: int = 10, max_depth: int = 200) -> List[str]:
        node = self._find_prefix_node(prefix)
        if node is None:
            return []

        res: List[str] = []

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


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Trie Autocomplete AKA", layout="wide")
st.title("Trie Autocomplete — Iteratif vs Rekursif (Multi Prefix + 3 Grafik)")

st.markdown("""
Aplikasi ini:
- membangun **Trie** dari kolom `product_name`,
- menerima **banyak prefix manual**,
- menjalankan autocomplete **Iteratif & Rekursif** secara adil,
- menampilkan **tabel nilai** dan **3 grafik**: Iteratif, Rekursif, Gabungan.
""")

# ---------- Sidebar: stabilizer ----------
st.sidebar.header("Stabilitas & Dataset")
MAX_ROWS = st.sidebar.slider("Batasi jumlah baris dataset (MAX_ROWS)", 200, 20000, 5000, step=200)
MAX_LEN = st.sidebar.slider("Batasi panjang string (MAX_LEN)", 30, 200, 90, step=10)

st.sidebar.divider()
st.sidebar.header("Pengaturan Autocomplete")
top_k = st.sidebar.slider("Top-k suggestions", 1, 50, 10)
max_depth = st.sidebar.slider("Batas kedalaman rekursif (max_depth)", 50, 500, 200, step=10)

st.sidebar.divider()
st.sidebar.header("Benchmark (opsional)")
do_benchmark = st.sidebar.checkbox("Tampilkan menu benchmark N vs waktu", value=False)

uploaded = st.file_uploader("Upload CSV (wajib ada kolom: product_name)", type=["csv"])
if not uploaded:
    st.info("Upload file CSV untuk mulai.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_and_preprocess(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "product_name" not in df.columns:
        raise ValueError("Kolom 'product_name' tidak ditemukan.")
    df["clean_name"] = df["product_name"].astype(str).apply(preprocess_text)
    df = df[df["clean_name"].str.len() > 0].reset_index(drop=True)
    return df

try:
    df = load_and_preprocess(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# Batasi data (biar stabil)
df = df.head(MAX_ROWS).copy()
words_all = [w[:MAX_LEN] for w in df["clean_name"].tolist()]

with st.expander("Preview data (20 baris)", expanded=False):
    st.dataframe(df[["product_name", "clean_name"]].head(20), width="stretch")

st.write(f"Data dipakai: **{len(words_all)}** baris (MAX_ROWS={MAX_ROWS}), panjang string dipotong **{MAX_LEN}** char.")

# ---------- Build Trie hanya saat N berubah ----------
st.subheader("Bangun Trie untuk Demo")
max_demo = min(len(words_all), 5000)
demo_n = st.slider("Ukuran data untuk Trie (N)", 100, max_demo, min(1000, max_demo), step=100)

if "trie_demo" not in st.session_state or st.session_state.get("trie_n") != demo_n or st.session_state.get("max_len") != MAX_LEN:
    with st.spinner(f"Membangun Trie untuk N={demo_n} ..."):
        st.session_state.trie_demo = build_trie(words_all[:demo_n])
        st.session_state.trie_n = demo_n
        st.session_state.max_len = MAX_LEN
        gc.collect()

trie: Trie = st.session_state.trie_demo

# ---------- Multi prefix manual input ----------
st.subheader("Input Prefix Manual (Bisa Banyak)")
st.caption("Masukkan beberapa prefix, 1 baris = 1 prefix. Contoh: win, lug, tsa, blue")

default_prefixes = "win\nlug\ntsa\nblue\ncarry"
prefix_text = st.text_area("Daftar prefix", value=default_prefixes, height=140)

prefixes_raw = [line.strip() for line in prefix_text.splitlines() if line.strip()]
prefixes = [preprocess_text(p) for p in prefixes_raw if preprocess_text(p)]

if not prefixes:
    st.warning("Prefix kosong. Isi dulu minimal 1 prefix.")
    st.stop()

# ---------- Run compare button ----------
if st.button("Jalankan Perbandingan (Iteratif vs Rekursif)"):
    rows = []

    for p in prefixes:
        # iteratif time
        t0 = time.perf_counter()
        sug_it = trie.suggest_iterative(p, limit=top_k)
        t_it = (time.perf_counter() - t0) * 1000

        # rekursif time (try/except)
        try:
            t1 = time.perf_counter()
            sug_rec = trie.suggest_recursive(p, limit=top_k, max_depth=max_depth)
            t_rec = (time.perf_counter() - t1) * 1000
            rec_ok = True
        except RecursionError:
            sug_rec = []
            t_rec = None
            rec_ok = False

        rows.append({
            "prefix": p,
            "iterative_ms": t_it,
            "recursive_ms": t_rec if rec_ok else float("nan"),
            "iterative_suggestions": len(sug_it),
            "recursive_suggestions": len(sug_rec),
            "iterative_sample": " | ".join(sug_it[:3]),
            "recursive_sample": " | ".join(sug_rec[:3]),
        })

    res = pd.DataFrame(rows)

    st.subheader("Tabel Nilai (Runtime per Prefix)")
    st.dataframe(res, width="stretch")

    # ---------- 3 GRAFIK ----------
    # 1) Grafik Iteratif
    st.subheader("Grafik 1 — Iteratif (ms) per Prefix")
    fig1 = plt.figure()
    plt.plot(res["prefix"], res["iterative_ms"], marker="o")
    plt.xlabel("Prefix")
    plt.ylabel("Waktu Iteratif (ms)")
    plt.title("Runtime Autocomplete Iteratif per Prefix")
    plt.grid(True)
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)

    # 2) Grafik Rekursif
    st.subheader("Grafik 2 — Rekursif (ms) per Prefix")
    fig2 = plt.figure()
    plt.plot(res["prefix"], res["recursive_ms"], marker="o")
    plt.xlabel("Prefix")
    plt.ylabel("Waktu Rekursif (ms)")
    plt.title("Runtime Autocomplete Rekursif per Prefix")
    plt.grid(True)
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    # 3) Grafik Gabungan (Overlay)
    st.subheader("Grafik 3 — Gabungan (Iteratif vs Rekursif) per Prefix")
    fig3 = plt.figure()
    plt.plot(res["prefix"], res["iterative_ms"], marker="o", label="Iteratif")
    plt.plot(res["prefix"], res["recursive_ms"], marker="o", label="Rekursif")
    plt.xlabel("Prefix")
    plt.ylabel("Waktu (ms)")
    plt.title("Perbandingan Runtime: Iteratif vs Rekursif")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig3, clear_figure=True)
    plt.close(fig3)

    # download result
    st.download_button(
        "Download hasil perbandingan (CSV)",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="hasil_perbandingan_prefix.csv",
        mime="text/csv"
    )

    st.markdown("""
**Interpretasi cepat (untuk laporan):**
- Grafik Iteratif & Rekursif menunjukkan runtime untuk setiap prefix.
- Grafik Gabungan memudahkan melihat prefix mana yang lebih “berat” di traversal.
- Secara teori Big-O sama, perbedaan biasanya dari overhead stack vs stack manual serta ukuran subtree prefix.
""")

# =========================================================
# OPTIONAL: Benchmark N vs waktu (batch)
# =========================================================
if do_benchmark:
    st.divider()
    st.subheader("Benchmark N vs Waktu (Opsional)")
    st.caption("Ini untuk memenuhi bagian tugas: uji berbagai ukuran masukan dan gambar grafik N vs waktu.")

    sizes_text = st.text_input("Ukuran N benchmark (koma)", value="100,200,500,1000,2000,3000,5000")
    prefix_tests = st.slider("Jumlah prefix uji per N", 10, 200, 50, step=10)

    def parse_sizes(s: str) -> List[int]:
        out = []
        for part in s.split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return sorted(set([x for x in out if x > 0]))

    sizes = parse_sizes(sizes_text)
    if st.button("Jalankan Benchmark N vs Waktu"):
        rows = []
        for N in sizes:
            if N > len(words_all):
                continue

            t_build0 = time.perf_counter()
            trie_bm = build_trie(words_all[:N])
            build_ms = (time.perf_counter() - t_build0) * 1000

            # prefix pool aman
            prefix_pool = [w[:3] for w in words_all[:N] if len(w) >= 3]
            prefix_pool = list(dict.fromkeys(prefix_pool))
            k = min(prefix_tests, len(prefix_pool))
            if k == 0:
                continue
            pref = random.sample(prefix_pool, k)

            t0 = time.perf_counter()
            for p in pref:
                trie_bm.suggest_iterative(p, limit=top_k)
            it_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            for p in pref:
                trie_bm.suggest_recursive(p, limit=top_k, max_depth=max_depth)
            rec_ms = (time.perf_counter() - t1) * 1000

            rows.append({"N": N, "build_ms": build_ms, "iterative_ms": it_ms, "recursive_ms": rec_ms})

            del trie_bm
            gc.collect()

        bm = pd.DataFrame(rows)
        st.dataframe(bm, width="stretch")

        fig = plt.figure()
        plt.plot(bm["N"], bm["iterative_ms"], marker="o", label="Iteratif")
        plt.plot(bm["N"], bm["recursive_ms"], marker="o", label="Rekursif")
        plt.plot(bm["N"], bm["build_ms"], marker="o", label="Build Trie")
        plt.xlabel("Ukuran Input (N)")
        plt.ylabel("Waktu (ms)")
        plt.title("Benchmark Trie: N vs Waktu")
        plt.grid(True)
        plt.legend()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.download_button(
            "Download hasil benchmark (CSV)",
            data=bm.to_csv(index=False).encode("utf-8"),
            file_name="hasil_benchmark_n.csv",
            mime="text/csv"
        )
