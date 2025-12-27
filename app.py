# =========================================================
# app.py — Trie Autocomplete AKA (FULL VERSION)
# Manual input: N iterasi + prefix uji (tanpa default)
# Output: tabel nilai + 3 grafik (iteratif, rekursif, gabungan)
# Dataset besar: MAX_ROWS slider otomatis sampai total baris
# =========================================================

import re
import time
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

    # ----- ITERATIF -----
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

    # ----- REKURSIF (AMAN) -----
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
# HELPERS
# =========================
def parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return sorted(set([x for x in out if x > 0]))


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Trie Autocomplete AKA", layout="wide")
st.title("Trie Autocomplete — Iteratif vs Rekursif (Input Manual, Dataset Besar)")

st.markdown("""
Aplikasi ini:
1) membaca dataset CSV (kolom wajib: **product_name**),  
2) membangun Trie dari subset data,  
3) menjalankan eksperimen untuk **beberapa ukuran input N** (manual) dan **prefix uji** (manual),  
4) menampilkan **tabel nilai** + **3 grafik**: Iteratif, Rekursif, Gabungan.
""")

uploaded = st.file_uploader("Upload CSV (kolom wajib: product_name)", type=["csv"])
if not uploaded:
    st.info("Upload file CSV dulu untuk mulai.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_and_preprocess(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "product_name" not in df.columns:
        raise ValueError("Kolom 'product_name' tidak ditemukan di CSV.")
    df["clean_name"] = df["product_name"].astype(str).apply(preprocess_text)
    df = df[df["clean_name"].str.len() > 0].reset_index(drop=True)
    return df

try:
    df_full = load_and_preprocess(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

TOTAL_ROWS = len(df_full)

# ===== Sidebar stabilitas (max = total baris dataset) =====
st.sidebar.header("Stabilitas (Dataset Besar)")
MAX_ROWS = st.sidebar.slider(
    f"Batasi jumlah baris dataset (MAX_ROWS) | total={TOTAL_ROWS}",
    min_value=1000,
    max_value=TOTAL_ROWS,
    value=min(50000, TOTAL_ROWS),   # default aman
    step=5000
)

MAX_LEN = st.sidebar.slider(
    "Batasi panjang string (MAX_LEN)",
    min_value=30,
    max_value=200,
    value=80,
    step=10
)

st.sidebar.divider()
st.sidebar.header("Pengaturan Algoritma")
TOP_K = st.sidebar.slider("Top-k suggestions", 1, 50, 10)
MAX_DEPTH = st.sidebar.slider("Batas kedalaman rekursif (max_depth)", 50, 500, 200, step=10)

# Terapkan batasan
df = df_full.head(MAX_ROWS).copy()
words_all = [w[:MAX_LEN] for w in df["clean_name"].tolist()]

st.write(f"Total data di file: **{TOTAL_ROWS}** baris")
st.write(f"Data yang dipakai: **{len(words_all)}** baris (MAX_ROWS={MAX_ROWS}), panjang string dipotong **{MAX_LEN}** karakter")

with st.expander("Preview data (20 baris)", expanded=False):
    st.dataframe(df[["product_name", "clean_name"]].head(20), width="stretch")

st.divider()

# =========================
# INPUT MANUAL: N + PREFIX (tanpa default)
# =========================
st.subheader("Input Manual Eksperimen")

st.markdown("**1) Daftar ukuran input N (dipisah koma)**")
n_text = st.text_input("Contoh: 10,50,100,500,1000", value="")

N_list = parse_int_list(n_text)
if not N_list:
    st.warning("Masukkan minimal 1 nilai N.")
    st.stop()

# filter N agar <= jumlah data yang dipakai
N_list = [N for N in N_list if N <= len(words_all)]
if not N_list:
    st.warning("Semua N lebih besar dari data yang dipakai (MAX_ROWS). Naikkan MAX_ROWS atau kecilkan N.")
    st.stop()

st.markdown("**2) Daftar prefix/kata uji (1 baris = 1 prefix)**")
prefix_text = st.text_area("Masukkan prefix tanpa default (contoh: win\\nlug\\ntsa)", value="", height=140)

prefixes_raw = [line.strip() for line in prefix_text.splitlines() if line.strip()]
prefixes = [preprocess_text(p) for p in prefixes_raw if preprocess_text(p)]
if not prefixes:
    st.warning("Masukkan minimal 1 prefix/kata uji.")
    st.stop()

st.info(f"Eksperimen akan menguji **{len(N_list)} nilai N** × **{len(prefixes)} prefix** = **{len(N_list) * len(prefixes)}** percobaan.")

# =========================
# RUN EXPERIMENT
# =========================
st.subheader("Jalankan Eksperimen")

if st.button("RUN: Bandingkan Iteratif vs Rekursif"):
    rows = []
    total_steps = len(N_list) * len(prefixes)
    step = 0
    prog = st.progress(0)

    # jalankan per N
    for N in N_list:
        with st.spinner(f"Membangun Trie untuk N={N} ..."):
            trie = build_trie(words_all[:N])

        for p in prefixes:
            # Iteratif
            t0 = time.perf_counter()
            _ = trie.suggest_iterative(p, limit=TOP_K)
            t_it = (time.perf_counter() - t0) * 1000

            # Rekursif
            try:
                t1 = time.perf_counter()
                _ = trie.suggest_recursive(p, limit=TOP_K, max_depth=MAX_DEPTH)
                t_rec = (time.perf_counter() - t1) * 1000
            except RecursionError:
                t_rec = float("nan")

            rows.append({
                "N": N,
                "prefix": p,
                "iterative_ms": t_it,
                "recursive_ms": t_rec
            })

            step += 1
            prog.progress(min(step / total_steps, 1.0))

        # release memory
        del trie
        gc.collect()

    res = pd.DataFrame(rows)

    st.subheader("Tabel Nilai (per N dan prefix)")
    st.dataframe(res, width="stretch")

    # ===== agregasi mean per N untuk grafik =====
    agg = res.groupby("N", as_index=False).agg(
        iterative_ms=("iterative_ms", "mean"),
        recursive_ms=("recursive_ms", "mean")
    )

    st.subheader("Grafik 1 — Iteratif (Mean ms) vs N")
    fig1 = plt.figure()
    plt.plot(agg["N"], agg["iterative_ms"], marker="o")
    plt.xlabel("N")
    plt.ylabel("Waktu Iteratif (ms)")
    plt.title("Iteratif: N vs Waktu (Mean over prefixes)")
    plt.grid(True)
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)

    st.subheader("Grafik 2 — Rekursif (Mean ms) vs N")
    fig2 = plt.figure()
    plt.plot(agg["N"], agg["recursive_ms"], marker="o")
    plt.xlabel("N")
    plt.ylabel("Waktu Rekursif (ms)")
    plt.title("Rekursif: N vs Waktu (Mean over prefixes)")
    plt.grid(True)
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    st.subheader("Grafik 3 — Gabungan (Iteratif vs Rekursif) vs N")
    fig3 = plt.figure()
    plt.plot(agg["N"], agg["iterative_ms"], marker="o", label="Iteratif")
    plt.plot(agg["N"], agg["recursive_ms"], marker="o", label="Rekursif")
    plt.xlabel("N")
    plt.ylabel("Waktu (ms)")
    plt.title("Perbandingan: N vs Waktu (Mean over prefixes)")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig3, clear_figure=True)
    plt.close(fig3)

    st.download_button(
        "Download hasil tabel (CSV)",
        data=res.to_csv(index=False).encode("utf-8"),
        file_name="hasil_iterasi_manual.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download agregasi grafik (CSV)",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name="hasil_agregasi_mean_per_N.csv",
        mime="text/csv"
    )

    st.markdown("""
**Catatan untuk laporan:**
- Tabel berisi runtime untuk setiap pasangan (N, prefix).
- Grafik memakai **rata-rata runtime per N** dari semua prefix manual.
- Jika `recursive_ms` menjadi NaN, berarti traversal rekursif melewati batas `max_depth` pada sebagian kasus.
""")
