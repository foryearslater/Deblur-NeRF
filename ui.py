#!/usr/bin/env python3
"""
Deblur-NeRF 增强UI系统 - 完整版
包含参数管理、实时监控、结果对比、模型展示等功能
"""

import streamlit as st
import os
import numpy as np
from pathlib import Path
import subprocess
import time
import sys
import re
import shlex
import html
from textwrap import dedent
from datetime import datetime
from PIL import Image
import psutil
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_IMPORT_ERROR = None
except Exception as exc:
    pd = None
    PANDAS_IMPORT_ERROR = exc

try:
    import plotly.graph_objects as go
    PLOTLY_IMPORT_ERROR = None
except Exception as exc:
    go = None
    PLOTLY_IMPORT_ERROR = exc

# ==================== 页面配置 ====================
PROJECT_TITLE = "基于神经辐射场和运动感知的图像去模糊技术研究"
PROJECT_FOOTER = f"{PROJECT_TITLE} UI"
VISUAL_SERIES_COLORS = ["#0f617d", "#d88b2d", "#1f7a5c", "#be4d3f", "#183a60"]

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义样式 ====================
st.markdown("""
<style>
:root {
    --bg-1: #f5efe4;
    --bg-2: #f6fafc;
    --panel: rgba(255, 255, 255, 0.82);
    --panel-strong: rgba(255, 255, 255, 0.95);
    --panel-border: rgba(20, 50, 77, 0.12);
    --panel-border-strong: rgba(20, 50, 77, 0.18);
    --ink-900: #10263f;
    --ink-700: #35536d;
    --ink-500: #6b7f92;
    --brand: #0f617d;
    --brand-deep: #183a60;
    --accent: #d88b2d;
    --accent-soft: #f2e0c3;
    --success-color: #1f7a5c;
    --warning-color: #b97720;
    --danger-color: #be4d3f;
    --shadow-lg: 0 22px 48px rgba(17, 35, 56, 0.12);
    --shadow-md: 0 14px 32px rgba(17, 35, 56, 0.08);
    --shadow-sm: 0 8px 20px rgba(17, 35, 56, 0.05);
    --radius-xl: 28px;
    --radius-lg: 20px;
    --radius-md: 16px;
    --radius-sm: 12px;
}

html, body, [class*="css"] {
    font-family: "Avenir Next", "SF Pro Display", "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    color: var(--ink-900);
}

body {
    color: var(--ink-900);
}

.stApp {
    color: var(--ink-900);
    background:
        radial-gradient(circle at 0% 0%, rgba(216, 139, 45, 0.22), transparent 30%),
        radial-gradient(circle at 100% 12%, rgba(15, 97, 125, 0.18), transparent 28%),
        linear-gradient(180deg, var(--bg-1) 0%, #fbfaf5 38%, var(--bg-2) 100%);
}

.main .block-container {
    max-width: 1380px;
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top, rgba(255,255,255,0.08), transparent 26%),
        linear-gradient(180deg, #193553 0%, #0d2236 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #eef5ff;
}

section[data-testid="stSidebar"] .stCaption {
    color: rgba(238, 245, 255, 0.76) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08);
}

.sidebar-brand {
    position: relative;
    overflow: hidden;
    padding: 1.35rem 1.15rem;
    margin-bottom: 1rem;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 14px 30px rgba(0,0,0,0.16);
}

.sidebar-brand::after {
    content: "";
    position: absolute;
    inset: auto -2rem -2rem auto;
    width: 6rem;
    height: 6rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
}

.sidebar-brand__eyebrow {
    position: relative;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.72);
    margin-bottom: 0.55rem;
}

.sidebar-brand__title {
    position: relative;
    font-size: 1.08rem;
    font-weight: 800;
    line-height: 1.55;
    color: #ffffff;
}

.sidebar-brand__meta {
    position: relative;
    margin-top: 0.6rem;
    font-size: 0.84rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.78);
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 2.35rem 2.5rem;
    margin-bottom: 1.35rem;
    border-radius: var(--radius-xl);
    background:
        radial-gradient(circle at right top, rgba(255,255,255,0.18), transparent 26%),
        linear-gradient(135deg, rgba(15,97,125,0.98) 0%, rgba(24,58,96,0.97) 54%, rgba(216,139,45,0.92) 100%);
    box-shadow: var(--shadow-lg);
    color: #ffffff;
}

.hero-shell::before {
    content: "";
    position: absolute;
    inset: 1.2rem auto auto -1.4rem;
    width: 8rem;
    height: 8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
}

.hero-shell::after {
    content: "";
    position: absolute;
    inset: auto -2rem -2rem auto;
    width: 10rem;
    height: 10rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
}

.hero-shell.compact {
    padding: 1.8rem 2rem;
}

.hero-eyebrow {
    position: relative;
    z-index: 1;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-title {
    position: relative;
    z-index: 1;
    margin: 1rem 0 0.55rem 0;
    font-size: 2.55rem;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.hero-shell.compact .hero-title {
    font-size: 2.05rem;
}

.hero-subtitle {
    position: relative;
    z-index: 1;
    max-width: 960px;
    margin: 0;
    font-size: 1.02rem;
    line-height: 1.85;
    color: rgba(255,255,255,0.88);
}

.hero-tag-row {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 1.2rem;
}

.hero-tag {
    padding: 0.45rem 0.78rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 0.85rem;
    color: #ffffff;
}

.main-title {
    font-size: 2.5rem;
    color: var(--ink-900);
    font-weight: 800;
    margin-bottom: 1rem;
    text-align: center;
}

.section-header {
    margin: 0.5rem 0 1rem 0;
    padding-left: 1rem;
    border-left: 5px solid var(--accent);
    font-size: 1.9rem;
    color: var(--ink-900);
    font-weight: 800;
}

.subsection-header {
    margin: 0.35rem 0 1rem 0;
    padding-left: 0.95rem;
    border-left: 4px solid var(--accent);
    font-size: 1.22rem;
    font-weight: 800;
    color: var(--ink-900);
    letter-spacing: 0.01em;
}

.info-box,
.success-box,
.warning-box,
.error-box {
    padding: 1.15rem 1.2rem;
    border-radius: var(--radius-md);
    margin: 1rem 0;
    border: 1px solid var(--panel-border);
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(8px);
}

.info-box {
    background: linear-gradient(135deg, rgba(238,247,252,0.92) 0%, rgba(251,253,255,0.96) 100%);
    border-left: 5px solid var(--brand);
}

.success-box {
    background: linear-gradient(135deg, rgba(227,244,236,0.94) 0%, rgba(248,252,249,0.98) 100%);
    border-left: 5px solid var(--success-color);
}

.warning-box {
    background: linear-gradient(135deg, rgba(254,245,223,0.95) 0%, rgba(255,252,243,0.98) 100%);
    border-left: 5px solid var(--warning-color);
}

.error-box {
    background: linear-gradient(135deg, rgba(251,232,229,0.95) 0%, rgba(255,247,246,0.98) 100%);
    border-left: 5px solid var(--danger-color);
}

.param-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,249,251,0.92));
    padding: 1rem 1.05rem;
    border-radius: var(--radius-md);
    margin: 0.5rem 0;
    border: 1px solid var(--panel-border);
    box-shadow: var(--shadow-sm);
}

.param-name {
    font-weight: 800;
    color: var(--ink-900);
}

.param-desc {
    font-size: 0.94rem;
    color: var(--ink-700);
    line-height: 1.7;
    margin-top: 0.45rem;
}

.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,249,251,0.9));
    padding: 1rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--panel-border);
    box-shadow: var(--shadow-sm);
    text-align: center;
}

.tab-content {
    padding: 1rem 0;
}

.compare-panel {
    background: linear-gradient(135deg, rgba(255,250,242,0.96) 0%, rgba(255,255,255,0.96) 100%);
    padding: 1.2rem;
    border-radius: var(--radius-lg);
    border: 1px solid var(--panel-border-strong);
    box-shadow: var(--shadow-md);
    margin-bottom: 1rem;
}

.quick-nav-card {
    padding: 0.15rem 0 0.85rem 0;
}

.quick-nav-card__icon {
    font-size: 1.5rem;
    margin-bottom: 0.3rem;
}

.quick-nav-card__title {
    font-size: 1.02rem;
    font-weight: 800;
    color: var(--ink-900);
    margin-bottom: 0.3rem;
}

.quick-nav-card__desc {
    min-height: 4.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--ink-700);
}

.quick-nav-card__meta {
    margin-top: 0.6rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--brand);
}

.soft-panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(247,250,252,0.88));
    border: 1px solid var(--panel-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: 1.15rem 1.2rem;
}

.soft-panel p {
    margin: 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-md);
}

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,250,252,0.9));
    border: 1px solid var(--panel-border);
    padding: 0.75rem 0.8rem;
    border-radius: 18px;
    box-shadow: var(--shadow-sm);
}

div[data-testid="stMetricLabel"] {
    color: var(--ink-700);
}

div[data-testid="stMetricValue"] {
    color: var(--ink-900);
}

section[data-testid="stSidebar"] div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.1);
    box-shadow: none;
}

section[data-testid="stSidebar"] div[data-testid="stMetricLabel"],
section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

div[data-baseweb="tab-list"] {
    gap: 0.55rem;
    padding: 0.35rem;
    background: rgba(255,255,255,0.62);
    border: 1px solid var(--panel-border);
    border-radius: 999px;
    box-shadow: var(--shadow-sm);
    width: fit-content;
}

button[data-baseweb="tab"] {
    height: 42px;
    padding: 0 1rem;
    border-radius: 999px;
    color: var(--ink-700);
    font-weight: 700;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--brand), var(--accent));
    color: #ffffff !important;
}

div.stButton > button {
    width: 100%;
    min-height: 2.95rem;
    border-radius: 14px;
    border: 1px solid rgba(16, 38, 63, 0.12);
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(244,248,250,0.92));
    color: var(--ink-900);
    font-weight: 700;
    box-shadow: var(--shadow-sm);
    transition: all 0.18s ease;
}

div.stButton > button:hover {
    border-color: rgba(15, 97, 125, 0.34);
    color: var(--brand-deep);
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(17, 35, 56, 0.12);
}

button[kind="secondary"] {
    border-radius: 14px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div,
textarea,
input {
    border-radius: 14px !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div {
    background: rgba(255,255,255,0.88);
    border: 1px solid var(--panel-border);
}

div[data-testid="stExpander"] {
    overflow: hidden;
    background: rgba(255,255,255,0.76);
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    box-shadow: var(--shadow-sm);
}

div[data-testid="stExpander"] details summary p {
    font-weight: 700;
    color: var(--ink-900);
}

div[data-testid="stImage"] img,
div[data-testid="stVideo"] video {
    border-radius: 18px;
    border: 1px solid rgba(16, 38, 63, 0.08);
    box-shadow: var(--shadow-md);
}

[data-testid="stMarkdownContainer"] p {
    line-height: 1.75;
}

@media (max-width: 900px) {
    .main .block-container {
        padding-top: 1.4rem;
    }

    .hero-shell,
    .hero-shell.compact {
        padding: 1.45rem 1.25rem;
    }

    .hero-title,
    .hero-shell.compact .hero-title {
        font-size: 1.8rem;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        line-height: 1.75;
    }

    .quick-nav-card__desc {
        min-height: auto;
    }
}

/* ==================== 视觉刷新层 ==================== */
body {
    overflow-x: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        linear-gradient(90deg, rgba(16, 38, 63, 0.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(16, 38, 63, 0.035) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.88), transparent 72%);
}

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(circle at 18% 24%, rgba(255, 255, 255, 0.5), transparent 18%),
        radial-gradient(circle at 84% 72%, rgba(15, 97, 125, 0.08), transparent 24%);
}

[data-testid="stHeader"] {
    background: rgba(251, 249, 244, 0.72);
    backdrop-filter: blur(18px);
}

.main .block-container {
    position: relative;
    z-index: 1;
}

.hero-shell {
    isolation: isolate;
    border: 1px solid rgba(255,255,255,0.18);
    outline: 1px solid rgba(15, 97, 125, 0.08);
}

.hero-shell::before {
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.08),
        0 0 70px rgba(255,255,255,0.22);
}

.hero-shell::after {
    background:
        radial-gradient(circle, rgba(255,255,255,0.14) 0 18%, transparent 19%),
        radial-gradient(circle, rgba(255,255,255,0.08), transparent 58%);
}

.hero-title {
    max-width: 1080px;
    text-shadow: 0 14px 36px rgba(0,0,0,0.18);
}

.hero-subtitle {
    text-wrap: pretty;
}

.hero-tag {
    backdrop-filter: blur(12px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.16);
}

.subsection-header {
    position: relative;
    border-left: 0;
    padding-left: 0;
}

.subsection-header::before {
    content: "";
    display: inline-block;
    width: 0.72rem;
    height: 0.72rem;
    margin-right: 0.55rem;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--accent), var(--brand));
    box-shadow: 0 0 0 6px rgba(216, 139, 45, 0.12);
    vertical-align: 0.08rem;
}

.quick-nav-card {
    position: relative;
    overflow: hidden;
    height: 100%;
    min-height: 188px;
    padding: 1.1rem 1.05rem;
    border-radius: 24px;
    background:
        radial-gradient(circle at 0% 0%, rgba(216,139,45,0.16), transparent 35%),
        linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,250,251,0.86));
    border: 1px solid rgba(16, 38, 63, 0.1);
    box-shadow: var(--shadow-sm);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.quick-nav-card:hover {
    transform: translateY(-3px);
    border-color: rgba(15, 97, 125, 0.22);
    box-shadow: var(--shadow-md);
}

.quick-nav-card::after {
    content: "";
    position: absolute;
    top: -2.4rem;
    right: -2.4rem;
    width: 5.4rem;
    height: 5.4rem;
    border-radius: 999px;
    background: rgba(15, 97, 125, 0.08);
}

.quick-nav-card__icon {
    position: relative;
    z-index: 1;
    display: inline-grid;
    place-items: center;
    width: 2.75rem;
    height: 2.75rem;
    border-radius: 17px;
    background: linear-gradient(135deg, rgba(15, 97, 125, 0.12), rgba(216, 139, 45, 0.18));
}

.quick-nav-card__title,
.quick-nav-card__desc,
.quick-nav-card__meta {
    position: relative;
    z-index: 1;
}

.quick-nav-card__meta {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.quick-nav-card__meta::before {
    content: "";
    width: 1.15rem;
    height: 2px;
    border-radius: 999px;
    background: currentColor;
    opacity: 0.55;
}

.soft-panel {
    position: relative;
    overflow: hidden;
}

.soft-panel::after {
    content: "";
    position: absolute;
    inset: auto 1rem 1rem auto;
    width: 5rem;
    height: 5rem;
    border-radius: 999px;
    background: rgba(15, 97, 125, 0.06);
}

.workflow-strip {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.65rem 0 1.25rem;
}

.workflow-step {
    position: relative;
    overflow: hidden;
    min-height: 132px;
    padding: 1rem;
    border-radius: 22px;
    background: linear-gradient(160deg, rgba(255,255,255,0.9), rgba(246,250,252,0.82));
    border: 1px solid rgba(16,38,63,0.1);
    box-shadow: var(--shadow-sm);
}

.workflow-step::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 4px;
    background: linear-gradient(90deg, var(--brand), var(--accent));
}

.workflow-step__index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    border-radius: 999px;
    background: rgba(16,38,63,0.08);
    color: var(--brand-deep);
    font-size: 0.82rem;
    font-weight: 800;
}

.workflow-step__title {
    margin-top: 0.72rem;
    font-weight: 850;
    color: var(--ink-900);
}

.workflow-step__desc {
    margin-top: 0.38rem;
    color: var(--ink-700);
    font-size: 0.88rem;
    line-height: 1.65;
}

.insight-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.95rem;
    margin: 0.65rem 0 1.2rem;
}

.insight-card {
    position: relative;
    overflow: hidden;
    padding: 1.15rem;
    border-radius: 24px;
    background:
        radial-gradient(circle at 100% 0%, rgba(216,139,45,0.14), transparent 34%),
        linear-gradient(180deg, rgba(255,255,255,0.94), rgba(247,250,251,0.86));
    border: 1px solid rgba(16,38,63,0.1);
    box-shadow: var(--shadow-sm);
}

.insight-card__kicker {
    color: var(--brand);
    font-size: 0.75rem;
    font-weight: 850;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.insight-card__title {
    margin-top: 0.52rem;
    color: var(--ink-900);
    font-size: 1.04rem;
    font-weight: 850;
}

.insight-card__body {
    margin-top: 0.42rem;
    color: var(--ink-700);
    font-size: 0.92rem;
    line-height: 1.72;
}

.empty-state {
    position: relative;
    overflow: hidden;
    padding: 1.25rem 1.35rem;
    border-radius: 24px;
    background:
        radial-gradient(circle at 100% 0%, rgba(15,97,125,0.12), transparent 30%),
        linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,250,251,0.9));
    border: 1px dashed rgba(15, 97, 125, 0.32);
    box-shadow: var(--shadow-sm);
}

.empty-state__title {
    color: var(--ink-900);
    font-weight: 850;
    font-size: 1.06rem;
}

.empty-state__body {
    margin-top: 0.35rem;
    color: var(--ink-700);
    line-height: 1.72;
}

.empty-state__hint {
    margin-top: 0.7rem;
    display: inline-flex;
    border-radius: 999px;
    padding: 0.38rem 0.72rem;
    background: rgba(15, 97, 125, 0.08);
    color: var(--brand-deep);
    font-size: 0.84rem;
    font-weight: 750;
}

.model-facts {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0 0.25rem;
}

.model-fact {
    padding: 0.8rem 0.9rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(16,38,63,0.08);
}

.model-fact__label {
    color: var(--ink-500);
    font-size: 0.78rem;
    font-weight: 750;
}

.model-fact__value {
    margin-top: 0.25rem;
    color: var(--ink-900);
    font-size: 0.96rem;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.image-card-title {
    margin: 0.15rem 0 0.45rem;
    color: var(--ink-900);
    font-weight: 850;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    border-radius: 999px;
    padding: 0.38rem 0.72rem;
    background: rgba(31, 122, 92, 0.1);
    color: var(--success-color);
    font-size: 0.84rem;
    font-weight: 800;
}

.status-pill.warning {
    background: rgba(185, 119, 32, 0.12);
    color: var(--warning-color);
}

.status-pill::before {
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 999px;
    background: currentColor;
    box-shadow: 0 0 0 5px color-mix(in srgb, currentColor 16%, transparent);
}

div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--brand), var(--accent));
}

div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
}

div[data-testid="stRadio"] label {
    color: var(--ink-700);
}

div[role="radiogroup"] {
    gap: 0.45rem;
}

div[role="radiogroup"] label {
    padding: 0.45rem 0.72rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.58);
    border: 1px solid rgba(16,38,63,0.08);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.13);
}

code {
    border-radius: 8px;
}

pre {
    border-radius: 18px !important;
    border: 1px solid rgba(16,38,63,0.08) !important;
}

@media (max-width: 1100px) {
    .workflow-strip,
    .insight-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 760px) {
    .workflow-strip,
    .insight-grid,
    .model-facts {
        grid-template-columns: 1fr;
    }

    .quick-nav-card {
        min-height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================== Session State 初始化 ====================
if 'page' not in st.session_state:
    st.session_state.page = "首页"
if 'current_config' not in st.session_state:
    st.session_state.current_config = {}
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0
if 'sidebar_page' not in st.session_state:
    st.session_state.sidebar_page = st.session_state.page
if 'loaded_model_exp' not in st.session_state:
    st.session_state.loaded_model_exp = None
if 'page_sync_pending' not in st.session_state:
    st.session_state.page_sync_pending = False

# ==================== 参数库和帮助文本 ====================

PARAM_HELP = {
    # 基本参数
    "expname": {
        "desc": "实验名称",
        "help": "用于标识不同的运行实验，影响日志和检查点的存储位置",
        "type": "str",
        "default": "experiment_001"
    },
    "datadir": {
        "desc": "数据目录",
        "help": "包含输入图像和相机参数的目录路径",
        "type": "path",
        "default": "./data/"
    },
    "basedir": {
        "desc": "日志基目录",
        "help": "保存训练输出、检查点和渲染结果的目录",
        "type": "path",
        "default": "./logs/"
    },
    "tbdir": {
        "desc": "TensorBoard日志目录",
        "help": "保存TensorBoard日志用于实时训练监控",
        "type": "path",
        "default": "./tb_logs/"
    },
    "factor": {
        "desc": "降采样因子",
        "help": "对输入图像应用的降采样系数。大的值会更快但精度低",
        "type": "int",
        "default": 4,
        "recommended": [1, 2, 4, 8],
        "tips": "推荐值：4-8用于测试，1-2用于最终结果"
    },
    
    # 网络架构
    "netdepth": {
        "desc": "网络层数（粗采样）",
        "help": "NeRF粗采样网络的深度（MLP层数）",
        "type": "int",
        "default": 8,
        "range": [4, 16],
        "tips": "更深的网络容量更大但速度更慢"
    },
    "netwidth": {
        "desc": "网络宽度（粗采样）",
        "help": "每层的隐层维度数",
        "type": "int",
        "default": 256,
        "range": [64, 512],
        "recommended": [128, 256, 512],
        "tips": "GPU内存不足时减小该值"
    },
    "netdepth_fine": {
        "desc": "网络层数（精细采样）",
        "help": "NeRF精细采样网络的深度",
        "type": "int",
        "default": 8
    },
    "netwidth_fine": {
        "desc": "网络宽度（精细采样）",
        "help": "精细采样网络的隐层维度",
        "type": "int",
        "default": 256
    },
    "use_viewdirs": {
        "desc": "使用视角信息",
        "help": "使用完整的5D输入（位置+视角）而不仅仅是3D位置",
        "type": "bool",
        "default": True,
        "tips": "提高渲染质量，但增加计算量"
    },
    
    # 训练参数
    "N_iters": {
        "desc": "总迭代次数",
        "help": "完整的训练迭代数",
        "type": "int",
        "default": 50000,
        "range": [10000, 200000],
        "tips": "场景复杂度越高需要越多迭代"
    },
    "N_rand": {
        "desc": "每步采样射线数",
        "help": "每个训练步骤采样的随机射线数（批大小）",
        "type": "int",
        "default": 4096,
        "range": [1024, 8192],
        "tips": "受GPU内存限制。内存不足时减小该值"
    },
    "lrate": {
        "desc": "初始学习率",
        "help": "Adam优化器的初始学习率",
        "type": "float",
        "default": 5e-4,
        "range": [1e-5, 1e-3],
        "tips": "推荐值：5e-4，过大易发散，过小收敛慢"
    },
    "lrate_decay": {
        "desc": "学习率衰减周期",
        "help": "学习率衰减的周期（千步为单位）",
        "type": "int",
        "default": 250,
        "tips": "每250k步将学习率降低0.1倍"
    },
    "chunk": {
        "desc": "渲染块大小",
        "help": "并行处理的射线数，影响显存使用",
        "type": "int",
        "default": 32768,
        "tips": "内存不足时减小该值"
    },
    
    # 采样参数
    "N_samples": {
        "desc": "粗采样点数",
        "help": "沿每条射线的粗采样点数",
        "type": "int",
        "default": 64,
        "range": [32, 128],
        "tips": "影响细节程度和计算速度"
    },
    "N_importance": {
        "desc": "精细采样点数",
        "help": "精细采样的额外采样点数",
        "type": "int",
        "default": 0,
        "tips": "0表示仅使用粗采样网络"
    },
    "perturb": {
        "desc": "采样抖动",
        "help": "对采样点的随机扰动（0=无扰动，1=完全扰动）",
        "type": "float",
        "default": 1.0
    },
    
    # 模糊核参数
    "kernel_type": {
        "desc": "模糊核类型",
        "help": "none-原始NeRF | deformablesparsekernel-Deblur-NeRF 稀疏模糊核", 
        "type": "choice",
        "default": "deformablesparsekernel",
        "options": ["none", "deformablesparsekernel"]
    },
    "kernel_ptnum": {
        "desc": "稀疏点数",
        "help": "模糊核中的稀疏采样点数",
        "type": "int",
        "default": 5,
        "range": [3, 10],
        "tips": "越多精度越高，但速度越慢"
    },
    "kernel_hwindow": {
        "desc": "模糊核窗口",
        "help": "物理等效模糊核的大小（像素）",
        "type": "int",
        "default": 10,
        "tips": "根据模糊程度调整"
    },
}

PRESET_CONFIGS = {
    "测试": {
        "factor": 8,
        "N_iters": 10000,
        "N_rand": 2048,
        "chunk": 8192,
        "netwidth": 128,
        "N_samples": 32,
        "desc": "测试，低精度"
    },
    "标准配置": {
        "factor": 4,
        "N_iters": 50000,
        "N_rand": 4096,
        "chunk": 32768,
        "netwidth": 256,
        "N_samples": 64,
        "desc": "平衡质量和速度"
    },
    "高质量": {
        "factor": 2,
        "N_iters": 100000,
        "N_rand": 4096,
        "chunk": 32768,
        "netwidth": 512,
        "N_samples": 128,
        "desc": "最高质量，需要大显存"
    },
    "camera_motion_blur": {
        "kernel_type": "deformablesparsekernel",
        "kernel_ptnum": 5,
        "kernel_hwindow": 10,
        "desc": "相机动模糊优化"
    },
    "defocus_blur": {
        "kernel_type": "deformablesparsekernel",
        "kernel_ptnum": 7,
        "kernel_hwindow": 15,
        "desc": "失焦模糊优化"
    },
}

SUPPORTED_DATASET_TYPES = ["llff"]
DEFAULT_DATASET_TYPE = SUPPORTED_DATASET_TYPES[0]
FORM_MANAGED_CONFIG_KEYS = {
    "expname",
    "datadir",
    "basedir",
    "tbdir",
    "dataset_type",
    "factor",
    "netdepth",
    "netwidth",
    "netdepth_fine",
    "netwidth_fine",
    "use_viewdirs",
    "N_iters",
    "N_rand",
    "lrate",
    "lrate_decay",
    "chunk",
    "N_samples",
    "N_importance",
    "perturb",
    "raw_noise_std",
    "kernel_type",
    "kernel_ptnum",
    "kernel_hwindow",
    "kernel_img_embed",
    "i_print",
    "i_tensorboard",
    "i_weights",
    "i_testset",
    "i_video",
}

# ==================== 核心函数 ====================


def _parse_config_line(raw_line):
    """解析配置行，兼容 key=value、裸 flag 和行内注释。"""
    line = raw_line.split("#", 1)[0].strip()
    if not line:
        return None, None
    if "=" in line:
        key, value = line.split("=", 1)
        return key.strip(), value.strip()
    return line, "True"


def get_config_files():
    """获取所有配置文件"""
    config_dir = Path("configs")
    if not config_dir.exists():
        return []
    files = []
    for f in config_dir.glob("*.txt"):
        files.append(f.name)
    for d in config_dir.iterdir():
        if d.is_dir():
            for f in d.glob("*.txt"):
                files.append(f"{d.name}/{f.name}")
    return sorted(files)

def load_config(config_file):
    """加载配置文件"""
    config_path = Path("configs") / config_file
    if not config_path.exists():
        return {}
    
    config = {}
    try:
        with open(config_path, 'r') as f:
            for raw_line in f:
                key, value = _parse_config_line(raw_line)
                if key is not None:
                    config[key] = value
    except Exception as e:
        st.error(f"❌ 加载配置出错：{e}")
    return config

def save_config(config_data, config_name):
    """保存配置文件"""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    if "/" in config_name:
        subdir = config_dir / config_name.split("/")[0]
        subdir.mkdir(exist_ok=True)
        config_path = subdir / config_name.split("/")[1]
    else:
        config_path = config_dir / config_name
    
    try:
        with open(config_path, 'w') as f:
            f.write("# Deblur-NeRF Configuration\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for key, value in config_data.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                f.write(f"{key} = {value}\n")
        st.success(f"✅ 配置已保存到 {config_path}")
        return True
    except Exception as e:
        st.error(f"❌ 保存配置出错：{e}")
        return False


def _set_page(page_name):
    """更新目标页面，侧边栏状态在下一次 rerun 前同步。"""
    st.session_state.page = page_name
    st.session_state.page_sync_pending = True


def _sync_config_editor_widget_state():
    """在控件实例化前，将配置同步到带 key 的编辑器控件。"""
    current_config = st.session_state.get("current_config", {})
    desired_dataset_type = _normalize_dataset_type(
        current_config.get("dataset_type", DEFAULT_DATASET_TYPE)
    )
    desired_kernel_type = _normalize_kernel_type(
        current_config.get("kernel_type", "deformablesparsekernel")
    )

    if st.session_state.get("dataset_type") != desired_dataset_type:
        st.session_state.dataset_type = desired_dataset_type
    if st.session_state.get("kernel_type") != desired_kernel_type:
        st.session_state.kernel_type = desired_kernel_type


def _to_bool(value, default=False):
    """将配置值安全转换为布尔值"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _normalize_kernel_type(value):
    """兼容 UI 别名与训练脚本真实 kernel_type"""
    normalized = str(value).strip().lower()
    if normalized in {"", "kernel", "deformablesparsekernel"}:
        return "deformablesparsekernel"
    if normalized == "none":
        return "none"
    return normalized


def _normalize_dataset_type(value):
    """当前项目仅支持 llff，其余值回退到默认类型。"""
    normalized = str(value).strip().lower()
    if normalized in SUPPORTED_DATASET_TYPES:
        return normalized
    return DEFAULT_DATASET_TYPE


def _plotly_is_available():
    """检查图表依赖是否可用。"""
    if go is None:
        st.warning(f"图表功能暂不可用：`plotly` 导入失败（{PLOTLY_IMPORT_ERROR}）。")
        return False
    return True


def _format_shell_command(command_args):
    """将参数列表格式化为可复制的 shell 命令"""
    return " ".join(shlex.quote(str(arg)) for arg in command_args)


def _launch_background_process(command_args, log_path):
    """后台启动任务并将输出重定向到日志文件"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", buffering=1) as log_file:
        log_file.write(
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Launching: {_format_shell_command(command_args)}\n"
        )
        process = subprocess.Popen(
            command_args,
            cwd=os.getcwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return process.pid


def get_experiment_config_path(exp_name):
    """返回实验目录中可用于重新渲染的配置文件"""
    exp_dir = get_experiment_dir(exp_name)
    config_path = exp_dir / "config.txt"
    if config_path.exists():
        return config_path
    args_path = exp_dir / "args.txt"
    if args_path.exists():
        return args_path
    return None


def build_training_command(config_path, config_data, *, num_gpus=1, save_ckpt=True, render_testset=True,
                           enable_tb=True, auto_continue=False):
    """根据 UI 选项构建训练命令"""
    command = [
        sys.executable, "run_nerf.py",
        "--config", str(config_path),
        "--num_gpu", str(int(num_gpus)),
    ]

    total_iters = _to_int(config_data.get("N_iters", 50000), 50000)
    i_weights = _to_int(config_data.get("i_weights", 20000), 20000)
    i_testset = _to_int(config_data.get("i_testset", 20000), 20000)
    i_tensorboard = _to_int(config_data.get("i_tensorboard", 200), 200)

    if not auto_continue:
        command.append("--no_reload")

    command.extend([
        "--i_weights", str(i_weights if save_ckpt else total_iters + 1),
        "--i_testset", str(min(i_testset, total_iters) if render_testset else total_iters + 1),
        "--i_tensorboard", str(i_tensorboard if enable_tb else total_iters + 1),
    ])

    return command


def build_inference_command(exp_name, render_mode, render_factor=4, num_gpus=1):
    """构建 render_only 推理命令"""
    config_path = get_experiment_config_path(exp_name)
    if config_path is None:
        return None, "未找到可用于重新渲染的 config.txt 或 args.txt，无法自动发起推理。"

    command = [
        sys.executable, "run_nerf.py",
        "--config", str(config_path),
        "--render_only",
        "--render_factor", str(int(render_factor)),
        "--num_gpu", str(int(num_gpus)),
    ]

    if render_mode == "测试集":
        command.append("--render_test")
    elif render_mode == "EPI路径":
        command.append("--render_epi")

    return command, None


def validate_config(config_data):
    """验证配置参数"""
    warnings = []
    errors = []
    
    # 检查必需参数
    required = ["expname", "datadir", "basedir"]
    for key in required:
        if not config_data.get(key):
            errors.append(f"缺少必需参数：{key}")
    
    # 检查参数范围
    try:
        N_rand = int(config_data.get("N_rand", 4096))
        if N_rand > 8192:
            warnings.append("N_rand过大，可能导致显存不足")
        
        netwidth = int(config_data.get("netwidth", 256))
        if netwidth > 512:
            warnings.append("网络宽度较大，需要足够的显存")
        
        factor = int(config_data.get("factor", 4))
        if factor < 1:
            errors.append("factor必须 >= 1")

        dataset_type = str(config_data.get("dataset_type", DEFAULT_DATASET_TYPE)).strip().lower()
        if dataset_type not in SUPPORTED_DATASET_TYPES:
            errors.append("当前项目仅支持 LLFF 数据集（dataset_type = llff）")

        kernel_type = _normalize_kernel_type(config_data.get("kernel_type", "deformablesparsekernel"))
        if kernel_type not in {"none", "deformablesparsekernel"}:
            errors.append("kernel_type 仅支持 none 或 deformablesparsekernel")

        datadir = config_data.get("datadir", "")
        if datadir and not Path(datadir).exists():
            warnings.append(f"数据目录当前不存在：{datadir}")

    except ValueError as e:
        errors.append(f"参数类型错误：{e}")
    
    return errors, warnings

def get_experiments():
    """获取所有实验"""
    return [record["id"] for record in get_experiment_records()]

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                return {
                    'name': parts[0].strip(),
                    'total': float(parts[1].strip()),
                    'used': float(parts[2].strip()),
                    'util': float(parts[3].strip())
                }
    except:
        pass
    return None

def get_system_stats():
    """获取系统状态"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.3)
        memory = psutil.virtual_memory()
        return {
            'cpu': cpu_percent,
            'memory': memory.percent,
            'memory_used': memory.used / (1024**3),
            'memory_total': memory.total / (1024**3),
        }
    except:
        return {}

def get_experiment_stats(exp_name):
    """获取实验统计"""
    exp_dir = get_experiment_dir(exp_name)
    if not exp_dir.exists():
        return None
    
    stats = {
        'name': get_experiment_name(exp_name),
        'created': datetime.fromtimestamp(exp_dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        'modified': datetime.fromtimestamp(exp_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        'size': 0,
        'ckpt_count': 0,
        'images_count': 0,
        'config_file': None
    }
    
    try:
        for item in exp_dir.rglob("*"):
            if item.is_file():
                stats['size'] += item.stat().st_size
                if item.suffix in {'.pt', '.tar'} or 'ckpt' in item.name:
                    stats['ckpt_count'] += 1
                elif item.suffix in ['.png', '.jpg', '.jpeg']:
                    stats['images_count'] += 1
                elif item.name.endswith('.txt') and 'config' in item.name.lower():
                    stats['config_file'] = str(item)
    except:
        pass
    
    stats['size_mb'] = stats['size'] / (1024**2)
    return stats


def _extract_latest_iter_from_ckpt(exp_dir: Path):
    """从 *.tar 检查点文件名提取最新迭代数"""
    latest = None
    for ckpt in exp_dir.glob("*.tar"):
        m = re.match(r"^(\d+)\.tar$", ckpt.name)
        if not m:
            continue
        it = int(m.group(1))
        latest = it if latest is None else max(latest, it)
    return latest


def parse_latest_train_iter_from_log(exp_name):
    """从 training.log 中提取最近一次训练迭代"""
    log_path = get_experiment_log_path(exp_name)
    if not log_path.exists():
        return None

    latest_iter = None
    iter_pattern = re.compile(r"\[TRAIN\]\s+Iter:\s*(\d+)")
    try:
        with open(log_path, "r") as f:
            for line in f:
                match = iter_pattern.search(line)
                if match:
                    latest_iter = int(match.group(1))
    except Exception:
        return None
    return latest_iter


def detect_training_log_issue(exp_name):
    """从 training.log 中查找明显异常线索"""
    log_path = get_experiment_log_path(exp_name)
    if not log_path.exists():
        return None

    suspicious_patterns = (
        "traceback",
        "runtimeerror",
        "outofmemoryerror",
        "cuda out of memory",
        "error:",
    )
    latest_issue = None
    try:
        with open(log_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                lowered = line.lower()
                if line and any(pattern in lowered for pattern in suspicious_patterns):
                    latest_issue = line
    except Exception:
        return None
    return latest_issue


def parse_test_metrics(exp_name):
    """解析 logs/<exp>/test_metrics.txt 中的 iter 与 PSNR"""
    metric_records = parse_test_metric_records(exp_name)
    return sorted(
        [
            (record["iter"], record["psnr"])
            for record in metric_records
            if record.get("psnr") is not None
        ],
        key=lambda x: x[0]
    )


def parse_test_metric_records(exp_name):
    """解析 logs/<exp>/test_metrics.txt 中的完整评估指标"""
    metric_file = get_experiment_dir(exp_name) / "test_metrics.txt"
    if not metric_file.exists():
        return []

    records = []
    iter_pattern = re.compile(r"iter(\d+)")
    value_patterns = {
        "mse": re.compile(r"MSE:([0-9eE+\-.]+)"),
        "psnr": re.compile(r"PSNR:([0-9eE+\-.]+)"),
        "ssim": re.compile(r"SSIM:([0-9eE+\-.]+)"),
        "lpips": re.compile(r"LPIPS:([0-9eE+\-.]+)"),
    }

    try:
        with open(metric_file, "r") as f:
            for line in f:
                iter_match = iter_pattern.search(line)
                if not iter_match:
                    continue

                record = {"iter": int(iter_match.group(1))}
                for key, pattern in value_patterns.items():
                    match = pattern.search(line)
                    record[key] = float(match.group(1)) if match else None
                records.append(record)
    except Exception:
        return []

    return sorted(records, key=lambda x: x["iter"])


def get_latest_metric_record(exp_name):
    """返回实验最新一条指标记录"""
    records = parse_test_metric_records(exp_name)
    if not records:
        return None
    return records[-1]


def infer_training_status(exp_name):
    """基于实验目录推断训练状态"""
    exp_dir = get_experiment_dir(exp_name)
    if not exp_dir.exists():
        return {
            "label": "未开始",
            "detail": "实验目录不存在",
            "progress_percent": 0,
            "latest_iter": 0,
            "target_iters": 0,
            "is_running_hint": False,
        }

    stats = get_experiment_stats(exp_name) or {}
    args_data = get_experiment_args(exp_name)
    target_iters = _to_int(args_data.get("N_iters", 0), 0)
    latest_ckpt_iter = _extract_latest_iter_from_ckpt(exp_dir) or 0
    latest_log_iter = parse_latest_train_iter_from_log(exp_name) or 0
    latest_iter = max(latest_ckpt_iter, latest_log_iter)
    progress_percent = int(min(100, (latest_iter / target_iters * 100))) if target_iters > 0 else 0
    if latest_iter > 0 and progress_percent == 0 and target_iters > 0:
        progress_percent = 1

    has_outputs = (stats.get("ckpt_count", 0) > 0) or (stats.get("images_count", 0) > 0)
    has_log = get_experiment_log_path(exp_name).exists()
    log_issue = detect_training_log_issue(exp_name)

    if latest_ckpt_iter > 0:
        label = "训练中/已训练"
        detail = f"检测到最新检查点: iter={latest_ckpt_iter}"
        is_running_hint = True
    elif has_outputs:
        label = "已产出结果"
        detail = "检测到图像输出，但尚未检测到标准检查点文件"
        is_running_hint = True
    elif log_issue:
        label = "训练中断/日志异常"
        detail = f"训练日志显示异常: {log_issue}"
        is_running_hint = False
    elif latest_log_iter > 0:
        label = "训练进行中（尚未保存检查点）"
        detail = f"日志显示最近训练到 iter={latest_log_iter}，但尚未到首次保存检查点/测试图的时机"
        is_running_hint = True
    elif has_log:
        label = "训练已启动/初始化中"
        detail = "已检测到 training.log，但尚未出现训练迭代、检查点或测试图像"
        is_running_hint = True
    else:
        label = "未检测到有效训练输出"
        detail = "当前实验目录无检查点与渲染图像"
        is_running_hint = False

    return {
        "label": label,
        "detail": detail,
        "progress_percent": progress_percent,
        "latest_iter": latest_iter,
        "target_iters": target_iters,
        "is_running_hint": is_running_hint,
        "latest_log_iter": latest_log_iter,
        "has_log": has_log,
    }

def list_results(exp_name):
    """列出实验结果"""
    exp_dir = get_experiment_dir(exp_name)
    result_dirs = [
        exp_dir / "renderonly_test_000000",
        exp_dir / "videos",
        exp_dir / "results",
        exp_dir
    ]
    
    images = []
    for result_dir in result_dirs:
        if result_dir.exists():
            images.extend(sorted([f for f in result_dir.glob("*.png")]))
            images.extend(sorted([f for f in result_dir.glob("*.jpg")]))
    
    return list(set(images))  # 去重


def _parse_kv_file(file_path):
    """解析 key = value 文本文件"""
    data = {}
    path = Path(file_path)
    if not path.exists():
        return data
    try:
        with open(path, "r") as f:
            for raw in f:
                key, value = _parse_config_line(raw)
                if key is not None:
                    data[key] = value
    except Exception:
        return {}
    return data


def _display_path(path_obj):
    """尽量以相对路径展示目录，避免 UI 太长"""
    path_obj = Path(path_obj)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path_obj)


def _build_experiment_record(exp_dir):
    """根据实验目录构建实验元数据"""
    exp_dir = Path(exp_dir).expanduser()
    if not exp_dir.exists() or not exp_dir.is_dir():
        return None

    args_data = _parse_kv_file(exp_dir / "args.txt")
    try:
        modified_ts = exp_dir.stat().st_mtime
    except OSError:
        modified_ts = 0

    return {
        "id": str(exp_dir.resolve()),
        "name": args_data.get("expname", exp_dir.name),
        "path": exp_dir,
        "basedir": exp_dir.parent,
        "modified_ts": modified_ts,
    }


def get_known_basedirs():
    """从默认目录与配置文件中收集可能的实验 basedir"""
    candidate_dirs = [Path("./logs")]
    for config_name in get_config_files():
        config_data = _parse_kv_file(Path("configs") / config_name)
        basedir = str(config_data.get("basedir", "")).strip()
        if basedir:
            candidate_dirs.append(Path(basedir).expanduser())

    basedirs = []
    seen = set()
    for basedir in candidate_dirs:
        key = str(basedir.resolve())
        if key in seen:
            continue
        seen.add(key)
        basedirs.append(basedir)
    return basedirs


def get_experiment_records():
    """发现所有实验目录，并生成展示信息"""
    records = []
    seen = set()

    for basedir in get_known_basedirs():
        if not basedir.exists() or not basedir.is_dir():
            continue
        try:
            exp_dirs = [path for path in basedir.iterdir() if path.is_dir()]
        except Exception:
            continue

        for exp_dir in exp_dirs:
            record = _build_experiment_record(exp_dir)
            if not record or record["id"] in seen:
                continue
            seen.add(record["id"])
            records.append(record)

    name_counts = defaultdict(int)
    for record in records:
        name_counts[record["name"]] += 1

    for record in records:
        basedir_label = _display_path(record["basedir"])
        if name_counts[record["name"]] > 1:
            record["label"] = f"{record['name']} [{basedir_label}]"
        else:
            record["label"] = record["name"]

    return sorted(records, key=lambda record: record["modified_ts"], reverse=True)


def get_experiment_record(exp_ref):
    """按内部 ID、展示名或目录路径解析实验"""
    if exp_ref is None:
        return None

    if isinstance(exp_ref, dict) and "path" in exp_ref:
        return exp_ref

    for record in get_experiment_records():
        if exp_ref in {record["id"], record["name"], record.get("label")}:
            return record

    candidate_path = Path(str(exp_ref)).expanduser()
    if candidate_path.exists() and candidate_path.is_dir():
        record = _build_experiment_record(candidate_path)
        if record:
            record["label"] = record["name"]
            return record
    return None


def get_experiment_dir(exp_ref):
    """返回实验目录"""
    record = get_experiment_record(exp_ref)
    if record:
        return record["path"]
    return Path("./logs") / str(exp_ref)


def get_experiment_name(exp_ref):
    """返回实验短名称"""
    record = get_experiment_record(exp_ref)
    if record:
        return record["name"]
    return Path(str(exp_ref)).name


def get_experiment_display_name(exp_ref):
    """返回适合 UI 展示的实验名称"""
    record = get_experiment_record(exp_ref)
    if record:
        return record.get("label", record["name"])
    return get_experiment_name(exp_ref)


def get_experiment_widget_key(exp_ref):
    """根据实验标识生成稳定、安全的 widget key 片段"""
    record = get_experiment_record(exp_ref)
    raw_key = record["id"] if record else str(exp_ref)
    return re.sub(r"[^\w]+", "_", raw_key).strip("_")


def get_experiment_args(exp_name):
    """读取 logs/<exp>/args.txt"""
    args_file = get_experiment_dir(exp_name) / "args.txt"
    return _parse_kv_file(args_file)


def get_experiment_log_path(exp_name):
    """返回实验默认训练日志路径"""
    return get_experiment_dir(exp_name) / "training.log"


def _to_int(value, default=0):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _numeric_stem(path_obj):
    stem = Path(path_obj).stem
    m = re.match(r"^(\d+)$", stem)
    return int(m.group(1)) if m else None


def list_source_images_by_exp(exp_name):
    """根据实验参数定位训练输入图像序列（LLFF）"""
    args_data = get_experiment_args(exp_name)
    datadir = args_data.get("datadir", "")
    factor = _to_int(args_data.get("factor", 1), 1)
    if not datadir:
        return []

    dataset_dir = Path(datadir)
    candidate_dirs = []
    if factor > 1:
        candidate_dirs.append(dataset_dir / f"images_{factor}")
    candidate_dirs.append(dataset_dir / "images")
    if factor == 1:
        candidate_dirs.append(dataset_dir / "images_1")

    exts = {".png", ".jpg", ".jpeg", ".JPG", ".PNG"}
    for img_dir in candidate_dirs:
        if img_dir.exists() and img_dir.is_dir():
            files = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix in exts]
            if files:
                return files
    return []


def list_result_images(exp_name):
    """返回实验可视化结果图（优先最新 testset，其次 renderonly）"""
    exp_dir = get_experiment_dir(exp_name)
    if not exp_dir.exists():
        return []

    # 优先 testset_xxxxxx 最新目录
    testset_dirs = sorted(
        [d for d in exp_dir.glob("testset_*") if d.is_dir()],
        key=lambda d: d.name
    )
    target_dirs = []
    if testset_dirs:
        target_dirs.append(testset_dirs[-1])
    target_dirs.extend(sorted([d for d in exp_dir.glob("renderonly_test_*") if d.is_dir()], key=lambda d: d.name, reverse=True))
    target_dirs.extend(sorted([d for d in exp_dir.glob("renderonly_path_*") if d.is_dir()], key=lambda d: d.name, reverse=True))
    target_dirs.append(exp_dir)

    images = []
    for result_dir in target_dirs:
        if not result_dir.exists():
            continue
        candidates = sorted([*result_dir.glob("*.png"), *result_dir.glob("*.jpg"), *result_dir.glob("*.jpeg")])
        # 排除辅助图
        candidates = [
            p for p in candidates
            if ("_disp" not in p.stem and "_pt" not in p.stem and not p.name.startswith("w_"))
        ]
        if candidates:
            # 数字文件名优先按数字排序
            if all(_numeric_stem(p) is not None for p in candidates):
                candidates = sorted(candidates, key=lambda p: _numeric_stem(p))
            images = candidates
            break

    return images


def list_result_videos(exp_name):
    """返回实验可视化结果视频（优先 render_only 路径渲染，其次训练期间导出视频）"""
    exp_dir = get_experiment_dir(exp_name)
    if not exp_dir.exists():
        return []

    search_dirs = sorted(
        [d for d in exp_dir.glob("renderonly_path_*") if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    search_dirs.append(exp_dir)

    candidates = []
    for result_dir in search_dirs:
        candidates.extend(sorted(result_dir.glob("*.mp4")))
        candidates.extend(sorted(result_dir.glob("*.mov")))
        candidates.extend(sorted(result_dir.glob("*.avi")))

    # 训练过程中导出的 spiral 视频保存在实验根目录，按文件名倒序展示最近结果。
    candidates.extend(sorted(exp_dir.glob("*_spiral_*_rgb.mp4"), reverse=True))
    candidates.extend(sorted(exp_dir.glob("*_spiral_*_disp.mp4"), reverse=True))

    unique_videos = []
    seen = set()
    for video_path in candidates:
        resolved = str(video_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_videos.append(video_path)
    return unique_videos


def match_before_after_images(exp_name):
    """根据文件名编号建立训练前后图像对"""
    before_images = list_source_images_by_exp(exp_name)
    after_images = list_result_images(exp_name)
    if not before_images or not after_images:
        return [], before_images, after_images

    pairs = []
    # 常见情况：after 为 000.png 编号，与数据集索引一致
    for after_path in after_images:
        idx = _numeric_stem(after_path)
        if idx is not None and 0 <= idx < len(before_images):
            pairs.append((before_images[idx], after_path))

    # 回退：按顺序对齐
    if not pairs:
        pair_len = min(len(before_images), len(after_images))
        pairs = [(before_images[i], after_images[i]) for i in range(pair_len)]

    return pairs, before_images, after_images


def build_image_pair_records(exp_name):
    """构建可供场景视角浏览的前后图记录"""
    pairs, before_images, after_images = match_before_after_images(exp_name)
    records = []
    for idx, (before_path, after_path) in enumerate(pairs):
        records.append({
            "index": idx,
            "before_path": before_path,
            "after_path": after_path,
            "before_name": before_path.name,
            "after_name": after_path.name,
            "label": f"{idx:03d} | {before_path.name} -> {after_path.name}",
        })
    return records, before_images, after_images


def render_before_after_compare(exp_name, widget_key_prefix="compare"):
    """渲染训练前后图像对比组件"""
    pairs, before_images, after_images = match_before_after_images(exp_name)

    if not after_images:
        st.info("未找到渲染结果图。请先训练并生成 testset 或 renderonly 输出。")
        return
    if not before_images:
        st.info("未找到数据集原图。请检查该实验的 `args.txt` 中 datadir/factor 配置。")
        return
    if not pairs:
        st.info("找到了图像，但无法建立前后对应关系。")
        return

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            st.metric("原图数量", len(before_images))
        with col_b:
            st.metric("结果数量", len(after_images))
        with col_c:
            st.metric("可对比对数", len(pairs))

        idx = st.slider(
            "选择图像索引",
            min_value=0,
            max_value=len(pairs) - 1,
            value=0,
            key=f"{widget_key_prefix}_img_idx"
        )
        alpha = st.slider(
            "融合滑块（0=原图，1=训练后）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=f"{widget_key_prefix}_alpha"
        )

    before_path, after_path = pairs[idx]
    try:
        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")
        if before_img.size != after_img.size:
            after_img = after_img.resize(before_img.size, Image.BICUBIC)
        blend_img = Image.blend(before_img, after_img, alpha=alpha)

        diff_np = np.abs(np.array(after_img, dtype=np.int16) - np.array(before_img, dtype=np.int16)).astype(np.uint8)
        diff_img = Image.fromarray(diff_np)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**训练前**: `{before_path.name}`")
            st.image(before_img, width='stretch')
            st.markdown(f"**训练后**: `{after_path.name}`")
            st.image(after_img, width='stretch')
        with col2:
            st.markdown("**融合对比**")
            st.image(blend_img, width='stretch')
            st.markdown("**绝对差分图**")
            st.image(diff_img, width='stretch')
    except Exception as e:
        st.error(f"图像读取失败: {e}")


def model_loader_page():
    """模型加载与场景结果查看页面"""
    render_section_intro(
        "🧠 模型加载与结果查看",
        "按照 Deblur-NeRF 的场景级工作流，先选择已训练实验，再浏览指定视角的原图、重建结果、融合效果和差分细节。",
        tags=["场景模型", "视角浏览", "结果核验"],
    )

    render_info_box(
        "<strong>页面说明</strong><br>"
        "这里遵循 Deblur-NeRF 原始项目的工作方式：先加载一个已经训练完成的场景模型，再选择该场景中的某个视角，查看原图与模型渲染结果。<br>"
        "<strong>注意</strong>: 当前项目是场景级 NeRF，不做任意单张陌生图片的通用去模糊推理。"
    )

    exps = get_experiments()
    if not exps:
        render_empty_state(
            "暂无可加载的训练模型",
            "完成至少一次训练后，模型实验会出现在这里，用于按场景视角查看恢复效果。",
            "先进入训练页启动一个实验",
        )
        return

    selected_exp = st.selectbox(
        "选择模型实验",
        exps,
        format_func=get_experiment_display_name,
        key="model_loader_exp_select"
    )

    exp_widget_key = get_experiment_widget_key(selected_exp)
    stats = get_experiment_stats(selected_exp) or {}
    status = infer_training_status(selected_exp)
    args_data = get_experiment_args(selected_exp)
    exp_dir = get_experiment_dir(selected_exp)
    config_path = get_experiment_config_path(selected_exp)
    latest_metrics = get_latest_metric_record(selected_exp) or {}
    source_images = list_source_images_by_exp(selected_exp)
    result_images = list_result_images(selected_exp)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("模型名称", get_experiment_name(selected_exp))
    with col2:
        st.metric("检查点数", stats.get("ckpt_count", 0))
    with col3:
        st.metric("场景原图数", len(source_images))
    with col4:
        st.metric("状态", status["label"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("结果图数", len(result_images))
    with col2:
        latest_iter_text = str(status["latest_iter"]) if status["latest_iter"] else "未检测到"
        st.metric("最新迭代", latest_iter_text)
    with col3:
        psnr_text = f"{latest_metrics['psnr']:.3f} dB" if latest_metrics.get("psnr") is not None else "N/A"
        st.metric("最新 PSNR", psnr_text)
    with col4:
        ssim_text = f"{latest_metrics['ssim']:.4f}" if latest_metrics.get("ssim") is not None else "N/A"
        st.metric("最新 SSIM", ssim_text)

    with st.container(border=True):
        st.markdown("### 模型信息")
        render_model_fact_grid([
            ("实验目录", exp_dir),
            ("配置文件", config_path if config_path else "未找到"),
            ("数据目录", args_data.get('datadir', 'N/A')),
            ("迭代进度", f"{status['latest_iter']} / {status['target_iters'] or '未知'}"),
        ])

        if st.button("📦 加载这个模型", width='stretch', key=f"load_model_{exp_widget_key}"):
            st.session_state.loaded_model_exp = selected_exp
            st.success(f"✅ 已加载模型：{get_experiment_display_name(selected_exp)}")

    active_model = st.session_state.loaded_model_exp or selected_exp
    st.caption(f"当前已加载模型: {get_experiment_display_name(active_model)}")
    if active_model != selected_exp:
        st.info("当前结果区域仍使用“已加载模型”。如果你想切换到新选择的实验，请点击上方“加载这个模型”。")

    pair_records, before_images, after_images = build_image_pair_records(active_model)
    if not pair_records:
        if not before_images:
            render_empty_state(
                "未找到数据集原图",
                "该实验的 args.txt 中 datadir/factor 可能无法定位到 images 或 images_factor 目录。",
                "检查实验配置的数据路径",
            )
        elif not after_images:
            render_empty_state(
                "未找到渲染结果图",
                "模型已可选择，但尚未检测到 testset 或 renderonly 输出图像。",
                "先在推理页执行测试集渲染",
            )
        else:
            render_empty_state(
                "无法建立图像对应关系",
                "找到了原图和结果图，但文件编号或数量暂时无法自动配对。",
                "可检查输出文件名是否按视角编号排列",
            )
        return

    st.divider()
    st.markdown("### 选择场景视角")

    active_widget_key = get_experiment_widget_key(active_model)
    selected_idx = st.slider(
        "选择视角索引",
        min_value=0,
        max_value=len(pair_records) - 1,
        value=0,
        key=f"model_loader_pair_idx_{active_widget_key}"
    )
    selected_record = pair_records[selected_idx]

    with st.container(border=True):
        st.markdown("### 当前视角信息")
        render_model_fact_grid([
            ("场景模型", get_experiment_display_name(active_model)),
            ("原图文件", selected_record['before_name']),
            ("结果文件", selected_record['after_name']),
            ("视角编号", f"{selected_record['index']} / {len(pair_records) - 1}"),
        ])

    st.divider()
    st.markdown("### 训练结果展示")

    try:
        before_img = Image.open(selected_record["before_path"]).convert("RGB")
        after_img = Image.open(selected_record["after_path"]).convert("RGB")
        if before_img.size != after_img.size:
            after_img = after_img.resize(before_img.size, Image.BICUBIC)
        blend_img = Image.blend(before_img, after_img, alpha=0.5)

        diff_np = np.abs(np.array(after_img, dtype=np.int16) - np.array(before_img, dtype=np.int16)).astype(np.uint8)
        diff_img = Image.fromarray(diff_np)

        col1, col2, col3 = st.columns(3)
        with col1:
            render_html(f'<div class="image-card-title">输入原图: {html.escape(selected_record["before_name"])}</div>')
            st.image(before_img, width='stretch')
        with col2:
            render_html(f'<div class="image-card-title">训练后结果图: {html.escape(selected_record["after_name"])}</div>')
            st.image(after_img, width='stretch')
        with col3:
            render_html('<div class="image-card-title">融合预览</div>')
            st.image(blend_img, width='stretch')
            render_html('<div class="image-card-title">差分图</div>')
            st.image(diff_img, width='stretch')

        st.caption(
            f"原图路径: {selected_record['before_path']} | "
            f"结果图路径: {selected_record['after_path']}"
        )
    except Exception as e:
        st.error(f"结果图展示失败: {e}")

def create_param_card(param_key, param_info):
    """创建参数信息卡片"""
    tip = param_info.get('tips', '').replace('推荐值', '💡 推荐值')
    return (
        '<div class="param-card">'
        f'<div class="param-name">{html.escape(param_info.get("desc", param_key))}</div>'
        f'<div class="param-desc">{html.escape(param_info.get("help", ""))}</div>'
        f'<div style="font-size: 0.85rem; color: #0066cc; margin-top: 0.3rem;">{html.escape(tip)}</div>'
        '</div>'
    )


def render_html(markup):
    """稳定渲染 HTML，避免缩进的多行字符串被 Markdown 当作代码块显示。"""
    cleaned = dedent(str(markup)).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


def render_info_box(body_html, box_class="info-box"):
    """渲染提示盒。body_html 由调用方控制，可包含少量安全 HTML 标签。"""
    render_html(f'<div class="{html.escape(box_class)}">{body_html}</div>')


def render_quick_nav_card(icon, title, desc, meta):
    """渲染首页快捷入口卡片"""
    render_html(
        (
            '<div class="quick-nav-card">'
            f'<div class="quick-nav-card__icon">{html.escape(icon)}</div>'
            f'<div class="quick-nav-card__title">{html.escape(title)}</div>'
            f'<div class="quick-nav-card__desc">{html.escape(desc)}</div>'
            f'<div class="quick-nav-card__meta">{html.escape(meta)}</div>'
            '</div>'
        )
    )


def render_hero_section(title, subtitle, eyebrow="Research Workspace", tags=None, compact=False):
    """渲染统一的页面头图区域"""
    shell_class = "hero-shell compact" if compact else "hero-shell"
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle).replace("\n", "<br>")
    safe_eyebrow = html.escape(eyebrow)
    tags_html = ""
    if tags:
        tags_html = '<div class="hero-tag-row">' + "".join(
            f'<span class="hero-tag">{html.escape(tag)}</span>' for tag in tags
        ) + "</div>"
    render_html(
        (
            f'<section class="{shell_class}">'
            f'<div class="hero-eyebrow">{safe_eyebrow}</div>'
            f'<h1 class="hero-title">{safe_title}</h1>'
            f'<p class="hero-subtitle">{safe_subtitle}</p>'
            f'{tags_html}'
            '</section>'
        )
    )


def render_section_intro(title, subtitle, tags=None):
    """渲染统一的功能页标题区"""
    render_hero_section(
        title=title,
        subtitle=subtitle,
        eyebrow="Research Module",
        tags=tags or [],
        compact=True,
    )


def render_subsection_title(title):
    """渲染二级功能标题"""
    render_html(f'<h3 class="subsection-header">{html.escape(title)}</h3>')


def render_soft_panel(content_html):
    """渲染轻量说明面板"""
    render_html(f'<div class="soft-panel">{content_html}</div>')


def render_empty_state(title, body, hint=None):
    """渲染统一空状态提示"""
    hint_html = f'<div class="empty-state__hint">{html.escape(hint)}</div>' if hint else ""
    render_html(
        (
            '<div class="empty-state">'
            f'<div class="empty-state__title">{html.escape(title)}</div>'
            f'<div class="empty-state__body">{html.escape(body)}</div>'
            f'{hint_html}'
            '</div>'
        )
    )


def render_workflow_strip(steps):
    """渲染首页研究流程条"""
    cards_html = []
    for idx, step in enumerate(steps, start=1):
        cards_html.append(
            (
                '<div class="workflow-step">'
                f'<div class="workflow-step__index">{idx:02d}</div>'
                f'<div class="workflow-step__title">{html.escape(step["title"])}</div>'
                f'<div class="workflow-step__desc">{html.escape(step["desc"])}</div>'
                '</div>'
            )
        )
    render_html(f'<div class="workflow-strip">{"".join(cards_html)}</div>')


def render_insight_cards(cards):
    """渲染说明型信息卡片"""
    cards_html = []
    for card in cards:
        cards_html.append(
            (
                '<div class="insight-card">'
                f'<div class="insight-card__kicker">{html.escape(card["kicker"])}</div>'
                f'<div class="insight-card__title">{html.escape(card["title"])}</div>'
                f'<div class="insight-card__body">{html.escape(card["body"])}</div>'
                '</div>'
            )
        )
    render_html(f'<div class="insight-grid">{"".join(cards_html)}</div>')


def render_model_fact_grid(facts):
    """渲染模型/实验元信息网格"""
    fact_html = []
    for label, value in facts:
        fact_html.append(
            (
                '<div class="model-fact">'
                f'<div class="model-fact__label">{html.escape(str(label))}</div>'
                f'<div class="model-fact__value">{html.escape(str(value))}</div>'
                '</div>'
            )
        )
    render_html(f'<div class="model-facts">{"".join(fact_html)}</div>')


def style_plotly_figure(fig, *, height=400, show_legend=True):
    """统一 Plotly 图表视觉风格"""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.54)",
        font=dict(
            family="Avenir Next, Segoe UI, PingFang SC, Microsoft YaHei, sans-serif",
            color="#10263f",
        ),
        margin=dict(l=28, r=22, t=58, b=34),
        hoverlabel=dict(
            bgcolor="#10263f",
            bordercolor="rgba(255,255,255,0.16)",
            font=dict(color="#ffffff"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ) if show_legend else dict(visible=False),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(16,38,63,0.08)",
        zerolinecolor="rgba(16,38,63,0.08)",
        linecolor="rgba(16,38,63,0.16)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(16,38,63,0.08)",
        zerolinecolor="rgba(16,38,63,0.08)",
        linecolor="rgba(16,38,63,0.16)",
    )
    return fig

# ==================== UI 页面组件 ====================

def home_page():
    """首页 - 仪表板"""
    render_hero_section(
        title=PROJECT_TITLE,
        subtitle="面向神经辐射场与运动感知去模糊研究的统一可视化工作台，覆盖配置管理、训练监控、模型查看、结果推理与实验分析。",
        eyebrow="NeRF Deblurring Research Platform",
        tags=["配置到训练", "结果可视化", "实验对比分析"],
    )

    render_subsection_title("研究流程")
    render_workflow_strip([
        {"title": "准备数据", "desc": "整理 LLFF 场景目录、图像序列与相机位姿。"},
        {"title": "建立配置", "desc": "选择预设或编辑关键训练、采样与模糊核参数。"},
        {"title": "启动训练", "desc": "后台执行 Deblur-NeRF，记录日志、检查点和评估结果。"},
        {"title": "查看模型", "desc": "按场景视角浏览原图、恢复图、融合图和差分细节。"},
        {"title": "分析结论", "desc": "用 PSNR、SSIM、LPIPS 与可视对比辅助论文实验总结。"},
    ])
    
    # 导航
    render_subsection_title("🚀 快速开始")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_quick_nav_card("⚙️", "配置管理", "创建、加载和编辑训练配置，整理实验参数基线。", "Step 01")
        if st.button("⚙️ 配置管理", width='stretch'):
            _set_page("配置")
            st.rerun()
    with col2:
        render_quick_nav_card("🚀", "训练系统", "启动训练任务，跟踪实验进度，并查看关键曲线变化。", "Step 02")
        if st.button("🚀 训练系统", width='stretch'):
            _set_page("训练")
            st.rerun()
    with col3:
        render_quick_nav_card("🎨", "推理结果", "渲染测试集或路径视频，并快速查看生成的图像与视频结果。", "Step 03")
        if st.button("🎨 推理结果", width='stretch'):
            _set_page("推理")
            st.rerun()
    with col4:
        render_quick_nav_card("📊", "分析对比", "并排比较不同实验输出，提炼 PSNR、SSIM 等性能指标。", "Step 04")
        if st.button("📊 分析对比", width='stretch'):
            _set_page("分析")
            st.rerun()
    with col5:
        render_quick_nav_card("🧠", "模型加载", "加载训练完成的模型，逐视角检查恢复效果和结构细节。", "Step 05")
        if st.button("🧠 模型加载", width='stretch'):
            _set_page("模型")
            st.rerun()
    
    st.divider()
    
    # 项目统计
    render_subsection_title("📈 项目概览")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        configs = get_config_files()
        st.metric("📋 配置文件", len(configs))
    with col2:
        exps = get_experiments()
        st.metric("🔬 实验数", len(exps))
    with col3:
        gpu_info = get_gpu_info()
        gpu_text = "✅" if gpu_info else "❌"
        st.metric("🎮 GPU", gpu_text)
    with col4:
        stats = get_system_stats()
        st.metric("💻 CPU", f"{stats.get('cpu', 0):.0f}%")
    with col5:
        st.metric("📦 状态", "就绪" if Path("requirements.txt").exists() else "待设置")
    
    st.divider()
    
    # 系统状态详情
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            render_subsection_title("💻 系统信息")
            stats = get_system_stats()
            if stats:
                st.progress(stats['cpu'] / 100, text=f"CPU: {stats['cpu']:.1f}%")
                st.progress(stats['memory'] / 100, text=f"内存: {stats['memory']:.1f}%")
            
            gpu_info = get_gpu_info()
            if gpu_info:
                st.markdown("**GPU 信息**")
                st.write(f"• 型号: {gpu_info['name']}")
                st.write(f"• 显存: {gpu_info['used']:.0f}MB / {gpu_info['total']:.0f}MB")
                st.progress(gpu_info['util'] / 100, text=f"利用率: {gpu_info['util']:.1f}%")
        
        with col2:
            render_subsection_title("📁 最近实验")
            exps = get_experiments()[:5]
            if exps:
                for exp in exps:
                    stats = get_experiment_stats(exp)
                    col_name, col_size = st.columns([3, 1])
                    with col_name:
                        st.write(f"📌 **{get_experiment_display_name(exp)}**")
                    with col_size:
                        st.caption(f"{stats['size_mb']:.1f}MB")
            else:
                render_empty_state(
                    "暂无实验数据",
                    "训练完成后，这里会自动显示最近实验、产物大小与模型输出概览。",
                    "先去配置页建立一个实验配置",
                )
    
    st.divider()
    
    # 提示
    render_subsection_title("💡 使用建议")
    render_soft_panel("<p>推荐流程：先建立配置基线，再启动训练，随后在模型与推理模块核查可视结果，最后进入分析页汇总定量指标。若 GPU 显存吃紧，优先降低 N_rand 与 chunk。</p>")
    render_insight_cards([
        {
            "kicker": "First Run",
            "title": "先跑测试配置",
            "body": "第一次验证数据路径时，优先使用低迭代、低分辨率配置，先确认管线能闭环。",
        },
        {
            "kicker": "Quality",
            "title": "再升质量参数",
            "body": "确认训练正常后再提高 factor、N_samples、netwidth 等参数，避免直接把显存拉爆。",
        },
        {
            "kicker": "Paper Ready",
            "title": "保留对比证据",
            "body": "建议每组实验都保存配置、日志、指标和关键视角图，后续写论文会轻松很多。",
        },
    ])

def config_page():
    """配置管理页面"""
    render_section_intro(
        "⚙️ 配置",
        "在这里维护实验配置、快速套用预设，并查看关键参数的解释与推荐范围，帮助训练过程更稳定地复现。",
        tags=["预设模板", "参数编辑", "配置说明"],
    )

    _sync_config_editor_widget_state()

    if st.session_state.get("config_editor_notice"):
        st.success(st.session_state.config_editor_notice)
        del st.session_state["config_editor_notice"]
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 预设", "📝 新建配置", "✏️ 编辑配置", "📚 参数详解"])
    
    # Tab 1: 预设
    with tab1:
        render_subsection_title("配置预设")
        
        render_info_box("<strong>选择预设方创建配置</strong><br>这些预设已经根据常见场景和硬件配置优化过参数。")
        
        cols = st.columns(2)
        for idx, (preset_name, preset_params) in enumerate(PRESET_CONFIGS.items()):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"### {preset_name}")
                    st.write(preset_params.get('desc', ''))
                    
                    # 显示主要参数
                    param_text = []
                    for key, value in preset_params.items():
                        if key != 'desc':
                            param_text.append(f"• {key}: {value}")
                    st.caption("\n".join(param_text))
                    
                    if st.button(f"✨ 使用 {preset_name}", width='stretch'):
                        st.session_state.current_config = dict(preset_params)
                        _sync_config_editor_widget_state()
                        st.success(f"✅ 已加载预设：{preset_name}")
                        st.info("📝 可在'新建配置'或'编辑配置'中进一步调整")
    
    # Tab 2: 新建配置
    with tab2:
        render_subsection_title("创建新配置")
        
        config_name = st.text_input("配置文件名", value="my_config", help="不需要.txt后缀")
        
        with st.expander("📌 基本设置", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                expname = st.text_input(
                    "实验名称",
                    value=st.session_state.current_config.get('expname', 'experiment_001'),
                    help=PARAM_HELP['expname']['help']
                )
                datadir = st.text_input(
                    "数据目录",
                    value=st.session_state.current_config.get('datadir', './data/'),
                    help=PARAM_HELP['datadir']['help']
                )
            with col2:
                basedir = st.text_input(
                    "日志目录",
                    value=st.session_state.current_config.get('basedir', './logs/'),
                    help=PARAM_HELP['basedir']['help']
                )
                tbdir = st.text_input(
                    "TensorBoard目录",
                    value=st.session_state.current_config.get('tbdir', './tb_logs/'),
                    help=PARAM_HELP['tbdir']['help']
                )
            
            col1, col2 = st.columns(2)
            with col1:
                factor = st.slider(
                    "降采样因子",
                    min_value=1, max_value=16,
                    value=int(st.session_state.current_config.get('factor', 4)),
                    help=PARAM_HELP['factor']['help']
                )
            with col2:
                dataset_type = st.selectbox(
                    "数据集类型",
                    SUPPORTED_DATASET_TYPES,
                    key="dataset_type",
                    help="当前 Deblur-NeRF 项目训练脚本只实现了 LLFF 数据格式。"
                )
                st.caption("训练脚本当前仅支持 `llff`，选择其他类型会直接退出。")
        
        with st.expander("🧠 网络架构"):
            col1, col2 = st.columns(2)
            with col1:
                netdepth = st.slider(
                    "网络深度",
                    min_value=4, max_value=16,
                    value=int(st.session_state.current_config.get('netdepth', 8)),
                    help=PARAM_HELP['netdepth']['help']
                )
                netwidth = st.select_slider(
                    "网络宽度",
                    options=[64, 128, 256, 512],
                    value=int(st.session_state.current_config.get('netwidth', 256)),
                    help=PARAM_HELP['netwidth']['help']
                )
            with col2:
                netdepth_fine = st.slider(
                    "精细网络深度",
                    min_value=4, max_value=16,
                    value=int(st.session_state.current_config.get('netdepth_fine', 8))
                )
                netwidth_fine = st.select_slider(
                    "精细网络宽度",
                    options=[64, 128, 256, 512],
                    value=int(st.session_state.current_config.get('netwidth_fine', 256))
                )
            
            use_viewdirs = st.checkbox(
                "使用视角信息",
                value=_to_bool(st.session_state.current_config.get('use_viewdirs', True), True),
                help=PARAM_HELP['use_viewdirs']['help']
            )
        
        with st.expander("⚡ 训练参数"):
            col1, col2 = st.columns(2)
            with col1:
                N_iters = st.slider(
                    "总迭代次数",
                    min_value=10000, max_value=200000, step=5000,
                    value=int(st.session_state.current_config.get('N_iters', 50000)),
                    help=PARAM_HELP['N_iters']['help']
                )
                N_rand = st.slider(
                    "每步采样射线数",
                    min_value=1024, max_value=8192, step=256,
                    value=int(st.session_state.current_config.get('N_rand', 4096)),
                    help=PARAM_HELP['N_rand']['help']
                )
            with col2:
                lrate = st.number_input(
                    "学习率",
                    value=float(st.session_state.current_config.get('lrate', 5e-4)),
                    format="%.1e",
                    help=PARAM_HELP['lrate']['help']
                )
                lrate_decay = st.slider(
                    "学习率衰减周期（千步）",
                    min_value=50, max_value=500, step=50,
                    value=int(st.session_state.current_config.get('lrate_decay', 250))
                )
            
            chunk = st.slider(
                "渲染块大小",
                min_value=1024, max_value=65536, step=1024,
                value=int(st.session_state.current_config.get('chunk', 32768)),
                help=PARAM_HELP['chunk']['help']
            )
        
        with st.expander("📍 采样参数"):
            col1, col2 = st.columns(2)
            with col1:
                N_samples = st.slider(
                    "粗采样点数",
                    min_value=32, max_value=128, step=16,
                    value=int(st.session_state.current_config.get('N_samples', 64)),
                    help=PARAM_HELP['N_samples']['help']
                )
                perturb = st.select_slider(
                    "采样抖动",
                    options=['0.0', '0.5', '1.0'],
                    value=str(st.session_state.current_config.get('perturb', '1.0'))
                )
            with col2:
                N_importance = st.slider(
                    "精细采样点数",
                    min_value=0, max_value=64, step=8,
                    value=int(st.session_state.current_config.get('N_importance', 0))
                )
                raw_noise_std = st.number_input(
                    "噪声标准差",
                    value=float(st.session_state.current_config.get('raw_noise_std', 0.0)),
                    format="%.5f"
                )
        
        with st.expander("🌫️ 模糊核参数"):
            normalized_kernel_type = _normalize_kernel_type(
                st.session_state.current_config.get('kernel_type', 'deformablesparsekernel')
            )
            kernel_type = st.selectbox(
                "模糊核类型",
                ["none", "deformablesparsekernel"],
                format_func=lambda value: "原始 NeRF" if value == "none" else "Deblur-NeRF 稀疏模糊核",
                help=PARAM_HELP['kernel_type']['help'],
                key="kernel_type"
            )
            
            if kernel_type != "none":
                col1, col2, col3 = st.columns(3)
                with col1:
                    kernel_ptnum = st.slider(
                        "稀疏点数",
                        min_value=3, max_value=10,
                        value=int(st.session_state.current_config.get('kernel_ptnum', 5)),
                        help=PARAM_HELP['kernel_ptnum']['help']
                    )
                with col2:
                    kernel_hwindow = st.slider(
                        "模糊核窗口",
                        min_value=5, max_value=30,
                        value=int(st.session_state.current_config.get('kernel_hwindow', 10)),
                        help=PARAM_HELP['kernel_hwindow']['help']
                    )
                with col3:
                    kernel_img_embed = st.select_slider(
                        "图像嵌入维度",
                        options=[16, 32, 64],
                        value=int(st.session_state.current_config.get('kernel_img_embed', 32))
                    )
        
        # 日志设置
        with st.expander("📊 日志设置"):
            col1, col2, col3 = st.columns(3)
            with col1:
                i_print = st.slider(
                    "打印频率", min_value=100, max_value=1000, step=100,
                    value=_to_int(st.session_state.current_config.get('i_print', 200), 200)
                )
                i_weights = st.slider(
                    "权重保存频率", min_value=5000, max_value=50000, step=5000,
                    value=_to_int(st.session_state.current_config.get('i_weights', 20000), 20000)
                )
            with col2:
                i_tensorboard = st.slider(
                    "TensorBoard频率", min_value=100, max_value=1000, step=100,
                    value=_to_int(st.session_state.current_config.get('i_tensorboard', 200), 200)
                )
                i_testset = st.slider(
                    "测试频率", min_value=5000, max_value=50000, step=5000,
                    value=_to_int(st.session_state.current_config.get('i_testset', 20000), 20000)
                )
            with col3:
                i_video = st.slider(
                    "视频生成频率", min_value=5000, max_value=50000, step=5000,
                    value=_to_int(st.session_state.current_config.get('i_video', 20000), 20000)
                )

        preserved_keys = sorted(
            key for key in st.session_state.current_config.keys()
            if key not in FORM_MANAGED_CONFIG_KEYS
        )
        if preserved_keys:
            with st.expander("🔐 将自动保留的高级参数", expanded=False):
                st.caption("这些参数当前表单不直接编辑，但在从已有配置加载后再次保存时会继续保留。")
                st.code(
                    "\n".join(
                        f"{key} = {st.session_state.current_config[key]}"
                        for key in preserved_keys
                    ),
                    language="ini",
                )
        
        # 保存按钮
        if st.button("💾 保存配置", width='stretch'):
            config_data = dict(st.session_state.current_config)
            config_data.update({
                "expname": expname,
                "datadir": datadir,
                "basedir": basedir,
                "tbdir": tbdir,
                "dataset_type": _normalize_dataset_type(dataset_type),
                "factor": str(factor),
                "netdepth": str(netdepth),
                "netwidth": str(netwidth),
                "netdepth_fine": str(netdepth_fine),
                "netwidth_fine": str(netwidth_fine),
                "use_viewdirs": str(use_viewdirs),
                "N_iters": str(N_iters),
                "N_rand": str(N_rand),
                "lrate": str(lrate),
                "lrate_decay": str(lrate_decay),
                "chunk": str(chunk),
                "N_samples": str(N_samples),
                "N_importance": str(N_importance),
                "perturb": str(perturb),
                "raw_noise_std": str(raw_noise_std),
                "kernel_type": _normalize_kernel_type(kernel_type),
                "kernel_ptnum": str(kernel_ptnum) if kernel_type != "none" else "",
                "kernel_hwindow": str(kernel_hwindow) if kernel_type != "none" else "",
                "kernel_img_embed": str(kernel_img_embed) if kernel_type != "none" else "",
                "i_print": str(i_print),
                "i_tensorboard": str(i_tensorboard),
                "i_weights": str(i_weights),
                "i_testset": str(i_testset),
                "i_video": str(i_video),
            })
            
            # 验证配置
            errors, warnings = validate_config(config_data)
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                if save_config(config_data, f"{config_name}.txt"):
                    st.session_state.current_config = config_data
                
                if warnings:
                    st.warning("⚠️ 警告：\n" + "\n".join(warnings))
    
    # Tab 3: 编辑配置
    with tab3:
        render_subsection_title("编辑现有配置")
        
        configs = get_config_files()
        if not configs:
            render_empty_state(
                "暂无配置文件",
                "configs 目录下还没有可编辑的配置文件。可以先从预设生成一个新配置。",
                "去预设页选择一个模板",
            )
        else:
            selected_config = st.selectbox("选择配置文件", configs, key="edit_config_select")
            config_data = load_config(selected_config)
            
            if config_data:
                st.success(f"✅ 已加载：{selected_config}")
                if st.button("📥 加载到新建配置编辑器", width='stretch'):
                    st.session_state.current_config = dict(config_data)
                    st.session_state.config_editor_notice = "✅ 已加载到“新建配置”页，可直接修改后另存为新文件。"
                    st.rerun()
                
                # 显示配置文件内容
                with st.expander("📋 原始配置内容", expanded=False):
                    st.code("\n".join([f"{k} = {v}" for k, v in config_data.items()]), language="ini")
                
                # 参数编辑（类似新建配置）
                st.write("编辑功能与新建配置相同，在'新建配置'标签页加载后修改即可保存为新文件")
    
    # Tab 4: 参数详解
    with tab4:
        render_subsection_title("参数详细说明")
        
        # 参数分类显示
        categories = {
            "基本设置": ["expname", "datadir", "basedir", "factor"],
            "网络架构": ["netdepth", "netwidth", "netdepth_fine", "netwidth_fine", "use_viewdirs"],
            "训练参数": ["N_iters", "N_rand", "lrate", "lrate_decay", "chunk"],
            "采样参数": ["N_samples", "N_importance", "perturb"],
            "模糊核": ["kernel_type", "kernel_ptnum", "kernel_hwindow"],
        }
        
        for category, params in categories.items():
            with st.expander(f"📚 {category}", expanded=False):
                for param in params:
                    if param in PARAM_HELP:
                        info = PARAM_HELP[param]
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"**{info.get('desc', param)}**")
                        with col2:
                            st.markdown(info.get('help', ''))
                        
                        # 显示额外信息
                        extra_info = []
                        if 'default' in info:
                            extra_info.append(f"**默认值**: {info['default']}")
                        if 'range' in info:
                            extra_info.append(f"**范围**: {info['range'][0]} ~ {info['range'][1]}")
                        if 'recommended' in info:
                            extra_info.append(f"**推荐值**: {', '.join(map(str, info['recommended']))}")
                        if 'tips' in info:
                            extra_info.append(f"💡 {info['tips']}")
                        
                        if extra_info:
                            st.caption(" | ".join(extra_info))
                        st.divider()

def training_page():
    """训练管理页面"""
    render_section_intro(
        "🚀 训练",
        "集中管理训练任务启动、过程监控与曲线分析，让实验状态、日志输出和核心指标变化更直观。",
        tags=["训练启动", "状态监控", "曲线分析"],
    )
    
    tab1, tab2, tab3 = st.tabs(["🎯 启动训练", "📊 训练监控", "📈 曲线分析"])
    
    with tab1:
        render_subsection_title("启动训练任务")
        
        configs = get_config_files()
        if not configs:
            render_empty_state(
                "暂无可用配置文件",
                "训练启动需要一个 configs 下的 .txt 配置文件。你可以先从预设模板创建配置。",
                "先进入配置页创建配置",
            )
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                selected_config = st.selectbox("选择配置文件", configs, help="选择要使用的配置", key="train_config_select")
                config_data = load_config(selected_config)
            
            if config_data:
                with col2:
                    st.markdown("**配置信息**")
                    st.write(f"实验: {config_data.get('expname', 'N/A')}")
                    st.write(f"数据: {config_data.get('datadir', 'N/A')}")
                
                st.divider()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**训练设置**")
                    save_ckpt = st.checkbox("定期保存检查点", value=True, help="在指定间隔保存训练权重")
                    render_test = st.checkbox("训练后渲染测试集", value=True, help="训练完成后自动渲染测试集")
                    num_gpus = st.number_input("使用GPU数", value=1, min_value=1, max_value=8, help="使用多GPU加速")
                
                with col2:
                    st.markdown("**监控设置**")
                    enable_tb = st.checkbox("启用TensorBoard", value=True, help="记录训练指标")
                    log_level = st.selectbox("日志级别", ["INFO", "DEBUG", "WARNING"], index=0, key="log_level_select")
                    auto_continue = st.checkbox("中断后自动继续", value=False, help="从最新检查点继续训练")
                
                st.divider()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🚀 启动训练", width='stretch', key="start_training"):
                        # 验证配置
                        errors, warnings = validate_config(config_data)
                        if errors:
                            st.error("❌ 配置验证失败，无法启动训练:")
                            for error in errors:
                                st.error(f"  • {error}")
                        else:
                            try:
                                config_path = Path("configs") / selected_config
                                command_args = build_training_command(
                                    config_path=config_path,
                                    config_data=config_data,
                                    num_gpus=num_gpus,
                                    save_ckpt=save_ckpt,
                                    render_testset=render_test,
                                    enable_tb=enable_tb,
                                    auto_continue=auto_continue,
                                )
                                log_path = Path(config_data.get("basedir", "./logs/")) / config_data.get("expname", "experiment") / "training.log"

                                with st.spinner("正在启动训练进程..."):
                                    pid = _launch_background_process(command_args, log_path)
                                    time.sleep(2)  # 等待进程启动
                                
                                st.success("✅ 训练已启动！")
                                st.info(f"""
                                **训练信息:**
                                - 配置: {selected_config}
                                - 实验: {config_data.get('expname', 'N/A')}
                                - 数据: {config_data.get('datadir', 'N/A')}
                                - 进程 PID: {pid}
                                - 日志: `{log_path}`
                                
                                **监控方式:**
                                1. 查看日志: `tail -f {log_path}`
                                2. TensorBoard: `tensorboard --logdir {config_data.get('tbdir', './tb_logs/')}`
                                3. 切换到'训练监控'标签页 (需要刷新)
                                """)
                                st.code(_format_shell_command(command_args), language="bash")
                                if warnings:
                                    for warning in warnings:
                                        st.warning(f"⚠️ {warning}")
                                
                            except Exception as e:
                                st.error(f"❌ 启动训练失败: {str(e)}")
                
                with col2:
                    if st.button("📖 查看命令", width='stretch'):
                        with st.expander("运行命令"):
                            command_args = build_training_command(
                                config_path=Path("configs") / selected_config,
                                config_data=config_data,
                                num_gpus=num_gpus,
                                save_ckpt=save_ckpt,
                                render_testset=render_test,
                                enable_tb=enable_tb,
                                auto_continue=auto_continue,
                            )
                            st.code(_format_shell_command(command_args), language="bash")
                            st.caption(f"当前日志级别选项为 `{log_level}`，训练脚本本身未提供独立日志级别参数。")
                
                with col3:
                    if st.button("✅ 验证配置", width='stretch'):
                        errors, warnings = validate_config(config_data)
                        if errors:
                            for error in errors:
                                st.error(f"❌ {error}")
                        else:
                            st.success("✅ 配置验证无误!")
                            if warnings:
                                for warning in warnings:
                                    st.warning(f"⚠️ {warning}")
    
    with tab2:
        render_subsection_title("训练实时监控")
        
        exps = get_experiments()
        
        if not exps:
            render_empty_state(
                "暂无训练实验",
                "还没有检测到实验目录。启动训练后，这里会展示进度、检查点、日志和指标曲线。",
                "先在启动训练页选择配置",
            )
        else:
            selected_exp = st.selectbox(
                "选择实验",
                exps,
                format_func=get_experiment_display_name,
                key="monitor_exp_select"
            )
            
            # 实验统计信息
            stats = get_experiment_stats(selected_exp)
            status = infer_training_status(selected_exp)
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📁 实验名", get_experiment_display_name(selected_exp))
                with col2:
                    st.metric("💾 大小", f"{stats['size_mb']:.1f}MB")
                with col3:
                    st.metric("📦 检查点", stats['ckpt_count'])
                with col4:
                    st.metric("🖼️ 图像", stats['images_count'])
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 训练指标**")
                metric_points = parse_test_metrics(selected_exp)
                if metric_points:
                    if _plotly_is_available():
                        iterations = [x[0] for x in metric_points]
                        psnr_values = [x[1] for x in metric_points]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=iterations, y=psnr_values,
                            mode='lines+markers',
                            name='PSNR',
                            line=dict(color=VISUAL_SERIES_COLORS[2], width=3, shape="spline"),
                            marker=dict(size=7, color=VISUAL_SERIES_COLORS[2], line=dict(width=1, color="#ffffff"))
                        ))
                        fig.update_layout(
                            title="PSNR 测试曲线（来自 test_metrics.txt）",
                            xaxis_title="迭代次数",
                            yaxis_title="PSNR (dB)",
                            hovermode='x unified',
                        )
                        style_plotly_figure(fig, height=350, show_legend=False)
                        st.plotly_chart(fig, width='stretch')
                else:
                    render_empty_state(
                        "暂无测试指标",
                        "暂未找到 test_metrics.txt。训练产生测试集评估后，这里会显示 PSNR 曲线。",
                        "等待 i_testset 周期产出",
                    )
            
            with col2:
                st.markdown("**🎯 训练状态**")
                progress_percent = status["progress_percent"]
                col_status, col_refresh = st.columns([3, 1])
                with col_status:
                    st.progress(progress_percent / 100, text=f"进度: {progress_percent}%")
                
                box_class = "success-box" if status["is_running_hint"] else "warning-box"
                target_iters_text = status["target_iters"] if status["target_iters"] > 0 else "未知"
                render_info_box(
                    (
                        f'<strong>{html.escape(status["label"])}</strong><br>'
                        f'{html.escape(status["detail"])}<br>'
                        f'当前迭代: {status["latest_iter"]}/{target_iters_text}<br>'
                        f'检查点数: {stats["ckpt_count"]}<br>'
                        f'输出图像数: {stats["images_count"]}'
                    ),
                    box_class=box_class,
                )
    
    with tab3:
        render_subsection_title("训练曲线分析")
        
        exps = get_experiments()
        
        if not exps:
            render_empty_state(
                "暂无实验数据",
                "训练实验生成后，曲线分析会自动读取 test_metrics.txt 并支持多实验对比。",
                "先完成至少一个训练实验",
            )
        else:
            selected_exps = st.multiselect(
                "选择要对比的实验",
                exps,
                default=[exps[0]] if exps else [],
                max_selections=3,
                format_func=get_experiment_display_name
            )
            
            if selected_exps:
                col1, col2 = st.columns(2)
                metric_map = {exp: parse_test_metric_records(exp) for exp in selected_exps}
                has_metric_data = any(metric_map.values())
                
                with col1:
                    st.markdown("**PSNR 对比**")
                    if _plotly_is_available():
                        fig = go.Figure()
                        for idx, exp in enumerate(selected_exps):
                            records = metric_map.get(exp, [])
                            if not records:
                                continue
                            iterations = [record["iter"] for record in records if record.get("psnr") is not None]
                            psnr_values = [record["psnr"] for record in records if record.get("psnr") is not None]
                            if not iterations:
                                continue
                            fig.add_trace(go.Scatter(
                                x=iterations, y=psnr_values,
                                mode='lines+markers',
                                name=get_experiment_display_name(exp),
                                line=dict(color=VISUAL_SERIES_COLORS[idx % len(VISUAL_SERIES_COLORS)], width=3),
                                marker=dict(
                                    size=7,
                                    color=VISUAL_SERIES_COLORS[idx % len(VISUAL_SERIES_COLORS)],
                                    line=dict(width=1, color="#ffffff"),
                                )
                            ))
                        fig.update_layout(
                            title="PSNR 对比",
                            xaxis_title="迭代次数",
                            yaxis_title="PSNR (dB)",
                            hovermode='x unified',
                        )
                        style_plotly_figure(fig, height=400)
                        if has_metric_data and fig.data:
                            st.plotly_chart(fig, width='stretch')
                        else:
                            render_empty_state(
                                "暂无 PSNR 对比数据",
                                "未找到可用于对比的 PSNR 指标，请先生成 test_metrics.txt。",
                                "训练到测试集评估周期后再查看",
                            )
                    else:
                        if go is not None:
                            st.info("未找到可用于对比的 PSNR 指标，请先生成 `test_metrics.txt`。")
                
                with col2:
                    st.markdown("**MSE 对比**")
                    if _plotly_is_available():
                        fig = go.Figure()
                        for idx, exp in enumerate(selected_exps):
                            records = metric_map.get(exp, [])
                            if not records:
                                continue
                            iterations = [record["iter"] for record in records if record.get("mse") is not None]
                            loss_values = [record["mse"] for record in records if record.get("mse") is not None]
                            if not iterations:
                                continue
                            fig.add_trace(go.Scatter(
                                x=iterations, y=loss_values,
                                mode='lines+markers',
                                name=get_experiment_display_name(exp),
                                line=dict(color=VISUAL_SERIES_COLORS[idx % len(VISUAL_SERIES_COLORS)], width=3),
                                marker=dict(
                                    size=7,
                                    color=VISUAL_SERIES_COLORS[idx % len(VISUAL_SERIES_COLORS)],
                                    line=dict(width=1, color="#ffffff"),
                                )
                            ))
                        fig.update_layout(
                            title="MSE 对比",
                            xaxis_title="迭代次数",
                            yaxis_title="MSE",
                            hovermode='x unified',
                        )
                        style_plotly_figure(fig, height=400)
                        if has_metric_data and fig.data:
                            st.plotly_chart(fig, width='stretch')
                        else:
                            render_empty_state(
                                "暂无 MSE 曲线数据",
                                "当前选择的实验尚未记录可绘制的 MSE 数据。",
                                "完成测试集评估后再查看",
                            )
                    else:
                        if go is not None:
                            st.info("暂无 MSE 曲线数据。")

def inference_page():
    """推理与结果页面"""
    render_section_intro(
        "🎨 推理与结果展示",
        "运行测试集渲染、输出路径视频，并从图像、视频和质量指标三个维度回看模型恢复效果。",
        tags=["渲染输出", "前后对比", "质量评估"],
    )

    # 避免在 st.tabs 中无条件渲染所有媒体资源。
    # Streamlit 会在每次 rerun 时执行所有 tab 内容，图像/视频较多时容易反复创建新的内存媒体 ID，
    # 重启服务或快速切页后就更容易看到 "MediaFileHandler: Missing file <hash>" 日志。
    inference_section = st.radio(
        "推理功能",
        ["🎯 运行推理", "🖼️ 结果查看", "🔍 前后对比", "📊 质量指标"],
        horizontal=True,
        label_visibility="collapsed",
        key="inference_section",
    )

    if inference_section == "🎯 运行推理":
        render_subsection_title("运行推理")
        
        exps = get_experiments()
        if not exps:
            render_empty_state(
                "暂无可推理实验",
                "还没有检测到已训练实验。推理需要实验目录中的 config.txt 或 args.txt 才能自动构建命令。",
                "先完成训练或检查 logs 目录",
            )
        else:
            selected_exp = st.selectbox(
                "选择实验",
                exps,
                format_func=get_experiment_display_name,
                key="inference_exp_select"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**推理设置**")
                render_factor = st.select_slider(
                    "渲染降采样因子",
                    options=[0, 1, 2, 4, 8],
                    value=4,
                    help="0 表示全分辨率；数值越大越快，但输出分辨率越低。"
                )
                render_poses = st.selectbox("渲染方式", ["测试集", "螺旋路径", "EPI路径"], key="render_poses_select")
            
            with col2:
                st.markdown("**输出设置**")
                expected_output = "PNG 图像序列" if render_poses == "测试集" else "MP4 视频"
                st.text_input("预期输出", value=expected_output, disabled=True)
                num_workers = st.slider("GPU并行数", 1, 8, 1, help="映射到 `run_nerf.py --num_gpu`")
            
            st.divider()
            if render_poses == "测试集":
                st.caption("当前脚本在测试集模式下会输出 PNG 图像序列。")
            else:
                st.caption("当前脚本在路径渲染模式下会输出 MP4 视频。")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🎯 启动推理", width='stretch'):
                    command_args, error_message = build_inference_command(
                        exp_name=selected_exp,
                        render_mode=render_poses,
                        render_factor=render_factor,
                        num_gpus=num_workers,
                    )
                    if error_message:
                        st.error(f"❌ {error_message}")
                    else:
                        exp_dir = get_experiment_dir(selected_exp)
                        mode_name = "test" if render_poses == "测试集" else ("epi" if render_poses == "EPI路径" else "path")
                        log_path = exp_dir / f"render_{mode_name}.log"
                        pid = _launch_background_process(command_args, log_path)
                        st.success("✅ 推理任务已启动")
                        st.info(f"""
                        **推理信息:**
                        - 实验: {get_experiment_display_name(selected_exp)}
                        - 模式: {render_poses}
                        - 进程 PID: {pid}
                        - 日志: `{log_path}`
                        """)
                        st.code(_format_shell_command(command_args), language="bash")
            
            with col2:
                if st.button("🎬 生成视频", width='stretch'):
                    command_args, error_message = build_inference_command(
                        exp_name=selected_exp,
                        render_mode="螺旋路径",
                        render_factor=render_factor,
                        num_gpus=num_workers,
                    )
                    if error_message:
                        st.error(f"❌ {error_message}")
                    else:
                        log_path = get_experiment_dir(selected_exp) / "render_video.log"
                        pid = _launch_background_process(command_args, log_path)
                        st.success("✅ 视频生成任务已启动")
                        st.info(f"日志: `{log_path}` | PID: {pid}")
                        st.code(_format_shell_command(command_args), language="bash")

    elif inference_section == "🖼️ 结果查看":
        render_subsection_title("查看渲染结果")
        
        exps = get_experiments()
        if not exps:
            render_empty_state(
                "暂无实验",
                "训练或推理产物会在这里以图像网格和视频列表展示。",
                "先启动一次训练",
            )
        else:
            selected_exp = st.selectbox(
                "选择实验",
                exps,
                format_func=get_experiment_display_name,
                key="result_exp"
            )
            
            images = list_result_images(selected_exp)
            videos = list_result_videos(selected_exp)
            if not images and not videos:
                render_empty_state(
                    "暂无结果图像或视频",
                    "当前实验还没有检测到 testset、renderonly 或路径视频输出。",
                    "运行测试集渲染或生成视频",
                )
            if images:
                render_html(f'<span class="status-pill">找到 {len(images)} 张结果图像</span>')
                
                # 图像网格显示
                cols = st.columns(3)
                for idx, img_path in enumerate(images[:9]):
                    with cols[idx % 3]:
                        try:
                            img = Image.open(img_path)
                            render_html(f'<div class="image-card-title">{html.escape(img_path.name)}</div>')
                            st.image(img, width='stretch')
                        except Exception as e:
                            st.error(f"加载失败: {img_path.name}")
                
                if len(images) > 9:
                    st.info(f"还有 {len(images) - 9} 张未显示的图像")
                    
                    if st.button("📥 加载更多"):
                        cols = st.columns(3)
                        for idx, img_path in enumerate(images[9:18]):
                            with cols[(idx+9) % 3]:
                                try:
                                    img = Image.open(img_path)
                                    render_html(f'<div class="image-card-title">{html.escape(img_path.name)}</div>')
                                    st.image(img, width='stretch')
                                except Exception as e:
                                    st.error(f"加载失败: {img_path.name}")

            if videos:
                st.divider()
                render_html(f'<span class="status-pill">找到 {len(videos)} 个结果视频</span>')
                for video_path in videos[:4]:
                    render_html(f'<div class="image-card-title">{html.escape(video_path.name)}</div>')
                    st.video(str(video_path))
                    st.caption(f"路径: `{video_path}`")

                if len(videos) > 4:
                    st.info(f"还有 {len(videos) - 4} 个视频未展示")

    elif inference_section == "🔍 前后对比":
        render_subsection_title("训练前后图像对比")
        st.caption("自动匹配实验输入图像与渲染输出图，支持融合滑块与差分查看。")
        exps = get_experiments()
        if not exps:
            render_empty_state(
                "暂无实验",
                "前后对比需要实验目录、输入图像和渲染输出共同存在。",
                "先训练并生成 testset 输出",
            )
        else:
            selected_exp = st.selectbox(
                "选择实验",
                exps,
                format_func=get_experiment_display_name,
                key="before_after_exp"
            )
            render_before_after_compare(selected_exp, widget_key_prefix=f"before_after_{get_experiment_widget_key(selected_exp)}")

    else:
        render_subsection_title("质量指标分析")
        
        render_info_box(
            "<strong>图像质量评估指标</strong><br>"
            "• <strong>PSNR</strong>: 峰值信噪比，值越高越好<br>"
            "• <strong>SSIM</strong>: 结构相似度，范围0-1，越接近1越好<br>"
            "• <strong>LPIPS</strong>: 感知损失，值越低越好"
        )
        
        exps = get_experiments()
        
        if len(exps) < 1:
            render_empty_state(
                "需要至少 1 个实验",
                "质量指标分析会读取 test_metrics.txt 中的 PSNR、SSIM 与 LPIPS。",
                "先完成测试集评估",
            )
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_exp = st.selectbox(
                    "选择实验",
                    exps,
                    format_func=get_experiment_display_name,
                    key="inference_metric_exp_select"
                )
            with col2:
                metric_type = st.selectbox("指标类型", ["PSNR", "SSIM", "LPIPS"], key="inference_metric_type_select")

            metric_records = parse_test_metric_records(selected_exp)
            metric_key = metric_type.lower()
            metric_values = [record[metric_key] for record in metric_records if record.get(metric_key) is not None]
            metric_iters = [record["iter"] for record in metric_records if record.get(metric_key) is not None]

            if not metric_values:
                render_empty_state(
                    "未找到真实评估指标",
                    "当前实验尚未产生可读取的 PSNR、SSIM 或 LPIPS 记录。",
                    "完成测试集评估后再查看",
                )
            elif _plotly_is_available():
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metric_iters,
                    y=metric_values,
                    marker=dict(
                        color=metric_values,
                        colorscale=[[0, "#183a60"], [0.52, "#0f617d"], [1, "#d88b2d"]],
                        line=dict(color="rgba(255,255,255,0.7)", width=1),
                    ),
                    text=[f"{v:.4f}" for v in metric_values],
                    textposition='auto',
                    name=metric_type
                ))
                fig.update_layout(
                    title=f"{metric_type} 历次评估结果",
                    xaxis_title="迭代次数",
                    yaxis_title=metric_type,
                )
                style_plotly_figure(fig, height=400, show_legend=False)
                st.plotly_chart(fig, width='stretch')

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最新值", f"{metric_values[-1]:.4f}")
                with col2:
                    st.metric("平均值", f"{np.mean(metric_values):.4f}")
                with col3:
                    st.metric("最大值", f"{np.max(metric_values):.4f}")
                with col4:
                    st.metric("最小值", f"{np.min(metric_values):.4f}")

def analysis_page():
    """对比分析页面"""
    render_section_intro(
        "📊 实验对比分析",
        "把不同实验结果放到同一视野下查看，兼顾主观图像比较和定量指标概览，方便快速判断方案优劣。",
        tags=["并行对比", "性能报告", "指标概览"],
    )
    
    tab1, tab2 = st.tabs(["🔄 并行对比", "📈 性能分析"])
    
    with tab1:
        render_subsection_title("多实验并行对比")
        
        exps = get_experiments()
        if len(exps) < 2:
            render_empty_state(
                "需要至少 2 个实验",
                "并行对比会把两组实验的同索引结果图与最新指标放在同一视野中。",
                "训练第二组实验后再回来",
            )
        else:
            col1, col2 = st.columns(2)
            with col1:
                exp1 = st.selectbox("实验1", exps, format_func=get_experiment_display_name, key="exp1_select")
            with col2:
                exp2 = st.selectbox("实验2", exps, index=min(1, len(exps)-1), format_func=get_experiment_display_name, key="exp2_select")
            
            if exp1 != exp2:
                st.divider()
                
                images1 = list_result_images(exp1)
                images2 = list_result_images(exp2)
                
                if images1 and images2:
                    img_idx = st.slider("选择图像索引", 0, min(len(images1), len(images2)) - 1, 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        render_html(f'<div class="image-card-title">{html.escape(get_experiment_display_name(exp1))}</div>')
                        try:
                            img1 = Image.open(images1[img_idx])
                            st.image(img1, width='stretch')
                        except:
                            st.error("图像加载失败")
                    
                    with col2:
                        render_html(f'<div class="image-card-title">{html.escape(get_experiment_display_name(exp2))}</div>')
                        try:
                            img2 = Image.open(images2[img_idx])
                            st.image(img2, width='stretch')
                        except:
                            st.error("图像加载失败")
                    
                    st.divider()
                    
                    # 对比指标
                    metric1 = get_latest_metric_record(exp1)
                    metric2 = get_latest_metric_record(exp2)
                    if metric1 and metric2 and metric1.get("psnr") is not None and metric2.get("psnr") is not None:
                        psnr_delta = metric2["psnr"] - metric1["psnr"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"PSNR ({get_experiment_name(exp1)})", f"{metric1['psnr']:.3f} dB")
                        with col2:
                            st.metric(f"PSNR ({get_experiment_name(exp2)})", f"{metric2['psnr']:.3f} dB")
                        with col3:
                            st.metric("PSNR 差异", f"{psnr_delta:.3f} dB", delta=f"{psnr_delta:.3f} dB")
                    else:
                        st.info("未找到两组实验的真实 PSNR 记录，暂无法显示量化对比。")
    
    with tab2:
        render_subsection_title("性能分析报告")
        
        exps = get_experiments()
        
        if not exps:
            render_empty_state(
                "暂无实验数据",
                "性能分析会自动汇总实验大小、检查点、输出图像和最新评估指标。",
                "先完成一次训练",
            )
        else:
            st.markdown("#### 📋 实验统计表")
            
            exp_data = []
            for exp in exps[:5]:
                stats = get_experiment_stats(exp)
                latest_metrics = get_latest_metric_record(exp) or {}
                exp_data.append({
                    "实验名": get_experiment_display_name(exp),
                    "大小(MB)": f"{stats['size_mb']:.1f}",
                    "检查点数": stats['ckpt_count'],
                    "输出数": stats['images_count'],
                    "最新PSNR": f"{latest_metrics['psnr']:.3f}" if latest_metrics.get('psnr') is not None else "N/A",
                    "最新SSIM": f"{latest_metrics['ssim']:.4f}" if latest_metrics.get('ssim') is not None else "N/A",
                    "创建时间": stats['created'],
                })

            if pd is None:
                st.warning(f"统计表暂不可用：`pandas` 导入失败（{PANDAS_IMPORT_ERROR}）。")
            else:
                df = pd.DataFrame(exp_data)
                st.dataframe(df, width='stretch')
            
            st.markdown("#### 📊 关键指标概览")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_size = np.mean([float(row["大小(MB)"]) for row in exp_data]) if exp_data else 0
                st.metric("平均实验大小", f"{avg_size:.1f}MB")
            
            with col2:
                total_checkpoints = sum([row["检查点数"] for row in exp_data])
                st.metric("总检查点数", total_checkpoints)
            
            with col3:
                psnr_values = [float(row["最新PSNR"]) for row in exp_data if row["最新PSNR"] != "N/A"]
                if psnr_values:
                    st.metric("平均最新 PSNR", f"{np.mean(psnr_values):.3f} dB")
                else:
                    total_images = sum([row["输出数"] for row in exp_data])
                    st.metric("总输出图像", total_images)

# ==================== 主程序 ====================

def main():
    if st.session_state.get("page_sync_pending"):
        st.session_state.sidebar_page = st.session_state.page
        st.session_state.page_sync_pending = False

    with st.sidebar:
        render_html(
            (
                '<div class="sidebar-brand">'
                '<div class="sidebar-brand__eyebrow">Research Console</div>'
                f'<div class="sidebar-brand__title">{html.escape(PROJECT_TITLE)}</div>'
                '<div class="sidebar-brand__meta">配置、训练、推理、模型查看与实验分析的一体化可视化工作台</div>'
                '</div>'
            )
        )
        st.divider()
        
        st.markdown("### 📌 导航菜单")
        page = st.radio(
            "选择页面",
            ["首页", "配置", "训练", "推理", "模型", "分析"],
            key="sidebar_page",
            label_visibility="collapsed"
        )
        st.session_state.page = page
        
        st.divider()
        st.markdown("### 💻 系统信息")
        
        stats = get_system_stats()
        if stats:
            st.progress(stats['cpu'] / 100, text=f"CPU: {stats['cpu']:.0f}%")
            st.progress(stats['memory'] / 100, text=f"内存: {stats['memory']:.0f}%")
        
        gpu_info = get_gpu_info()
        if gpu_info:
            st.markdown(f"**GPU**: {gpu_info['name'][:20]}...")
            st.progress(gpu_info['util'] / 100, text=f"用率: {gpu_info['util']:.0f}%")
        else:
            st.caption("GPU: 不可用")
        
        st.divider()
        st.markdown("### 📊 项目统计")
        
        col1, col2 = st.columns(2)
        with col1:
            configs = get_config_files()
            st.metric("配置", len(configs))
        with col2:
            exps = get_experiments()
            st.metric("实验", len(exps))
        
        st.divider()
        st.caption(f"v2.0 Enhanced | {datetime.now().strftime('%Y-%m-%d')}")
    
    # 路由
    if st.session_state.page == "首页":
        home_page()
    elif st.session_state.page == "配置":
        config_page()
    elif st.session_state.page == "训练":
        training_page()
    elif st.session_state.page == "推理":
        inference_page()
    elif st.session_state.page == "模型":
        model_loader_page()
    elif st.session_state.page == "分析":
        analysis_page()
    
    st.divider()
    st.caption(PROJECT_FOOTER)

if __name__ == "__main__":
    main()
