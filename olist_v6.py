import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import font_manager, rc
import platform
import sys

# Streamlit Auto-Launch Logic
if __name__ == "__main__":
    try:
        from streamlit.web import cli as stcli
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            print("🚀 Streamlit 대시보드를 실행합니다...")
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            sys.exit(stcli.main())
    except ImportError:
        pass

# --- Configuration & Style ---
st.set_page_config(page_title="판매자 배송 성과 대시보드", layout="wide")

# Korean Font Support
if platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['font.family'] = font_name
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# Paths
# Running from the same directory as the script
BASE_PATH = "data/"
OUTPUT_PATH = "output/"

# --- Helper: Bar labels ---
def add_bar_labels(ax, fmt="{:.1f}", padding=3, fontsize=9):
    for container in ax.containers:
        vals = getattr(container, "datavalues", None)
        if vals is None:
            continue
        labels = [fmt.format(v) if pd.notna(v) else "" for v in vals]
        ax.bar_label(container, labels=labels, padding=padding, fontsize=fontsize)


# --- Data Loading ---
@st.cache_data
def load_and_process_data():
    # Load raw data
    sellers = pd.read_csv(f"{BASE_PATH}proc_olist_sellers_dataset.csv")
    order_items = pd.read_csv(f"{BASE_PATH}proc_olist_order_items_dataset.csv")
    products = pd.read_csv(f"{BASE_PATH}proc_olist_products_dataset.csv")
    orders = pd.read_csv(f"{BASE_PATH}proc_olist_orders_dataset.csv")
    reviews = pd.read_csv(f"{BASE_PATH}olist_order_reviews_dataset_translated.csv")
    customers = pd.read_csv(f"{BASE_PATH}proc_olist_customers_dataset.csv")

    # Date conversion
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Category Mapping (optional)
    mapping_updates = {
        'office_furniture': '사무용 가구',
        'stationery': '사무용품',
        'computers': '컴퓨터'
    }
    for eng, kor in mapping_updates.items():
        products.loc[products['product_category_name_english'] == eng, 'product_category_name_korean'] = kor

    # --- Segmentation Logic ---
    item_orders = pd.merge(
        order_items,
        orders[['order_id', 'delivery_delay_time', 'order_purchase_timestamp']],
        on='order_id'
    )

    seller_stats = item_orders.groupby('seller_id').agg({
        'price': 'sum',
        'order_id': 'nunique',
        'delivery_delay_time': 'mean'
    }).rename(columns={'price': 'total_sales', 'order_id': 'order_count', 'delivery_delay_time': 'avg_delay'})

    seller_reviews = pd.merge(
        reviews[['order_id', 'review_score']],
        order_items[['order_id', 'seller_id']],
        on='order_id'
    )
    avg_seller_reviews = seller_reviews.groupby('seller_id')['review_score'].mean().rename('avg_review')

    seller_stats = seller_stats.join(avg_seller_reviews, how='left')
    refined_stats = seller_stats.dropna(subset=['total_sales', 'avg_review', 'avg_delay']).copy()

    refined_stats['sales_rank'] = refined_stats['total_sales'].rank(pct=True)
    refined_stats['op_score'] = (
        refined_stats['avg_review'].rank(pct=True) +
        (-refined_stats['avg_delay']).rank(pct=True)
    ) / 2

    def classify_seller(row):
        is_top_sales = row['sales_rank'] >= 0.8
        is_good_op = row['op_score'] >= 0.5
        if is_top_sales and is_good_op:
            return '핵심 판매자 (Core)'
        elif is_top_sales and not is_good_op:
            return '불안정 성장 (Unstable)'
        elif not is_top_sales and is_good_op:
            return '박리다매형 (Low-Margin)'
        else:
            return '초기단계 (Early-stage)'

    refined_stats['segment'] = refined_stats.apply(classify_seller, axis=1)

    # ===== (추가) 상/하위 통합 세그먼트 (표시용) =====
    segment_view_map = {
        '핵심 판매자 (Core)': '상위판매자 (핵심판매자 & 박리다매형)',
        '박리다매형 (Low-Margin)': '상위판매자 (핵심판매자 & 박리다매형)',
        '불안정 성장 (Unstable)': '하위판매자 (불안정성장 & 초기단계)',
        '초기단계 (Early-stage)': '하위판매자 (불안정성장 & 초기단계)'
    }
    refined_stats['segment_view'] = refined_stats['segment'].map(segment_view_map)

    # --- Metrics Data Prep ---
    df = orders.copy()
    df = df.merge(customers[['customer_id', 'customer_state']], on='customer_id', how='left')

    df_items = order_items.merge(df, on='order_id', how='inner')
    df_items = df_items.merge(
        products[['product_id', 'product_weight_g', 'product_category_name_korean']],
        on='product_id',
        how='left'
    )
    df_items = df_items.merge(refined_stats[['segment', 'segment_view']], left_on='seller_id', right_index=True, how='inner')

    # Calculate Base Metrics
    df_items['handling_days'] = (df_items['order_delivered_carrier_date'] - df_items['order_approved_at']).dt.total_seconds() / (24 * 3600)
    df_items['delivery_days'] = (df_items['order_delivered_customer_date'] - df_items['order_approved_at']).dt.total_seconds() / (24 * 3600)
    df_items['is_delayed'] = df_items['order_delivered_customer_date'] > df_items['order_estimated_delivery_date']

    # Clean base
    df_clean = df_items[
        (df_items['handling_days'] >= 0) &
        (df_items['delivery_days'] >= 0) &
        (df_items['delivery_days'] < 100)
    ].copy()

    # ===== 지연 원인 분해: 운송 시간(transit_days) =====
    df_clean['transit_days'] = (
        df_clean['order_delivered_customer_date'] - df_clean['order_delivered_carrier_date']
    ).dt.total_seconds() / (24 * 3600)

    df_clean = df_clean[
        (df_clean['transit_days'].notna()) &
        (df_clean['transit_days'] >= 0) &
        (df_clean['transit_days'] < 100)
    ].copy()

    # ===== 무게 구간(weight_group) =====
    df_clean['weight_group'] = pd.cut(
        df_clean['product_weight_g'],
        bins=[-1, 500, 2000, 100000],
        labels=['경량(<=0.5kg)', '중량(0.5~2kg)', '대형(2kg+)']
    )

    # Aggregated metrics (원 세그먼트 기준: 개별 판매자 비교용)
    segment_agg = df_clean.groupby('segment').agg({
        'delivery_days': 'mean',
        'handling_days': 'mean',
        'transit_days': 'mean',
        'is_delayed': 'mean',
        'freight_value': 'mean'
    })

    return df_clean, refined_stats, segment_agg, sellers, products, segment_view_map


# Load Data
df_clean, refined_stats, segment_agg, sellers_raw, products_raw, segment_view_map = load_and_process_data()

# --- Layout ---
st.title("📦 판매자 세그먼트별 배송 분석 대시보드")
st.markdown("다중 세그먼트를 선택하여 배송 성과를 **직접 비교**하거나, 개별 판매자를 심층 분석할 수 있습니다.")

# --- Sidebar Filters ---
st.sidebar.header("🔍 분석 조건 설정")

base_segments = [
    '핵심 판매자 (Core)', '불안정 성장 (Unstable)',
    '박리다매형 (Low-Margin)', '초기단계 (Early-stage)'
]
group_segments = [
    '상위판매자 (핵심판매자 & 박리다매형)',
    '하위판매자 (불안정성장 & 초기단계)'
]
all_segments = group_segments + base_segments

# ===== (추가) 상/하위 선택 시 4개 세그먼트 자동 해제 UX =====
DEFAULT_SELECTION = ['상위판매자 (핵심판매자 & 박리다매형)']
STATE_KEY = "segment_selector"

if STATE_KEY not in st.session_state:
    st.session_state[STATE_KEY] = DEFAULT_SELECTION

def enforce_segment_selection():
    sel = st.session_state.get(STATE_KEY, [])
    has_group = any(s in group_segments for s in sel)
    has_base = any(s in base_segments for s in sel)

    # "상/하위"와 "원 세그먼트"가 같이 선택되면 → 상/하위만 남김
    if has_group and has_base:
        st.session_state[STATE_KEY] = [s for s in sel if s in group_segments]

selected_segments = st.sidebar.multiselect(
    "비교할 세그먼트 선택 (복수 선택 가능)",
    all_segments,
    key=STATE_KEY,
    on_change=enforce_segment_selection
)

# enforce가 바로 반영되도록(초기/예외 케이스 안전장치)
enforce_segment_selection()
selected_segments = st.session_state[STATE_KEY]

if not selected_segments:
    st.warning("분석할 세그먼트를 하나 이상 선택해주세요.")
    st.stop()

# --- 선택값에 따라 집계 기준 컬럼 결정 ---
use_group_view = any(s in group_segments for s in selected_segments)
SEG_COL = 'segment_view' if use_group_view else 'segment'

# Filter Data by Selected
filtered_stats = refined_stats[refined_stats[SEG_COL].isin(selected_segments)]
filtered_df = df_clean[df_clean[SEG_COL].isin(selected_segments)]

# Seller Selector
seller_options = ['전체 판매자'] + filtered_stats.index.tolist()
selected_seller_id = st.sidebar.selectbox("개별 판매자 분석 (선택 시 해당 판매자 Deep Dive)", seller_options)

st.divider()

# --- 1) Metric Summary Table ---
st.header(f"📊 세그먼트별 성과 요약 ({', '.join(selected_segments)})")

summary_df = filtered_df.groupby(SEG_COL)[['delivery_days', 'is_delayed', 'handling_days', 'transit_days', 'freight_value']].mean()
summary_df.columns = ['평균 배송 시간(일)', '지연율(Ratio)', '평균 처리 시간(일)', '평균 운송 시간(일)', '평균 배송비(R$)']
summary_df['지연율(%)'] = (summary_df['지연율(Ratio)'] * 100).round(1).astype(str) + '%'
summary_df['정시 배송율(%)'] = ((1 - summary_df['지연율(Ratio)']) * 100).round(1).astype(str) + '%'

display_cols = ['평균 배송 시간(일)', '지연율(%)', '정시 배송율(%)', '평균 처리 시간(일)', '평균 운송 시간(일)', '평균 배송비(R$)']
display_df = summary_df[display_cols].T
st.dataframe(display_df.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)

# --- 2) Chart Section ---
SEG_COLORS = {
    # 그룹(상/하위)
    '상위판매자 (핵심판매자 & 박리다매형)': '#1f77b4',
    '하위판매자 (불안정성장 & 초기단계)': '#7f7f7f',
    # 원 세그먼트
    '핵심 판매자 (Core)': '#1f77b4',
    '불안정 성장 (Unstable)': '#d62728',
    '박리다매형 (Low-Margin)': '#2ca02c',
    '초기단계 (Early-stage)': '#7f7f7f'
}
current_palette = {k: v for k, v in SEG_COLORS.items() if k in selected_segments}

# ===============
# CASE A: 개별 판매자 모드 (항상 원 세그먼트 기준으로 비교)
# ===============
if selected_seller_id != '전체 판매자':
    st.markdown("---")
    st.header(f"👤 개별 판매자 분석: `{selected_seller_id}`")

    my_data = df_clean[df_clean['seller_id'] == selected_seller_id]
    my_segment = refined_stats.loc[selected_seller_id, 'segment']  # 원 세그먼트
    my_segment_view = refined_stats.loc[selected_seller_id, 'segment_view']  # 상/하위

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    my_vals = {
        'W_Delivery': my_data['delivery_days'].mean(),
        'X_Delay': my_data['is_delayed'].mean(),
        'Y_Handling': my_data['handling_days'].mean(),
        'T_Transit': my_data['transit_days'].mean(),
        'Z_Freight': my_data['freight_value'].mean()
    }
    seg_vals = segment_agg.loc[my_segment]

    with col1:
        st.metric("내 배송 시간", f"{my_vals['W_Delivery']:.1f}일",
                  f"{my_vals['W_Delivery'] - seg_vals['delivery_days']:.1f}일 (vs {my_segment})",
                  delta_color="inverse")
    with col2:
        st.metric("내 지연율", f"{my_vals['X_Delay']*100:.1f}%",
                  f"{(my_vals['X_Delay'] - seg_vals['is_delayed'])*100:.1f}%p",
                  delta_color="inverse")
    with col3:
        st.metric("내 처리 시간", f"{my_vals['Y_Handling']:.1f}일",
                  f"{my_vals['Y_Handling'] - seg_vals['handling_days']:.1f}일",
                  delta_color="inverse")
    with col4:
        st.metric("내 운송 시간", f"{my_vals['T_Transit']:.1f}일",
                  f"{my_vals['T_Transit'] - seg_vals['transit_days']:.1f}일",
                  delta_color="inverse")
    with col5:
        st.metric("정시 배송율", f"{(1-my_vals['X_Delay'])*100:.1f}%",
                  f"{(((1-my_vals['X_Delay']) - (1-seg_vals['is_delayed']))*100):.1f}%p")
    with col6:
        st.metric("평균 배송비", f"R$ {my_vals['Z_Freight']:.1f}",
                  f"{my_vals['Z_Freight'] - seg_vals['freight_value']:.1f}",
                  delta_color="inverse")

    st.info(f"선택 판매자 원 세그먼트: **{my_segment}** / 통합 그룹: **{my_segment_view}**")

# ===============
# CASE B: 세그먼트 비교 모드
# ===============
else:
    st.markdown("---")
    st.header("📈 세그먼트 심층 비교")

    tab1, tab2, tabW, tab3 = st.tabs(["🚀 배송 시간 & 지연", "⚙️ 운영 & 처리 시간", "📦 무게 분석", "🌍 지역 & 비용"])

    # TAB 1
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("배송 시간 분포 비교")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(
                data=filtered_df, x='delivery_days', hue=SEG_COL,
                palette=current_palette, fill=True, common_norm=False, ax=ax
            )
            ax.set_xlim(0, 40)
            st.pyplot(fig)

        with c2:
            st.subheader("지연율 (Delay Rate)")
            agg_delay = filtered_df.groupby(SEG_COL)['is_delayed'].mean().reset_index()
            agg_delay['is_delayed_pct'] = agg_delay['is_delayed'] * 100

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=agg_delay, x=SEG_COL, y='is_delayed_pct',
                hue=SEG_COL, palette=current_palette, legend=False, ax=ax
            )
            add_bar_labels(ax, fmt="{:.1f}%", padding=3)
            ax.set_ylabel("지연율 (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            st.pyplot(fig)

    # TAB 2
    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("처리 시간 (Handling Time) 분포")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=filtered_df, x=SEG_COL, y='handling_days',
                hue=SEG_COL, palette=current_palette, legend=False,
                showfliers=False, ax=ax
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            st.pyplot(fig)

        with c2:
            st.subheader("운영 일관성 (처리 시간 표준편차)")
            agg_std = filtered_df.groupby(SEG_COL)['handling_days'].std().reset_index()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=agg_std, x=SEG_COL, y='handling_days',
                hue=SEG_COL, palette=current_palette, legend=False, ax=ax
            )
            add_bar_labels(ax, fmt="{:.2f}", padding=3)
            ax.set_ylabel("표준편차 (낮을수록 좋음)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            st.pyplot(fig)

        st.subheader("지연 원인 분해: 처리(승인→인계) vs 운송(인계→도착)")
        split_df = filtered_df.groupby(SEG_COL)[['handling_days', 'transit_days']].mean().reset_index()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(split_df[SEG_COL], split_df['handling_days'], label='처리(승인→인계)')
        ax.bar(split_df[SEG_COL], split_df['transit_days'], bottom=split_df['handling_days'], label='운송(인계→도착)')

        totals = (split_df['handling_days'] + split_df['transit_days']).values
        ax.bar_label(ax.containers[-1], labels=[f"{t:.1f}" for t in totals], padding=3, fontsize=9)

        ax.set_ylabel("평균 소요일(일)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
        ax.legend()
        st.pyplot(fig)

    # TAB W (무게)
    with tabW:
        st.subheader("📦 무게(Weight)별 배송 성과 비교")
        wdf = filtered_df.dropna(subset=['weight_group']).copy()

        st.markdown("#### 1) 무게 구간별 지연율")
        w_delay = wdf.groupby([SEG_COL, 'weight_group'])['is_delayed'].mean().reset_index()
        w_delay['delay_pct'] = w_delay['is_delayed'] * 100

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=w_delay, x='weight_group', y='delay_pct',
            hue=SEG_COL, palette=current_palette, ax=ax
        )
        add_bar_labels(ax, fmt="{:.1f}%", padding=2)
        ax.set_ylabel("지연율(%)")
        ax.set_xlabel("무게 구간")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
        st.pyplot(fig)

        st.markdown("#### 2) 무게 구간별 소요일 분해 (처리 vs 운송 vs 전체)")
        w_time = wdf.groupby([SEG_COL, 'weight_group'])[['handling_days', 'transit_days', 'delivery_days']].mean().reset_index()
        w_time_long = w_time.melt(
            id_vars=[SEG_COL, 'weight_group'],
            value_vars=['handling_days', 'transit_days', 'delivery_days'],
            var_name='time_type',
            value_name='days'
        )
        time_map = {
            'handling_days': '처리(승인→인계)',
            'transit_days': '운송(인계→도착)',
            'delivery_days': '전체(승인→도착)'
        }
        w_time_long['time_type'] = w_time_long['time_type'].map(time_map)

        g = sns.catplot(
            data=w_time_long,
            x='weight_group', y='days',
            hue=SEG_COL,
            col='time_type',
            kind='bar',
            palette=current_palette,
            height=3.6, aspect=1.05,
            sharey=False
        )
        g.set_axis_labels("무게 구간", "평균 소요일(일)")
        for ax_ in g.axes.flatten():
            ax_.set_xticklabels(ax_.get_xticklabels(), rotation=10)
            for container in ax_.containers:
                vals = getattr(container, "datavalues", None)
                if vals is None:
                    continue
                ax_.bar_label(container, labels=[f"{v:.1f}" for v in vals], padding=2, fontsize=8)
        st.pyplot(g.fig)

        st.markdown("#### 3) 무게 구간별 처리시간 변동성(표준편차)")
        w_std = wdf.groupby([SEG_COL, 'weight_group'])['handling_days'].std().reset_index()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=w_std, x='weight_group', y='handling_days',
            hue=SEG_COL, palette=current_palette, ax=ax
        )
        add_bar_labels(ax, fmt="{:.2f}", padding=2)
        ax.set_ylabel("처리시간 표준편차(일)  ※낮을수록 일관적")
        ax.set_xlabel("무게 구간")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
        st.pyplot(fig)

    # TAB 3 (지역 & 비용)
    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("배송비 분포 (Freight Value)")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(
                data=filtered_df, x='freight_value',
                hue=SEG_COL, palette=current_palette,
                fill=True, common_norm=False, ax=ax
            )
            ax.set_xlim(0, 100)
            st.pyplot(fig)

        with c2:
            st.subheader("장거리 배송 성과 (타지역 기준)")
            df_geo = filtered_df.merge(sellers_raw[['seller_id', 'seller_state']], on='seller_id', how='left')
            df_geo['is_interstate'] = df_geo['seller_state'] != df_geo['customer_state']

            interstate_only = df_geo[df_geo['is_interstate'] == True]
            inter_agg = interstate_only.groupby(SEG_COL)['delivery_days'].mean().reset_index()

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=inter_agg, x=SEG_COL, y='delivery_days',
                hue=SEG_COL, palette=current_palette, legend=False, ax=ax
            )
            add_bar_labels(ax, fmt="{:.1f}", padding=3)
            ax.set_title("타 지역(Inter-state) 배송 시 평균 소요 시간")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            st.pyplot(fig)

            st.markdown("#### 장거리(타지역) 조건별 지연율: 무게 구간")
            if 'weight_group' not in df_geo.columns:
                df_geo['weight_group'] = pd.cut(
                    df_geo['product_weight_g'],
                    bins=[-1, 500, 2000, 100000],
                    labels=['경량(<=0.5kg)', '중량(0.5~2kg)', '대형(2kg+)']
                )

            long_only = df_geo[df_geo['is_interstate'] == True].copy()
            cond = long_only.groupby([SEG_COL, 'weight_group'])['is_delayed'].mean().reset_index()
            cond['delay_pct'] = cond['is_delayed'] * 100

            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(
                data=cond, x='weight_group', y='delay_pct',
                hue=SEG_COL, palette=current_palette, ax=ax
            )
            add_bar_labels(ax, fmt="{:.1f}%", padding=2)
            ax.set_ylabel("지연율(%)")
            ax.set_xlabel("무게 구간")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
            st.pyplot(fig)

st.divider()
st.caption("※ 데이터 출처: Olist E-Commerce Dataset (Processed)")
