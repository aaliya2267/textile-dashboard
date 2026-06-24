import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# =====================================
# COMPANY INFO
# =====================================

COMPANY_NAME    = "Bapuji Textile Industries"
COMPANY_CITY    = "Davanagere, Karnataka - 577002"
COMPANY_TAGLINE = "Quality Fabrics | Trusted Since Decades"

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Bapuji Textile – AI Sales Dashboard",
    page_icon="🧵",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .company-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(26,35,126,0.4);
    }
    .company-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .company-header p  { font-size: 0.95rem; opacity: 0.85; margin: 4px 0 0; }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-left: 4px solid #3949ab;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    .alert-downfall {
        background: #fff3e0;
        border-left: 5px solid #ff6f00;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .alert-downfall h4 { color: #e65100; margin: 0 0 6px; }
    .alert-downfall p  { color: #4e342e; margin: 0; font-size: 0.92rem; }

    .alert-growth {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .alert-growth h4 { color: #1b5e20; margin: 0 0 6px; }
    .alert-growth p  { color: #1b5e20; margin: 0; font-size: 0.92rem; }

    .reason-box {
        background: #fce4ec;
        border-left: 5px solid #c62828;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .reason-box h5 { color: #b71c1c; margin: 0 0 4px; font-size: 0.95rem; }
    .reason-box p  { color: #4a148c; margin: 0; font-size: 0.88rem; }

    .insight-box {
        background: #e3f2fd;
        border-left: 5px solid #1565c0;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .insight-box h5 { color: #0d47a1; margin: 0 0 4px; font-size: 0.95rem; }
    .insight-box p  { color: #1a237e; margin: 0; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# =====================================
# LOGIN SESSION
# =====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =====================================
# LOGIN PAGE
# =====================================

if not st.session_state.logged_in:

    st.markdown(f"""
    <div class="company-header">
        <h1>🧵 {COMPANY_NAME}</h1>
        <p>📍 {COMPANY_CITY} &nbsp;|&nbsp; {COMPANY_TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 1.5, 1])
    with col_c:
        st.subheader("🔐 Dashboard Login")
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")

        if st.button("🚀 Login", width='stretch'):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password. Try admin / 1234")

        st.caption("Default credentials → Username: `admin` | Password: `1234`")

# =====================================
# MAIN APPLICATION
# =====================================

else:

    # ---------------------------------
    # Sidebar Header
    # ---------------------------------

    st.sidebar.markdown(f"""
    <div style="
        background: linear-gradient(135deg,#1a237e,#3949ab);
        color:white; padding:14px 12px; border-radius:10px; text-align:center; margin-bottom:10px;
    ">
        <b style='font-size:1rem'>🧵 {COMPANY_NAME}</b><br>
        <small style='opacity:.8'>{COMPANY_CITY}</small>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------
    # Upload CSV
    # ---------------------------------

    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV File (Month, Sales)",
        type=["csv"],
        help="Upload a dataset with 'Month' and 'Sales' columns"
    )

    if uploaded_file is None:
        st.markdown(f"""
        <div class="company-header">
            <h1>🧵 {COMPANY_NAME}</h1>
            <p>Upload a CSV file in the sidebar to generate the analytics dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ---------------------------------
    # Read & Validate Data
    # ---------------------------------

    import os

    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df

    try:
        data = load_data(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()

    # Validate required columns
    required_cols = ["Month", "Sales"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        st.error(f"❌ CSV is missing required columns: {missing}. Please upload a file with **Month** and **Sales** columns.")
        st.stop()

    # Drop rows with nulls in key columns
    data = data.dropna(subset=required_cols).reset_index(drop=True)

    if len(data) < 3:
        st.error("❌ Not enough data rows (need at least 3). Please upload a richer dataset.")
        st.stop()

    # Ensure numeric types
    data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
    data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")
    data = data.dropna(subset=required_cols).reset_index(drop=True)

    # ---------------------------------
    # Machine Learning Models
    # ---------------------------------

    @st.cache_resource
    def train_models(df):
        X = df[["Month"]].values
        y = df["Sales"].values

        # Linear model
        lin_model = LinearRegression()
        lin_model.fit(X, y)

        # Polynomial model (degree=2) for better fit
        poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_model.fit(X, y)

        return lin_model, poly_model

    lin_model, poly_model = train_models(data)

    next_month = int(data["Month"].max()) + 1

    lin_pred = int(lin_model.predict([[next_month]])[0])
    poly_pred = int(poly_model.predict([[next_month]])[0])

    # Use linear prediction as primary
    prediction = lin_pred

    # R² score for linear model
    y_true = data["Sales"].values
    y_hat  = lin_model.predict(data[["Month"]].values)
    r2     = round(r2_score(y_true, y_hat), 3)
    mae    = round(mean_absolute_error(y_true, y_hat), 2)

    # Trend analysis
    slope = lin_model.coef_[0]
    recent_sales = data["Sales"].tail(3).values
    recent_avg   = recent_sales.mean()
    overall_avg  = data["Sales"].mean()
    growth_rate  = ((data["Sales"].iloc[-1] - data["Sales"].iloc[0]) / data["Sales"].iloc[0]) * 100

    # Month-over-month changes (last 3 months)
    data["MoM_Change"]   = data["Sales"].diff()
    data["MoM_Pct"]      = data["Sales"].pct_change() * 100
    recent_mom_avg       = data["MoM_Pct"].tail(3).mean()

    # ---------------------------------
    # Prediction Reasoning Engine
    # ---------------------------------

    def get_prediction_reasons(data, slope, recent_avg, overall_avg, recent_mom_avg, prediction):
        """
        Generates human-readable reasons explaining why sales might rise or fall.
        Returns (is_downfall: bool, reasons: list[dict])
        """
        reasons = []
        last_sales   = data["Sales"].iloc[-1]
        prev_sales   = data["Sales"].iloc[-2] if len(data) >= 2 else last_sales
        mom_last     = data["MoM_Pct"].iloc[-1] if len(data) >= 2 else 0
        mom_prev     = data["MoM_Pct"].iloc[-2] if len(data) >= 3 else mom_last

        is_downfall = prediction < last_sales

        # ---- Reason 1: Slowdown in growth momentum ----
        if recent_mom_avg < 5:
            reasons.append({
                "title": "📉 Slowing Growth Momentum",
                "detail": (
                    f"The average month-over-month sales growth over the last 3 months is only "
                    f"{recent_mom_avg:.1f}%. A growth rate below 5% signals that demand expansion "
                    f"is decelerating, which often precedes a plateau or dip."
                ),
                "type": "risk"
            })

        # ---- Reason 2: Recent acceleration reversal ----
        if mom_last < mom_prev - 5:
            reasons.append({
                "title": "🔄 Growth Rate Deceleration",
                "detail": (
                    f"Last month's growth rate ({mom_last:.1f}%) dropped significantly compared to "
                    f"the previous month ({mom_prev:.1f}%). A sharp deceleration in growth rate "
                    f"is a leading indicator of upcoming sales weakness."
                ),
                "type": "risk"
            })

        # ---- Reason 3: Recent avg vs overall avg ----
        if recent_avg < overall_avg * 0.95:
            reasons.append({
                "title": "📊 Recent Sales Below Historical Average",
                "detail": (
                    f"The 3-month recent average sales (₹{recent_avg:.0f}) is below the "
                    f"overall historical average (₹{overall_avg:.0f}). This indicates the market "
                    f"may be cooling off from its peak performance."
                ),
                "type": "risk"
            })

        # ---- Reason 4: Low R² (unreliable upward trend) ----
        if r2 < 0.85:
            reasons.append({
                "title": "📐 High Sales Volatility / Irregular Trend",
                "detail": (
                    f"The model's R² accuracy is {r2} (below 0.85), indicating that sales data "
                    f"has irregular fluctuations and does not follow a clean upward trend. "
                    f"Volatile historical patterns often lead to uncertain or declining forecasts."
                ),
                "type": "risk"
            })

        # ---- Reason 5: Seasonal/peak concern ----
        if len(data) >= 12:
            seasonal_peak = data["Sales"].rolling(window=3).mean().idxmax()
            curr_idx = len(data) - 1
            if curr_idx - seasonal_peak <= 2:
                reasons.append({
                    "title": "🌦 Post-Peak Seasonal Correction",
                    "detail": (
                        f"Sales appear to have recently peaked (around month {data.iloc[seasonal_peak]['Month']:.0f}). "
                        f"Post-peak periods in textile markets typically experience demand corrections "
                        f"of 10–20% as inventory restocking slows and off-season buying reduces."
                    ),
                    "type": "risk"
                })

        # ---- Reason 6: Positive reasons (if growth) ----
        if slope > 10:
            reasons.append({
                "title": "📈 Strong Upward Trend",
                "detail": (
                    f"The linear trend shows sales increasing by ₹{slope:.1f} per month on average. "
                    f"This sustained positive slope indicates growing market demand for Bapuji Textile products."
                ),
                "type": "positive"
            })

        if recent_mom_avg >= 8:
            reasons.append({
                "title": "🚀 Accelerating Demand",
                "detail": (
                    f"The 3-month average growth rate of {recent_mom_avg:.1f}% shows strong and "
                    f"accelerating customer demand. Continued growth at this pace is expected."
                ),
                "type": "positive"
            })

        # Default if no strong signals
        if not reasons:
            reasons.append({
                "title": "📌 Stable Market Conditions",
                "detail": (
                    "Sales data shows relatively stable conditions with no strong positive or negative "
                    "signals. The prediction is based on linear extrapolation of the current trend."
                ),
                "type": "neutral"
            })

        return is_downfall, reasons

    is_downfall, pred_reasons = get_prediction_reasons(
        data, slope, recent_avg, overall_avg, recent_mom_avg, prediction
    )

    # ---------------------------------
    # Dynamic Profit Estimate
    # ---------------------------------

    profit = int(data["Sales"].sum() * 10)

    # ---------------------------------
    # Sidebar Menu
    # ---------------------------------

    menu = st.sidebar.selectbox(
        "📂 Choose Page",
        [
            "🏠 Home",
            "📈 Prediction",
            "📦 Inventory",
            "📄 Reports"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"📊 Dataset: {len(data)} months of data")
    st.sidebar.caption(f"🤖 Model R²: {r2} | MAE: {mae}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # =====================================
    # COMPANY HEADER (shown on all pages)
    # =====================================

    st.markdown(f"""
    <div class="company-header">
        <h1>🧵 {COMPANY_NAME}</h1>
        <p>📍 {COMPANY_CITY} &nbsp;|&nbsp; {COMPANY_TAGLINE}</p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================
    # HOME PAGE
    # =====================================

    if menu == "🏠 Home":

        st.subheader("📊 Sales Overview Dashboard")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "📈 Predicted Next Month",
            f"₹{prediction:,}",
            delta=f"{prediction - int(data['Sales'].iloc[-1]):+,} vs last month"
        )

        col2.metric(
            "💰 Estimated Total Profit",
            f"₹{profit:,}"
        )

        col3.metric(
            "📊 Avg Monthly Sales",
            f"₹{int(overall_avg):,}"
        )

        col4.metric(
            "📉 Overall Growth",
            f"{growth_rate:.1f}%"
        )

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📊 Sales Trend", "📋 Raw Data", "🔍 Analysis Summary"])

        with tab1:
            st.subheader("📈 Monthly Sales Trend with Prediction")

            # Build forecast line
            future_months = list(range(int(data["Month"].max()) + 1, int(data["Month"].max()) + 4))
            future_sales  = [int(lin_model.predict([[m]])[0]) for m in future_months]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data["Month"], y=data["Sales"],
                mode="lines+markers",
                name="Actual Sales",
                line=dict(color="#3949ab", width=2.5),
                marker=dict(size=7)
            ))
            fig.add_trace(go.Scatter(
                x=[data["Month"].max()] + future_months,
                y=[data["Sales"].iloc[-1]] + future_sales,
                mode="lines+markers",
                name="Forecasted Sales",
                line=dict(color="#ff6f00", width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond")
            ))
            fig.update_layout(
                title=f"{COMPANY_NAME} – Monthly Sales Trend",
                xaxis_title="Month",
                yaxis_title="Sales (₹)",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
                plot_bgcolor="rgba(245,247,250,0.6)"
            )
            st.plotly_chart(fig, width='stretch')

            # MoM change chart
            st.subheader("📉 Month-over-Month Growth Rate (%)")
            mom_fig = px.bar(
                data.dropna(subset=["MoM_Pct"]),
                x="Month", y="MoM_Pct",
                color="MoM_Pct",
                color_continuous_scale=["#c62828", "#ffb300", "#2e7d32"],
                title="Monthly Growth Rate (%)",
                labels={"MoM_Pct": "Growth (%)"}
            )
            mom_fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(mom_fig, width='stretch')

        with tab2:
            st.subheader("📋 Sales Data Table")
            display_data = data[["Month", "Sales", "MoM_Change", "MoM_Pct"]].copy()
            display_data.columns = ["Month", "Sales (₹)", "MoM Change (₹)", "MoM Growth (%)"]
            st.dataframe(display_data.style.format({
                "Sales (₹)": "{:,.0f}",
                "MoM Change (₹)": "{:+,.0f}",
                "MoM Growth (%)": "{:+.1f}%"
            }), width='stretch')

        with tab3:
            st.subheader("🔍 Business Intelligence Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <b>📈 Linear Trend Slope</b><br>
                    ₹{slope:.2f} per month
                </div>
                <div class="metric-card">
                    <b>🎯 Model Accuracy (R²)</b><br>
                    {r2} ({('Excellent' if r2 > 0.95 else 'Good' if r2 > 0.85 else 'Moderate')})
                </div>
                <div class="metric-card">
                    <b>📉 Mean Absolute Error</b><br>
                    ₹{mae:,.2f}
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <b>📊 Recent 3-Month Avg Sales</b><br>
                    ₹{recent_avg:,.0f}
                </div>
                <div class="metric-card">
                    <b>🔄 Avg MoM Growth (3 months)</b><br>
                    {recent_mom_avg:.1f}%
                </div>
                <div class="metric-card">
                    <b>📅 Data Span</b><br>
                    Month {int(data['Month'].min())} – Month {int(data['Month'].max())} ({len(data)} records)
                </div>
                """, unsafe_allow_html=True)

    # =====================================
    # PREDICTION PAGE
    # =====================================

    elif menu == "📈 Prediction":

        st.subheader("📈 AI-Powered Demand Prediction")

        # Input future month
        future_month = st.number_input(
            "🗓 Enter Future Month Number",
            min_value=1,
            value=int(next_month),
            step=1,
            help="Enter the month number you want to predict sales for."
        )

        future_prediction = int(lin_model.predict([[future_month]])[0])
        last_actual       = int(data["Sales"].iloc[-1])
        diff              = future_prediction - last_actual
        pct_change        = (diff / last_actual) * 100 if last_actual != 0 else 0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("🗓 Month Predicted", f"Month {future_month}")
        col_b.metric("📊 Predicted Sales", f"₹{future_prediction:,}", delta=f"{diff:+,} ({pct_change:+.1f}%)")
        col_c.metric("📐 Model Confidence (R²)", str(r2))

        st.markdown("---")

        # Dynamic reasoning
        is_down, reasons = get_prediction_reasons(
            data, slope, recent_avg, overall_avg, recent_mom_avg, future_prediction
        )

        if future_prediction < last_actual:
            st.markdown(f"""
            <div class="alert-downfall">
                <h4>⚠ Sales Downfall Predicted for Month {future_month}</h4>
                <p>Predicted sales (₹{future_prediction:,}) are <b>{abs(diff):,} ({abs(pct_change):.1f}%) lower</b>
                than last month's actual sales (₹{last_actual:,}).</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-growth">
                <h4>✅ Sales Growth Predicted for Month {future_month}</h4>
                <p>Predicted sales (₹{future_prediction:,}) are <b>{abs(diff):,} ({abs(pct_change):.1f}%) higher</b>
                than last month's actual sales (₹{last_actual:,}).</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🧠 Why This Prediction? (Root Cause Analysis)")
        st.caption("The AI model analyzed your historical data and identified the following key factors driving this forecast:")

        risk_count = 0
        pos_count  = 0
        for r in reasons:
            if r["type"] == "risk":
                risk_count += 1
                st.markdown(f"""
                <div class="reason-box">
                    <h5>🔴 {r['title']}</h5>
                    <p>{r['detail']}</p>
                </div>
                """, unsafe_allow_html=True)
            elif r["type"] == "positive":
                pos_count += 1
                st.markdown(f"""
                <div class="alert-growth">
                    <h4>🟢 {r['title']}</h4>
                    <p>{r['detail']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="insight-box">
                    <h5>🔵 {r['title']}</h5>
                    <p>{r['detail']}</p>
                </div>
                """, unsafe_allow_html=True)

        # Risk Summary
        st.markdown("---")
        st.markdown("### 📋 Risk vs Opportunity Summary")
        rc1, rc2 = st.columns(2)
        rc1.error(f"🔴 Risk Factors Identified: **{risk_count}**")
        rc2.success(f"🟢 Positive Signals Identified: **{pos_count}**")

        # Recommendations
        st.markdown("### 💡 Recommendations for Management")
        if risk_count > pos_count:
            st.warning(
                "⚠ **Action Required:** Multiple downfall risk factors detected. "
                "Recommended actions:\n"
                "- 🔁 **Review Procurement:** Reduce raw material orders for next quarter.\n"
                "- 🎯 **Boost Marketing:** Launch targeted promotions in Davanagere and surrounding districts.\n"
                "- 📦 **Manage Inventory:** Avoid overstocking — match stock levels to predicted demand.\n"
                "- 💼 **Review Product Mix:** Focus on high-margin fabrics like Silk and Wool.\n"
                "- 🔎 **Monitor Competitors:** Check if competitors are offering better pricing."
            )
        else:
            st.success(
                "✅ **Positive Outlook:** Growth signals are strong. "
                "Recommended actions:\n"
                "- 📈 **Scale Production:** Increase fabric production to meet rising demand.\n"
                "- 🏪 **Expand Distribution:** Consider opening new retail touchpoints near Davanagere.\n"
                "- 💰 **Invest in Quality:** Use growth period profits to upgrade weaving machinery.\n"
                "- 🤝 **Partner with Retailers:** Lock in bulk supply agreements at current favorable rates."
            )

        # Prediction chart
        st.markdown("---")
        st.subheader("📊 Sales Forecast Visualization")
        past_months  = list(data["Month"])
        past_sales   = list(data["Sales"])
        pred_months  = list(range(int(data["Month"].max()) + 1, future_month + 1))
        pred_sales   = [int(lin_model.predict([[m]])[0]) for m in pred_months]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=past_months, y=past_sales,
            mode="lines+markers", name="Actual Sales",
            line=dict(color="#3949ab", width=2.5)
        ))
        if pred_months:
            fig2.add_trace(go.Scatter(
                x=[past_months[-1]] + pred_months,
                y=[past_sales[-1]] + pred_sales,
                mode="lines+markers", name="Forecasted",
                line=dict(color="#e53935", width=2, dash="dot"),
                marker=dict(size=8, symbol="diamond")
            ))
        fig2.update_layout(
            title=f"Bapuji Textile – Sales Forecast up to Month {future_month}",
            xaxis_title="Month", yaxis_title="Sales (₹)",
            hovermode="x unified", plot_bgcolor="rgba(245,247,250,0.6)"
        )
        st.plotly_chart(fig2, width='stretch')

    # =====================================
    # INVENTORY PAGE
    # =====================================

    elif menu == "📦 Inventory":

        st.subheader("📦 Inventory Management")

        st.markdown("#### Enter Current Stock Values (in metres / units)")

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            cotton    = st.number_input("🌿 Cotton Fabric Stock",    min_value=0, value=250, step=10)
            silk      = st.number_input("🪡 Silk Fabric Stock",      min_value=0, value=150, step=10)
            denim     = st.number_input("👖 Denim Fabric Stock",     min_value=0, value=300, step=10)
        with col_i2:
            polyester = st.number_input("🧪 Polyester Fabric Stock", min_value=0, value=200, step=10)
            rayon     = st.number_input("🌸 Rayon Fabric Stock",     min_value=0, value=100, step=10)
            wool      = st.number_input("🐑 Wool Fabric Stock",      min_value=0, value=120, step=10)

        stock = cotton + silk + denim + polyester + rayon + wool

        st.markdown("---")
        st.subheader("📊 Inventory Analysis")

        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("📦 Total Stock", f"{stock:,} units")
        ic2.metric("📈 Predicted Demand", f"₹{prediction:,}")
        shortage = max(0, prediction - stock)
        surplus  = max(0, stock - prediction)
        ic3.metric("📊 Stock vs Demand", f"{'+' if surplus else '-'}{surplus or shortage:,}", delta_color="normal")

        # Stock Composition Chart
        st.subheader("📊 Inventory Composition")
        fabric_names  = ["Cotton", "Silk", "Denim", "Polyester", "Rayon", "Wool"]
        fabric_values = [cotton, silk, denim, polyester, rayon, wool]
        inv_fig = px.pie(
            names=fabric_names, values=fabric_values,
            title="Fabric Stock Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.35
        )
        inv_fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(inv_fig, width='stretch')

        # Alert System
        st.subheader("🚨 Stock Alert")
        if stock >= prediction + 200:
            st.success("✅ Stock Sufficient — Inventory levels are healthy relative to predicted demand.")
        elif stock >= prediction:
            st.warning("⚠ Reorder Soon — Stock is adequate for now but will run low. Plan procurement.")
        elif stock >= prediction * 0.7:
            st.error(f"🚨 Low Stock Warning — Only {stock:,} units available vs ₹{prediction:,} demand. Restock immediately.")
        else:
            st.error(f"🔴 Critical Stock Shortage! Stock ({stock:,}) is well below demand ({prediction:,}). Urgent action needed.")

        # Per-fabric alert
        st.subheader("📋 Per-Fabric Status")
        fabric_df = pd.DataFrame({
            "Fabric": fabric_names,
            "Stock": fabric_values,
            "Status": ["✅ OK" if v >= 100 else "⚠ Low" if v >= 50 else "🔴 Critical" for v in fabric_values]
        })
        st.dataframe(fabric_df, width='stretch')

    # =====================================
    # REPORTS PAGE
    # =====================================

    elif menu == "📄 Reports":

        st.subheader("📄 Sales Reports & PDF Export")

        total_sales   = data["Sales"].sum()
        average_sales = data["Sales"].mean()
        highest_sales = data["Sales"].max()
        lowest_sales  = data["Sales"].min()
        best_month    = int(data.loc[data["Sales"].idxmax(), "Month"])
        worst_month   = int(data.loc[data["Sales"].idxmin(), "Month"])

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("💰 Total Sales",   f"₹{total_sales:,.0f}")
        rc2.metric("📊 Average Sales", f"₹{average_sales:,.0f}")
        rc3.metric("📈 Highest Month", f"Month {best_month}  (₹{highest_sales:,.0f})")
        rc4.metric("📉 Lowest Month",  f"Month {worst_month} (₹{lowest_sales:,.0f})")

        st.markdown("---")

        # Bar Chart
        st.subheader("📊 Sales by Month")
        fig_bar = px.bar(
            data, x="Month", y="Sales",
            title=f"{COMPANY_NAME} – Sales by Month",
            color="Sales",
            color_continuous_scale="Blues",
            labels={"Sales": "Sales (₹)"}
        )
        st.plotly_chart(fig_bar, width='stretch')

        # Sales data expander
        st.subheader("📋 Dataset")
        with st.expander("View Raw Data"):
            st.dataframe(data[["Month", "Sales"]], width='stretch')

        st.markdown("---")

        # ---------------------------------
        # Download PDF Report
        # ---------------------------------

        def generate_pdf():
            from datetime import datetime
            import os

            class PDF(FPDF):
                def header(self):
                    # ---- Professional Corporate Header ----
                    # Logo
                    logo_path = "logo.png"
                    if os.path.exists(logo_path):
                        self.image(logo_path, x=10, y=8, w=20)
                    
                    # Company Name (Right Aligned)
                    self.set_y(10)
                    self.set_font("Helvetica", 'B', 20)
                    self.set_text_color(26, 35, 126)   # Deep indigo
                    self.cell(0, 8, text=COMPANY_NAME, align='R', new_x="LMARGIN", new_y="NEXT")
                    
                    # Company Address & Tagline
                    self.set_font("Helvetica", size=9)
                    self.set_text_color(100, 100, 100)
                    self.cell(0, 5, text=f"{COMPANY_CITY} | {COMPANY_TAGLINE}", align='R', new_x="LMARGIN", new_y="NEXT")
                    self.cell(0, 5, text="Phone: +91-8192-XXXXXX | Email: management@bapujitextile.com", align='R', new_x="LMARGIN", new_y="NEXT")
                    
                    # Separator Line (push down slightly)
                    self.set_y(32)
                    self.set_draw_color(26, 35, 126)
                    self.set_line_width(0.8)
                    self.line(10, self.get_y(), 200, self.get_y())
                    self.set_line_width(0.2) # reset
                    self.set_y(36)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Helvetica", 'I', 8)
                    self.set_text_color(150, 150, 150)
                    self.cell(0, 10, text=f"Page {self.page_no()}", align='C', new_x="LMARGIN")
                    self.set_y(-15)
                    self.cell(0, 10, text="CONFIDENTIAL - Bapuji Textile Industries", align='L', new_x="LMARGIN")
                    self.set_y(-15)
                    self.cell(0, 10, text="Generated by AI Analytics Dashboard", align='R', new_x="LMARGIN", new_y="NEXT")

            pdf = PDF()
            pdf.add_page()
            
            # ---- Report Title Block ----
            pdf.set_y(40)
            pdf.set_font("Helvetica", 'B', 22)
            pdf.set_text_color(26, 35, 126)
            pdf.cell(0, 12, text="SALES & INVENTORY ANALYTICS REPORT", align='C', new_x="LMARGIN", new_y="NEXT")
            
            pdf.set_font("Helvetica", size=11)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, text=f"Report Date: {datetime.now().strftime('%B %d, %Y')}", align='C', new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 6, text="Prepared for: Executive Management Board", align='C', new_x="LMARGIN", new_y="NEXT")
            pdf.ln(8)

            # ---- Executive Summary ----
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", 'B', 13)
            pdf.cell(0, 9, text="Executive Summary", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)

            summary_items = [
                ("Total Sales",           f"Rs. {total_sales:,.0f}"),
                ("Average Monthly Sales",  f"Rs. {average_sales:,.0f}"),
                ("Highest Sales (Month)",  f"Rs. {highest_sales:,.0f}  (Month {best_month})"),
                ("Lowest Sales (Month)",   f"Rs. {lowest_sales:,.0f}  (Month {worst_month})"),
                ("Overall Growth",         f"{growth_rate:.1f}%"),
                ("Next Month Prediction",  f"Rs. {prediction:,.0f}"),
                ("Model Accuracy (R2)",    str(r2)),
            ]
            for label, value in summary_items:
                pdf.set_fill_color(240, 242, 250)
                pdf.cell(95, 8, text=label, border=1, fill=True, new_x="RIGHT")
                pdf.cell(95, 8, text=value, border=1, fill=False, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(6)

            # ---- Prediction & Reasoning ----
            pdf.set_font("Helvetica", 'B', 13)
            pdf.cell(0, 9, text="AI Prediction Analysis", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)

            if is_downfall:
                pdf.set_fill_color(255, 235, 210)
                pdf.set_text_color(180, 50, 0)
                pdf.cell(0, 9, text=f"WARNING: Sales Downfall Predicted - Next Month Forecast: Rs. {prediction:,}", border=1, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_fill_color(200, 240, 210)
                pdf.set_text_color(20, 100, 30)
                pdf.cell(0, 9, text=f"POSITIVE: Sales Growth Predicted - Next Month Forecast: Rs. {prediction:,}", border=1, fill=True, align='C', new_x="LMARGIN", new_y="NEXT")

            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            pdf.set_font("Helvetica", 'B', 11)
            pdf.cell(0, 8, text="Key Reasons for Prediction:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)

            for idx, r in enumerate(pred_reasons, 1):
                icon = "[RISK]" if r["type"] == "risk" else "[POSITIVE]" if r["type"] == "positive" else "[INFO]"
                pdf.set_font("Helvetica", 'B', 10)
                safe_title = r['title'].encode('latin-1', 'ignore').decode('latin-1').strip()
                pdf.multi_cell(0, 6, text=f"{idx}. {icon} {safe_title}", border=0, new_x="LMARGIN", new_y="NEXT")
                
                pdf.set_font("Helvetica", size=9)
                detail_text = r["detail"].replace("₹", "Rs.").encode('latin-1', 'ignore').decode('latin-1').strip()
                
                # Indent detail text explicitly
                pdf.set_x(15)
                pdf.multi_cell(185, 5, text=detail_text, border=0, new_x="LMARGIN", new_y="NEXT")
                pdf.set_x(10)
                pdf.ln(2)
            pdf.ln(4)

            # ---- Monthly Data Table ----
            pdf.set_font("Helvetica", 'B', 13)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 9, text="Monthly Sales Data", new_x="LMARGIN", new_y="NEXT")

            # Table Header
            pdf.set_fill_color(57, 73, 171)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", 'B', 11)
            pdf.cell(63, 8, text="Month",          border=1, align='C', fill=True, new_x="RIGHT")
            pdf.cell(64, 8, text="Sales (Rs.)",    border=1, align='C', fill=True, new_x="RIGHT")
            pdf.cell(63, 8, text="MoM Change",     border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")

            # Table Rows
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", size=10)
            for i in range(len(data)):
                if i % 2 == 0:
                    pdf.set_fill_color(245, 247, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                month_val = str(int(data.iloc[i]['Month']))
                sales_val = f"Rs. {data.iloc[i]['Sales']:,.0f}"
                mom_val   = (
                    f"{data.iloc[i]['MoM_Change']:+,.0f}"
                    if pd.notna(data.iloc[i]['MoM_Change']) else "-"
                )
                pdf.cell(63, 8, text=month_val, border=1, align='C', fill=True, new_x="RIGHT")
                pdf.cell(64, 8, text=sales_val, border=1, align='C', fill=True, new_x="RIGHT")
                pdf.cell(63, 8, text=mom_val,   border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")

            return bytes(pdf.output())

        pdf_bytes = generate_pdf()

        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name="bapuji_textile_sales_report.pdf",
            mime="application/pdf",
            width='stretch'
        )

        st.success("✅ PDF report includes company header, executive summary, AI prediction reasoning, and full monthly data table.")
