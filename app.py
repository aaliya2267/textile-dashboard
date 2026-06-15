import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from fpdf import FPDF

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="AI Textile Dashboard",
    page_icon="🧵",
    layout="wide"
)

# =====================================
# LOGIN SESSION
# =====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =====================================
# LOGIN PAGE
# =====================================

if not st.session_state.logged_in:

    st.title("🧵 Textile Dashboard Login")

    username = st.text_input("👤 Username")

    password = st.text_input(
        "🔒 Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "1234":

            st.session_state.logged_in = True
            st.rerun()

        else:

            st.error("❌ Wrong username or password")

# =====================================
# MAIN APPLICATION
# =====================================

else:

    st.sidebar.title("🧵 Textile Dashboard")

    # ---------------------------------
    # Upload CSV
    # ---------------------------------

    uploaded_file = st.sidebar.file_uploader(
        "📁 Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is None:

        st.title("📁 Upload Sales Data")

        st.info(
            "Please upload a CSV file from the sidebar to continue."
        )

        if st.sidebar.button("🚪 Logout"):

            st.session_state.logged_in = False
            st.rerun()

        st.stop()

    # ---------------------------------
    # Read Data
    # ---------------------------------

    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    try:
        data = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # ---------------------------------
    # Machine Learning Model
    # ---------------------------------

    @st.cache_resource
    def train_model(df):
        X = df[["Month"]]
        y = df["Sales"]
        m = LinearRegression()
        m.fit(X, y)
        return m

    model = train_model(data)

    next_month = data["Month"].max() + 1

    prediction = int(
        model.predict([[next_month]])[0]
    )

    # ---------------------------------
    # Dynamic Profit
    # ---------------------------------

    profit = int(data["Sales"].sum() * 10)

    # ---------------------------------
    # Sidebar Menu
    # ---------------------------------

    menu = st.sidebar.selectbox(
        "Choose Page",
        [
            "🏠 Home",
            "📈 Prediction",
            "📦 Inventory",
            "📄 Reports"
        ]
    )

    if st.sidebar.button("🚪 Logout"):

        st.session_state.logged_in = False
        st.rerun()

    # =====================================
    # HOME PAGE
    # =====================================

    if menu == "🏠 Home":

        st.title("🧵 AI Textile Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "📦 Total Stock",
            "1120"
        )

        col2.metric(
            "📈 Predicted Demand",
            prediction
        )

        col3.metric(
            "💰 Estimated Profit",
            f"₹{profit}"
        )

        tab1, tab2 = st.tabs(["📊 Dashboard Overview", "📋 Raw Data"])

        with tab1:
            st.subheader("📈 Monthly Sales Chart")
            fig = px.line(data, x="Month", y="Sales", markers=True, title="Monthly Sales Trend")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📋 Sales Data")
            st.dataframe(data)

    # =====================================
    # PREDICTION PAGE
    # =====================================

    elif menu == "📈 Prediction":

        st.title("📈 Demand Prediction")

        future_month = st.number_input(
            "Enter Future Month",
            min_value=1,
            value=int(next_month),
            step=1
        )

        future_prediction = model.predict(
            [[future_month]]
        )

        st.success(
            f"Predicted Sales: {int(future_prediction[0])}"
        )

    # =====================================
    # INVENTORY PAGE
    # =====================================

    elif menu == "📦 Inventory":

        st.title("📦 Inventory Management")

        st.subheader("Enter Stock Values")

        cotton = st.number_input(
            "Cotton Fabric Stock",
            min_value=0,
            value=250
        )

        silk = st.number_input(
            "Silk Fabric Stock",
            min_value=0,
            value=150
        )

        denim = st.number_input(
            "Denim Fabric Stock",
            min_value=0,
            value=300
        )

        polyester = st.number_input(
            "Polyester Fabric Stock",
            min_value=0,
            value=200
        )

        rayon = st.number_input(
            "Rayon Fabric Stock",
            min_value=0,
            value=100
        )

        wool = st.number_input(
            "Wool Fabric Stock",
            min_value=0,
            value=120
        )

        stock = (
            cotton
            + silk
            + denim
            + polyester
            + rayon
            + wool
        )

        st.subheader("📊 Inventory Analysis")

        st.write(
            f"📦 Total Stock = {stock}"
        )

        st.write(
            f"📈 Predicted Demand = {prediction}"
        )

        # Alert System

        if stock >= prediction + 200:

            st.success(
                "✅ Stock Sufficient"
            )

        elif stock >= prediction:

            st.warning(
                "⚠ Reorder Soon"
            )

        else:

            st.error(
                "🚨 Urgent Restock Needed"
            )

    # =====================================
    # REPORTS PAGE
    # =====================================

    elif menu == "📄 Reports":

        st.title("📄 Sales Reports")

        total_sales = data["Sales"].sum()

        average_sales = data["Sales"].mean()

        highest_sales = data["Sales"].max()

        lowest_sales = data["Sales"].min()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "💰 Total Sales",
                total_sales
            )

            st.metric(
                "📈 Highest Sales",
                highest_sales
            )

        with col2:

            st.metric(
                "📊 Average Sales",
                round(average_sales, 2)
            )

            st.metric(
                "📉 Lowest Sales",
                lowest_sales
            )

        st.subheader("📊 Sales Bar Chart")
        fig_bar = px.bar(data, x="Month", y="Sales", title="Sales by Month", color="Sales", color_continuous_scale="Blues")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📋 Dataset Used")
        with st.expander("View Raw Data"):
            st.dataframe(data)

        # Download PDF
        def generate_pdf():
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_fill_color(30, 30, 47)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", 'B', 24)
            pdf.cell(0, 15, text="Professional Sales Report", border=0, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(10)
            
            # Executive Summary
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", 'B', 14)
            pdf.cell(0, 10, text="Executive Summary", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=12)
            
            pdf.cell(95, 10, text=f"Total Sales: {total_sales}", border=1)
            pdf.cell(95, 10, text=f"Average Sales: {round(average_sales, 2)}", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.cell(95, 10, text=f"Highest Sales: {highest_sales}", border=1)
            pdf.cell(95, 10, text=f"Lowest Sales: {lowest_sales}", border=1, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(10)
            
            # Table Header
            pdf.set_fill_color(75, 108, 183)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(95, 10, text="Month", border=1, align='C', fill=True)
            pdf.cell(95, 10, text="Sales Volume", border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
            
            # Table Data
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", size=12)
            for i in range(len(data)):
                # Alternating row colors
                if i % 2 == 0:
                    pdf.set_fill_color(245, 247, 250)
                else:
                    pdf.set_fill_color(255, 255, 255)
                    
                pdf.cell(95, 10, text=str(data.iloc[i]['Month']), border=1, align='C', fill=True)
                pdf.cell(95, 10, text=str(data.iloc[i]['Sales']), border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
                
            return bytes(pdf.output())

        pdf_bytes = generate_pdf()

        st.download_button(
            label="⬇ Download PDF Report",
            data=pdf_bytes,
            file_name="sales_report.pdf",
            mime="application/pdf"
        )