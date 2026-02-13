"""
Streamlit UI for E-commerce Quality Control System
Dynamically accepts any order features
"""
import streamlit as st
import requests
import json
from PIL import Image
import io
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configuration
API_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title="E-commerce QC System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .risk-high { background-color: #ffebee; border-left: 5px solid #f44336; padding: 15px; border-radius: 5px; }
    .risk-medium { background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; border-radius: 5px; }
    .risk-low { background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 15px; border-radius: 5px; }
    .metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    .flag-badge { background-color: #f44336; color: white; padding: 5px 10px; border-radius: 15px; font-size: 0.8rem; margin: 2px; display: inline-block; }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def display_risk_card(risk_score, assessment):
    """Display risk assessment card"""
    if assessment == "HIGH_RISK":
        risk_class, icon, color = "risk-high", "⚠️", "#f44336"
    elif assessment == "MEDIUM_RISK":
        risk_class, icon, color = "risk-medium", "⚡", "#ff9800"
    else:
        risk_class, icon, color = "risk-low", "✅", "#4caf50"
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h2>{icon} {assessment.replace('_', ' ')}</h2>
        <p style="font-size: 1.5rem; margin: 10px 0;">Risk Score: {risk_score:.1%}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🛡️ E-commerce Quality Control</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Flexible System - Add Any Features You Need!</p>", unsafe_allow_html=True)

# Check API status
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    api_healthy, health_data = check_api_health()
    if api_healthy:
        st.success(f" API Connected | {health_data.get('total_models', 0)} Models Active", icon="✅")
    else:
        st.error(" API Not Running - Please start the backend server", icon="❌")
        st.code("python qc_api_flexible.py", language="bash")
        st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### 🎯 Analysis Type")
    
    analysis_mode = st.radio(
        "Select what to analyze:",
        ["📦 Order Analysis", "📸 Image Analysis", "🔄 Combined Analysis", "📊 Batch Analysis", "🔧 Custom JSON"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ✨ Flexible Features")
    st.success("""
    This system is **completely flexible**!
    
    ✅ Add any features
    ✅ Remove features
    ✅ No hardcoded fields
    ✅ Works with your data
    """)
    
    st.divider()
    
    st.markdown("### 🤖 Active Models")
    if health_data:
        for model in health_data.get('active_models', []):
            st.success(f"✓ {model.replace('_', ' ').title()}")


# ============================================================================
# ORDER ANALYSIS - WITH DYNAMIC FIELDS
# ============================================================================

if analysis_mode == "📦 Order Analysis":
    st.header("📦 Order Quality Analysis")
    st.info("💡 **Flexible Mode**: Add or remove any fields you need!")
    
    # Initialize session state for dynamic fields
    if 'order_fields' not in st.session_state:
        st.session_state.order_fields = {
            'order_id': {'type': 'text', 'value': f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}"},
            'total_price': {'type': 'number', 'value': 150.0},
            'total_freight': {'type': 'number', 'value': 25.0},
            'num_items': {'type': 'int', 'value': 2},
            'primary_category': {'type': 'select', 'value': 'electronics', 
                                'options': ['electronics', 'furniture', 'clothing', 'sports']},
        }
    
    # Dynamic field management
    with st.expander("➕ Add Custom Fields", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_field_name = st.text_input("Field Name", key="new_field_name")
        with col2:
            new_field_type = st.selectbox("Field Type", 
                ["text", "number", "int", "select", "checkbox"], key="new_field_type")
        with col3:
            if st.button("➕ Add Field"):
                if new_field_name and new_field_name not in st.session_state.order_fields:
                    st.session_state.order_fields[new_field_name] = {
                        'type': new_field_type,
                        'value': 0 if new_field_type in ['number', 'int'] else ''
                    }
                    st.rerun()
    
    # Display and collect all fields
    order_data = {}
    
    st.subheader("Order Fields")
    
    # Organize fields into columns
    fields_list = list(st.session_state.order_fields.items())
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, (field_name, field_config) in enumerate(fields_list):
        col_idx = idx % num_cols
        
        with cols[col_idx]:
            field_type = field_config['type']
            field_value = field_config['value']
            
            # Render appropriate input type
            if field_type == 'text':
                value = st.text_input(field_name, value=field_value, key=f"field_{field_name}")
            elif field_type == 'number':
                value = st.number_input(field_name, value=float(field_value), key=f"field_{field_name}")
            elif field_type == 'int':
                value = st.number_input(field_name, value=int(field_value), step=1, key=f"field_{field_name}")
            elif field_type == 'select':
                options = field_config.get('options', ['option1', 'option2'])
                value = st.selectbox(field_name, options, key=f"field_{field_name}")
            elif field_type == 'checkbox':
                value = 1 if st.checkbox(field_name, value=bool(field_value), key=f"field_{field_name}") else 0
            else:
                value = st.text_input(field_name, value=field_value, key=f"field_{field_name}")
            
            order_data[field_name] = value
            
            # Delete button
            if field_name != 'order_id':  # Can't delete order_id
                if st.button("🗑️", key=f"del_{field_name}"):
                    del st.session_state.order_fields[field_name]
                    st.rerun()
    
    if st.button("🔮 Analyze Order", type="primary", use_container_width=True):
        with st.spinner("Running analysis..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/analyze/order",
                    json={"order_data": order_data}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result["analysis"]
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display results (same as before)
                    st.markdown("### 📊 Overall Assessment")
                    display_risk_card(
                        analysis["overall_risk_score"],
                        analysis["overall_assessment"]
                    )
                    
                    if analysis["flags"]:
                        st.markdown("### 🚩 Active Flags")
                        flags_html = "".join([f'<span class="flag-badge">{flag}</span>' for flag in analysis["flags"]])
                        st.markdown(flags_html, unsafe_allow_html=True)
                    
                    # Predictions
                    st.markdown("### 📈 Detailed Predictions")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "quality" in analysis["predictions"]:
                            q = analysis["predictions"]["quality"]
                            if "error" not in q:
                                st.markdown("#### 🎯 Quality Prediction")
                                st.metric("Risk Level", q["risk_level"])
                                st.metric("Risk Score", f"{q['risk_score']:.1%}")
                    
                    with col2:
                        if "delivery" in analysis["predictions"]:
                            d = analysis["predictions"]["delivery"]
                            if "error" not in d:
                                st.markdown("#### 🚚 Delivery Prediction")
                                st.metric("Estimated Days", d["predicted_days"])
                                st.metric("Category", d["category"])
                    
                    if analysis["recommendations"]:
                        st.markdown("### 💡 Recommendations")
                        for i, rec in enumerate(analysis["recommendations"], 1):
                            st.info(f"{i}. {rec}")
                    
                    with st.expander("📄 Full Analysis", expanded=False):
                        st.json(analysis)
                
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


elif analysis_mode == "📸 Image Analysis":
    st.header("📸 Product Image Analysis")
    
    uploaded_image = st.file_uploader("Upload Product Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if uploaded_image and st.button("🔮 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                files = {"image": uploaded_image.getvalue()}
                
                response = requests.post(f"{API_URL}/api/analyze/image", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result["analysis"]
                    
                    st.success("✅ Image Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "image_classification" in analysis and analysis["image_classification"]:
                            ic = analysis["image_classification"]
                            if "error" not in ic:
                                st.markdown("### 🎯 Image Classification")
                                st.metric("Predicted Category", ic['predicted_class'])
                                st.metric("Confidence", f"{ic['confidence']:.1%}")
                    
                    with col2:
                        if "object_detection" in analysis and analysis["object_detection"]:
                            od = analysis["object_detection"]
                            if "error" not in od:
                                st.markdown("### 🔍 Object Detection")
                                st.metric("Objects Detected", od['num_objects'])
                                for obj in od.get("detected_objects", []):
                                    st.markdown(f"- {obj['class']} ({obj['confidence']:.1%})")
                    
                    with st.expander("📄 Full Results"):
                        st.json(result)
                
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


elif analysis_mode == "🔄 Combined Analysis":
    st.header("🔄 Complete Order + Image Analysis")
    
    # Simplified order input
    st.subheader("Order Data (JSON)")
    st.info("💡 Enter your order data as JSON - any fields accepted!")
    
    default_order = {
        "order_id": f"ORDER_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "total_price": 150.00,
        "num_items": 2,
        "primary_category": "electronics"
    }
    
    order_json = st.text_area(
        "Order Data (JSON format)",
        value=json.dumps(default_order, indent=2),
        height=200
    )
    
    # Image upload
    st.subheader("Product Image (Optional)")
    uploaded_image = st.file_uploader("Upload product image", type=['jpg', 'jpeg', 'png'], key="combined_image")
    
    if uploaded_image:
        st.image(Image.open(uploaded_image), caption="Product Image", width=300)
    
    if st.button("🔮 Run Complete Analysis", type="primary", use_container_width=True):
        with st.spinner("Running analysis..."):
            try:
                # Validate JSON
                try:
                    json.loads(order_json)
                except:
                    st.error("Invalid JSON format in order data")
                    st.stop()
                
                # Prepare request
                data = {"order_data": order_json}
                files = {}
                if uploaded_image:
                    files["image"] = uploaded_image.getvalue()
                
                response = requests.post(
                    f"{API_URL}/api/analyze/combined",
                    data=data,
                    files=files if files else None
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result["analysis"]
                    
                    st.success("✅ Complete Analysis Finished!")
                    
                    display_risk_card(
                        analysis["overall_risk_score"],
                        analysis["overall_assessment"]
                    )
                    
                    st.markdown("### 📄 Generated Report")
                    st.text(result.get("report", ""))
                    
                    with st.expander("📊 Full Analysis"):
                        st.json(analysis)
                
                else:
                    st.error(f"Error: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")


elif analysis_mode == "📊 Batch Analysis":
    st.header("📊 Batch Order Analysis")
    st.info("💡 Upload CSV with ANY columns - completely flexible!")
    
    uploaded_csv = st.file_uploader("Upload CSV File", type=['csv'])
    
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.write(f"**Loaded {len(df)} orders with {len(df.columns)} columns**")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("🔮 Analyze Batch", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {len(df)} orders..."):
                try:
                    orders_list = df.to_dict('records')
                    
                    response = requests.post(
                        f"{API_URL}/api/analyze/batch",
                        json={"orders": orders_list}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        summary = result["summary"]
                        
                        st.success("✅ Batch Analysis Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Orders", summary["total_orders"])
                        
                        risk_dist = summary["risk_distribution"]
                        high_risk = risk_dist.get("HIGH_RISK", 0)
                        col2.metric("High Risk Orders", high_risk)
                        
                        flags_count = sum(summary["flags_summary"].values()) if summary["flags_summary"] else 0
                        col3.metric("Total Flags", flags_count)
                        
                        # Charts
                        if risk_dist:
                            st.markdown("### Risk Distribution")
                            fig = px.pie(values=list(risk_dist.values()), names=list(risk_dist.keys()))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download
                        results_df = pd.DataFrame(result["results"])
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    else:
                        st.error(f"Error: {response.text}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")


elif analysis_mode == "🔧 Custom JSON":
    st.header("🔧 Custom JSON Analysis")
    st.info("💡 For advanced users: Send raw JSON with any structure!")
    
    st.markdown("### 📝 Enter Order Data (JSON)")
    
    example_json = {
        "order_data": {
            "order_id": "CUSTOM_001",
            "your_custom_field_1": "value1",
            "your_custom_field_2": 123,
            "your_custom_field_3": ["list", "of", "values"],
            "nested_field": {
                "sub_field": "works too!"
            }
        }
    }
    
    json_input = st.text_area(
        "JSON Input",
        value=json.dumps(example_json, indent=2),
        height=400
    )
    
    if st.button("🔮 Send Request", type="primary", use_container_width=True):
        try:
            # Validate JSON
            data = json.loads(json_input)
            
            response = requests.post(
                f"{API_URL}/api/analyze/order",
                json=data
            )
            
            if response.status_code == 200:
                st.success("✅ Request Successful!")
                st.json(response.json())
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**E-commerce QC System v2.0**")
with col2:
    st.markdown("✨ **Completely Flexible**")
with col3:
    st.markdown(f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.caption("© 2024 E-commerce Quality Control | Flexible API - Add Any Features!")