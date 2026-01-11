# import streamlit as st
# import pandas as pd
# import joblib

# st.title("Customer Data Entry")

# # Input fields
# age = st.number_input("Age", min_value=0, max_value=120, value=49)
# gender = st.radio("Gender", ["Male", "Female"], index=1)
# annual_income = st.number_input("Annual Income", min_value=0, value=44088)
# previous_purchases = st.number_input("Previous Purchases", min_value=0, value=35)
# spending_score = st.number_input("Spending Score", min_value=0, max_value=100, value=35)
# engagement_level = st.radio("Engagement Level", ["Low", "Medium", "High"], index=0)
# offering_type = st.radio("Offering Type", ["Product", "Service"], index=0)
# offering_category = st.radio(
#     "Offering Category",
#     ["Subscription", "Consulting", "Electronics", "Food", "Streaming"],
#     index=0
# )
# discount_used = st.radio("Discount Used", ["Yes", "No"], index=1)
# promo_used = st.radio("Promo Used", ["Yes", "No"], index=1)
# visit_frequency = st.radio("Visit Frequency", ["Low", "Medium", "High"], index=2)

# # Collect into dictionary
# customer_data = {
#     "Age": age,
#     "Gender": gender,
#     "Annual_Income": annual_income,
#     "Previous_Purchases": previous_purchases,
#     "Spending_Score": spending_score,
#     "Engagement_Level": engagement_level,
#     "Offering_Type": offering_type,
#     "Offering_Category": offering_category,
#     "Discount_Used": discount_used,
#     "Promo_Used": promo_used,
#     "Visit_Frequency": visit_frequency
# }

# # Convert to DataFrame for nice display
# df = pd.DataFrame([customer_data])

# st.write("### Customer Data Table")
# st.dataframe(df)

# # Binary Encoding

# df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
# df['Offering_Type'] = df['Offering_Type'].map({'Service':1,'Product':0})
# df['Discount_Used'] = df['Discount_Used'].map({'Yes':1, 'No':0})
# df['Promo_Used'] = df['Promo_Used'].map({'Yes':1, 'No':0})

# # One HOt Encoding
# st.write(df['Offering_Category'].unique())
# # Manual One-Hot Encoding for Offering_Category (drop_first=True)

# df['Offering_Category_Consulting'] = 1 if df.loc[0, 'Offering_Category'] == 'Consulting' else 0
# df['Offering_Category_Electronics'] = 1 if df.loc[0, 'Offering_Category'] == 'Electronics' else 0
# df['Offering_Category_Food'] = 1 if df.loc[0, 'Offering_Category'] == 'Food' else 0
# df['Offering_Category_Streaming'] = 1 if df.loc[0, 'Offering_Category'] == 'Streaming' else 0
# df['Offering_Category_Subscription'] = 1 if df.loc[0, 'Offering_Category'] == 'Subscription' else 0

# # Drop original column
# df.drop('Offering_Category', axis=1, inplace=True)

# #Ordinal Encoding

# size_map = {'Low': 1, 'Medium': 2, 'High': 3}

# df['Visit_Frequency'] = df['Visit_Frequency'].map(size_map)
# df['Engagement_Level'] = df['Engagement_Level'].map(size_map)

# #Feature Scaling


# X = df.drop("Offering_Type", axis=1)
# y = df["Offering_Type"]

# st.write(X)
# #Scaling Using

# scalar = joblib.load(r'models\scaler.pkl')
# x_scaled = scalar.transform(X)
# st.write(x_scaled)

# # Final Output
# model = joblib.load(r'models\decision_tree_model.pkl')
# if model.predict(x_scaled) ==0:
#   st.write("Customer will purchase a Product")
# else:
#   st.write("Customer will purchase a Service")



import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Customer Purchase Prediction")
st.caption("Enter customer details to predict whether they will purchase a Product or a Service")

st.divider()

# -------------------- INPUT SECTION --------------------
st.subheader("📋 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=49)
    annual_income = st.number_input("Annual Income", min_value=0, value=44088)
    spending_score = st.slider("Spending Score", 0, 100, 35)
    engagement_level = st.radio("Engagement Level", ["Low", "Medium", "High"])

with col2:
    gender = st.radio("Gender", ["Male", "Female"])
    previous_purchases = st.number_input("Previous Purchases", min_value=0, value=35)
    visit_frequency = st.radio("Visit Frequency", ["Low", "Medium", "High"])
    offering_type = st.radio("Offering Type", ["Product", "Service"])

st.subheader("🎯 Marketing & Offer Details")

offering_category = st.selectbox(
    "Offering Category",
    ["Subscription", "Consulting", "Electronics", "Food", "Streaming"]
)

col3, col4 = st.columns(2)

with col3:
    discount_used = st.radio("Discount Used", ["Yes", "No"])

with col4:
    promo_used = st.radio("Promo Used", ["Yes", "No"])

st.divider()

# -------------------- DATA PREPARATION --------------------
customer_data = {
    "Age": age,
    "Gender": gender,
    "Annual_Income": annual_income,
    "Previous_Purchases": previous_purchases,
    "Spending_Score": spending_score,
    "Engagement_Level": engagement_level,
    "Offering_Type": offering_type,
    "Offering_Category": offering_category,
    "Discount_Used": discount_used,
    "Promo_Used": promo_used,
    "Visit_Frequency": visit_frequency
}

df = pd.DataFrame([customer_data])

# Binary Encoding
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Offering_Type'] = df['Offering_Type'].map({'Service': 1, 'Product': 0})
df['Discount_Used'] = df['Discount_Used'].map({'Yes': 1, 'No': 0})
df['Promo_Used'] = df['Promo_Used'].map({'Yes': 1, 'No': 0})

# Manual One-Hot Encoding (drop_first=True)
df['Offering_Category_Consulting'] = 1 if offering_category == 'Consulting' else 0
df['Offering_Category_Electronics'] = 1 if offering_category == 'Electronics' else 0
df['Offering_Category_Food'] = 1 if offering_category == 'Food' else 0
df['Offering_Category_Streaming'] = 1 if offering_category == 'Streaming' else 0
df['Offering_Category_Subscription'] = 1 if df.loc[0, 'Offering_Category'] == 'Subscription' else 0

df.drop('Offering_Category', axis=1, inplace=True)

# Ordinal Encoding
size_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['Visit_Frequency'] = df['Visit_Frequency'].map(size_map)
df['Engagement_Level'] = df['Engagement_Level'].map(size_map)

# Features & Target
X = df.drop("Offering_Type", axis=1)

# -------------------- PREDICTION --------------------
st.divider()

if st.button("🔍 Predict Purchase", use_container_width=True):

    scaler = joblib.load(r'models\scaler.pkl')
    model = joblib.load(r'models\decision_tree_model.pkl')

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.success("✅ Customer is likely to purchase a **Product**")
    else:
        st.info("🔵 Customer is likely to purchase a **Service**")

    with st.expander("🔎 View Processed Input Data"):
        st.dataframe(df)


