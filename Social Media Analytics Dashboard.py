import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


# Define User Data File
USER_DATA_FILE = "users(ass)1.csv"

# Ensure user data file exists with correct headers
def load_users():
    if os.path.exists(USER_DATA_FILE):
        users = pd.read_csv(USER_DATA_FILE, dtype=str)  # Read data as strings
        if "username" not in users.columns or "password" not in users.columns:
            users = pd.DataFrame(columns=["username", "password"])
            users.to_csv(USER_DATA_FILE, index=False)
    else:
        users = pd.DataFrame(columns=["username", "password"])
        users.to_csv(USER_DATA_FILE, index=False)

    return users


# Save new user
def save_user(username, password):
    users = load_users()
    new_user = pd.DataFrame({"username": [username.strip()], "password": [password.strip()]})
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DATA_FILE, index=False)


# Authenticate user
def authenticate(username, password):
    users = load_users()

    # Strip spaces to avoid hidden whitespace issues
    users["username"] = users["username"].str.strip()
    users["password"] = users["password"].str.strip()

    return ((users["username"] == username.strip()) & (users["password"] == password.strip())).any()


# Sample Social Media Data
data = [
    {"Platform": "Facebook", "Users": 2900, "Engagement": 85, "Growth": 5},
    {"Platform": "Instagram", "Users": 1500, "Engagement": 92, "Growth": 12},
    {"Platform": "Twitter", "Users": 330, "Engagement": 70, "Growth": 3},
    {"Platform": "LinkedIn", "Users": 310, "Engagement": 60, "Growth": 4},
    {"Platform": "Snapchat", "Users": 500, "Engagement": 88, "Growth": 8},
    {"Platform": "TikTok", "Users": 1200, "Engagement": 95, "Growth": 25},
    {"Platform": "YouTube", "Users": 2300, "Engagement": 90, "Growth": 10},
    {"Platform": "Pinterest", "Users": 450, "Engagement": 65, "Growth": 6},
    {"Platform": "Reddit", "Users": 430, "Engagement": 72, "Growth": 4},
    {"Platform": "WhatsApp", "Users": 2300, "Engagement": 91, "Growth": 9},
]

df = pd.DataFrame(data)

# Session Management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


# Signup Page
def signup():
    st.title("Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    users = load_users()

    if st.button("Register"):
        if username.strip() in users["username"].values:
            st.error("Username already exists! Try another one.")
        else:
            save_user(username, password)
            st.success("Account created successfully! Please login.")


# Login Page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid username or password!")


# Dashboard
def dashboard():
    st.sidebar.title(f"Welcome, {st.session_state.username} 👋")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""

    st.title("📊 Social Media Analytics Dashboard")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Search Platform", "📈 Engagement Comparison", "📉 Growth Trend", "🎯 Data Insights", "📤 Export Data","📉 Prediction"
    ])

    # Tab 1: Search Platform
    with tab1:
        st.subheader("Search by Platform")
        search_query = st.text_input("🔍 Enter platform name:")
        filtered_df = df[df["Platform"].str.contains(search_query, case=False, na=False)]
        st.dataframe(filtered_df)

    # Tab 2: User Engagement Comparison (Bar Chart)
    with tab2:
        st.subheader("User Engagement Comparison")
        fig1 = px.bar(df, x="Platform", y="Engagement", title="User Engagement Across Platforms", color="Platform")
        st.plotly_chart(fig1)

    # Tab 3: Platform Growth Trend (Line Chart)
    with tab3:
        st.subheader("Platform Growth Trend")
        fig2 = px.line(df, x="Platform", y="Growth", title="Growth Rate of Social Media Platforms", markers=True)
        st.plotly_chart(fig2)

    # Tab 4: Data Insights (Scatter & Pie Chart)
    with tab4:
        col1, col2 = st.columns(2)

        # Scatter Plot: Users vs. Engagement
        with col1:
            st.subheader("Users vs. Engagement (Scatter)")
            fig3 = px.scatter(df, x="Users", y="Engagement", color="Platform", size="Growth",
                              title="User Engagement vs Number of Users")
            st.plotly_chart(fig3)

        # Pie Chart: User Distribution Across Platforms
        with col2:
            st.subheader("User Distribution Across Platforms")
            fig4 = px.pie(df, names="Platform", values="Users", title="User Distribution on Different Platforms")
            st.plotly_chart(fig4)

        # Platform Comparison with Selectbox
        st.subheader("Compare Platform Engagement & Growth")
        platform1 = st.selectbox("Select First Platform", df["Platform"])
        platform2 = st.selectbox("Select Second Platform", df["Platform"])

        if platform1 and platform2 and platform1 != platform2:
            comparison_df = df[df["Platform"].isin([platform1, platform2])]
            fig5 = px.bar(comparison_df, x="Platform", y=["Engagement", "Growth"], barmode="group",
                          title=f"Comparison Between {platform1} & {platform2}")
            st.plotly_chart(fig5)

    # Tab 5: Export Data
    with tab5:
        st.subheader("Download Data")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="social_media_data.csv", mime="text/csv")

    with tab6:
        # Train Model
        reg = LinearRegression()
        reg.fit(df[['Engagement', 'Growth']], df[['Users']])

        # User Input for Prediction
        st.write("### Predict Users Based on Engagement & Growth")
        engagement_input = st.number_input("Enter Engagement (%)", min_value=0, max_value=100, value=80, step=1)
        growth_input = st.number_input("Enter Growth Rate (%)", min_value=0, max_value=50, value=5, step=1)

        if st.button("Predict Users"):
            predicted_users = reg.predict([[engagement_input, growth_input]])[0][0]
            st.success(f"Predicted Users: {predicted_users:,.0f}")

        # Display Model Coefficients
        st.write("### Model Details:")
        st.write(f"**Coefficients:** {reg.coef_[0]}")
        st.write(f"**Intercept:** {reg.intercept_[0]:.2f}")

        # Create Plotly Figure
        fig = go.Figure()

        # Add Scatter Plot for Data Points
        fig.add_trace(go.Scatter(x=df.Engagement, y=df.Users, mode='markers',
                                 marker=dict(color='red', size=8),
                                 name='Data Points'))

        # Add Regression Line
        predicted_users_line = reg.predict(df[['Engagement', 'Growth']]).flatten()
        fig.add_trace(go.Scatter(x=df.Engagement, y=predicted_users_line,
                                 mode='lines', line=dict(color='blue'),
                                 name='Prediction Line'))

        # Layout
        fig.update_layout(title='📈 Engagement vs Users',
                          xaxis_title='Engagement (%)',
                          yaxis_title='Users (millions)',
                          legend_title='Legend')

        # Show Plotly Figure in Streamlit
        st.plotly_chart(fig)


# Main App Logic
if not st.session_state.logged_in:
    choice = st.sidebar.selectbox("Select Option", ["Login", "Signup"])
    if choice == "Login":
        login()
    else:
        signup()
else:
    dashboard()

