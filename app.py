
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Dashboard Analisis Penjualan Suplemen", layout="wide")
st.title("Dashboard Analisis Penjualan Suplemen")

@st.cache_data
def load_data():
    df = pd.read_csv("Supplement_Sales_Weekly_Expanded.csv")
    return df

def handle_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

@st.cache_data
def preprocess(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = handle_outliers(df, numeric_cols)

    le_location = LabelEncoder()
    le_category = LabelEncoder()

    df['Location_encoded'] = le_location.fit_transform(df['Location'])
    df['Category_encoded'] = le_category.fit_transform(df['Category'])

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

    return df, df_scaled, le_location, le_category

@st.cache_data
def train_model(df):
    X = df['Category_encoded'].values.reshape(-1, 1)
    y = df['Location_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return model, X_test, y_test, y_pred, y_proba

menu = st.sidebar.selectbox("Menu", ["Data Overview", "Model Evaluation", "Prediction Tool"])

df = load_data()
df_clean, df_scaled, le_location, le_category = preprocess(df)

if menu == "Data Overview":
    st.header("Data Overview")
    st.write("### Sample Data")
    st.dataframe(df.head())

    st.write("### Statistik Deskriptif")
    st.dataframe(df.describe())

    st.write("### Boxplot untuk Outlier")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df.select_dtypes(include=[np.number]), ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write("### Korelasi antar fitur")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

elif menu == "Model Evaluation":
    st.header("Evaluasi Model Naive Bayes")
    model, X_test, y_test, y_pred, y_proba = train_model(df_clean)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Akurasi", f"{acc:.3f}")

    st.write("### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, target_names=le_location.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write("### ROC Curve")
    n_classes = len(model.classes_)
    y_test_bin = label_binarize(y_test, classes=model.classes_)

    fig_roc, ax_roc = plt.subplots()

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f'{le_location.inverse_transform([model.classes_[i]])[0]} (AUC = {roc_auc:.2f})')
    st.pyplot(fig_roc)

    ax_roc.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area=%0.2f)' % roc_auc)
    ax_roc.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) curve')
    ax_roc.legend(loc='lower right')
    st.pyplot(fig_roc)

    st.write("### Elbow Method")
    fig_elbow, ax_elbow= plt.subplots()
    u = df.groupby("Product Name")["Units Sold"].mean().reset_index()
    u.columns = ['Product Name', 'Total Units Sold']

    scaler = StandardScaler()
    X = scaler.fit_transform(u[["Total Units Sold"]])

    inertia = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    ax_elbow.plot(k_range, inertia, marker='o')
    ax_elbow.set_title("Elbow Method for Optimal k")
    ax_elbow.set_xlabel("Number of Clusters")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)

    st.write("### K-Means Clustering")
    X = df.select_dtypes(include='number')
    model = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = model.fit_predict(X)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    fig_KM, ax_KM = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    for cluster in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == cluster]
        ax_KM.scatter(subset['PCA1'], subset['PCA2'],
                    color=colors[cluster],
                    label=f'Cluster {cluster}',
                    s=30, edgecolor='k', linewidth=0.3)

    ax_KM.set_title('K-Means Clustering (K=3) of Product Sales using PCA')
    ax_KM.set_xlabel('PCA Component 1')
    ax_KM.set_ylabel('PCA Component 2')
    ax_KM.grid(True)
    ax_KM.legend()
    fig_KM.tight_layout()
    st.pyplot(fig_KM)

    X = u[["Total Units Sold"]]
    kmeans_final = KMeans(n_clusters=3, random_state=42)
    u['Cluster'] = kmeans_final.fit_predict(X)

    u = u.sort_values(by="Total Units Sold", ascending=False)

    fig_KM, ax_KM = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=u, x="Product Name", y="Total Units Sold", hue="Cluster",
                    palette="viridis", s=100, ax=ax_KM)

    ax_KM.set_title("K-Means Clustering of Products Based on Total Units Sold")
    ax_KM.set_xlabel("Product Name")
    ax_KM.set_ylabel("Total Units Sold")
    ax_KM.tick_params(axis='x', rotation=90)
    fig_KM.tight_layout()
    st.pyplot(fig_KM)

elif menu == "Prediction Tool":
    st.header("Tool Prediksi Lokasi Berdasarkan Kategori Produk")
    category_input = st.selectbox("Pilih Kategori Produk:", df['Category'].unique())

    input_enc = le_category.transform([category_input]).reshape(-1, 1)
    model, _, _, _, _ = train_model(df_clean)

    pred = model.predict(input_enc)[0]
    prob = model.predict_proba(input_enc)[0]

    pred_label = le_location.inverse_transform([pred])[0]

    st.write(f"### Prediksi Lokasi: **{pred_label}**")
    st.write(f"### Probabilitas Tiap Lokasi:")
    prob_df = pd.DataFrame({
        "Location": le_location.inverse_transform(model.classes_),
        "Probability": prob
    })
    st.dataframe(prob_df.sort_values(by="Probability", ascending=False))
