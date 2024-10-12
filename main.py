import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the diabetes dataset
@st.cache_data
def load_data():
    diabetes_data = pd.read_csv("C:\\Users\\canan\\anaconda3\\diabetes.csv")
    return diabetes_data

# Function to train and evaluate the k-NN model
def train_model(data):
    # Split the dataset into features (X) and target variable (y)
    X = data.drop(columns=['Outcome'])  # Features
    y = data['Outcome']  # Target variable

    # Splitting the dataset into the training set and test set
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize the k-NN classifier
    k = 5  # Number of neighbors to consider
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Train the classifier
    knn_classifier.fit(X_train_scaled, y_train)

    return knn_classifier, scaler

# Predict diabetes for new data
def predict_diabetes(model, scaler, data):
    # Scale the input features
    data_scaled = scaler.transform(data)

    # Predict using the trained model
    prediction = model.predict(data_scaled)

    return prediction

# Visualize the dataset using a bar chart
def visualize_bar_chart(data, column):
    st.write(f"Bar Chart of {column}:")
    column_counts = data[column].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=column_counts.index, y=column_counts.values, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Visualize the dataset using a scatter plot
def visualize_scatter_plot(data, x_column, y_column):
    st.write(f"Scatter Plot of {x_column} vs {y_column}:")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_column, y=y_column, hue='Outcome', data=data, ax=ax)
    st.pyplot(fig)

# Visualize the dataset using a box plot
def visualize_box_plot(data):
    st.write("Box Plot of Dataset Features:")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# Visualize the correlation heatmap
def visualize_correlation_heatmap(data):
    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    # Title of the web app
    st.title("DIABETES PREDICTION")

    # Load the data
    diabetes_data = load_data()

    # Sidebar options
    st.sidebar.title("NAVIGATE")

    option = st.sidebar.radio("Select Option", ["Home", "View Dataset", "Visualization", "Prediction"])

    if option == "Home":
        st.write("Welcome to the Diabetes Risk Prediction Dashboard!")
        st.write("Explore health metrics, visualize data, and predict diabetes risk with machine learning.")
        st.write("Our goal is to empower users to understand and manage their risk effectively.")
        
        # Add image
        st.image("C:\\Users\\isama\\OneDrive\\Desktop\\ml_main\\diabetes img.jpeg")
        
    elif option == "View Dataset":
        st.write("### Descriptive Statistics")
        st.write(diabetes_data.describe())
        st.write("Viewing Diabetes Dataset:")
        st.write(diabetes_data.head())

    elif option == "Visualization":
        visualization_option = st.sidebar.selectbox("Choose Visualization", ["Bar Chart", "Scatter Plot", "Box Plot", "Correlation Heatmap"])

        if visualization_option == "Bar Chart":
            # Choose column for bar chart
            bar_chart_column = st.sidebar.selectbox("Choose Column for Bar Chart", diabetes_data.columns)
            visualize_bar_chart(diabetes_data, bar_chart_column)
        elif visualization_option == "Scatter Plot":
            # Choose columns for scatter plot
            scatterplot_x_column = st.sidebar.selectbox("Choose X Column for Scatter Plot", diabetes_data.columns)
            scatterplot_y_column = st.sidebar.selectbox("Choose Y Column for Scatter Plot", diabetes_data.columns)
            visualize_scatter_plot(diabetes_data, scatterplot_x_column, scatterplot_y_column)
        elif visualization_option == "Box Plot":
            visualize_box_plot(diabetes_data)
        elif visualization_option == "Correlation Heatmap":
            visualize_correlation_heatmap(diabetes_data)

    elif option == "Prediction":
        st.write("## Predict Diabetes")
        st.write("Fill in the following details to predict diabetes.")

        # Input features for prediction
        st.write("### Input Features")
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
            glucose = st.number_input("Glucose", min_value=0, max_value=199, value=0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=0)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=0)
        with col2:
            insulin = st.number_input("Insulin", min_value=0, max_value=846, value=0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=0.0)
            diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.42, value=0.0)
            age = st.number_input("Age", min_value=21, max_value=81, value=21)

        input_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree_function],
            "Age": [age]
        })

        if st.button("Predict"):
            model, scaler = train_model(diabetes_data)
            prediction = predict_diabetes(model, scaler, input_data)
            if prediction[0] == 1:
                st.write("The person is diabetic.")
            else:
                st.write("The person is not diabetic.")

# Run the main function
if __name__ == "__main__":
    main()
