import streamlit as st
from streamlit.logger import get_logger
from predict import predict

# streamlit logger
LOGGER = get_logger(__name__)

def run():
    # Set the main dashboard page browser tab title and icon
    st.set_page_config(
        page_title="Linear Regression Predictor",
        page_icon="📈",
    )

    # Build the sidebar first
    with st.sidebar:
        st.info("Enter X value for prediction")

        # Input for X value
        x_value = st.number_input(
            "Enter X value:",
            value=0.0,
            step=1.0,
            help="Enter any numeric value for X to get Y prediction"
        )

        # Predict button
        predict_button = st.button('Predict Y')

    # Dashboard body
    # Heading for the dashboard
    st.write("# Linear Regression Predictor! 📈")
    st.write("### Y = aX + b")

    # If predict button is pressed
    if predict_button:
        try:
            # Get prediction using the predict function
            y_prediction = predict(x_value)

            # Display result
            st.success(
                f"**Input:** X = {x_value}\n\n"
                f"**Prediction:** Y = {y_prediction:.4f}"
            )

            # Show the linear equation form
            st.info(f"Using the trained linear regression model: Y = aX + b")

        except Exception as e:
            # Error handling
            st.error(f"Error making prediction: {str(e)}")
            LOGGER.error(f"Prediction error: {e}")

    # Display some information about the model
    st.write("---")
    st.write("### About the Model")
    st.write("This dashboard uses a simple linear regression model trained on data following the pattern Y = 2X + 3 + noise.")
    st.write("Enter an X value in the sidebar to see the model's prediction for Y.")

if __name__ == "__main__":
    run()