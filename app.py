import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Title of the Streamlit app
st.title('Iris Model Inference')

# Sidebar with detailed instructions for the user
with st.sidebar:
    st.header('Data Requirements')
    st.write(
        'To perform model inference, please upload a CSV file with four numerical features.'
        'Column names are not important, but the order and format of the data are critical.'
    )
    
    # Expandable section to show data format details
    with st.expander('Data format details'):
        st.markdown('- **Encoding**: UTF-8')
        st.markdown('- **Separator**: Comma (",")')
        st.markdown('- **Decimal Delimiter**: Dot (".")')
        st.markdown('- **Header**: First row should contain the header')
    
    # Visual divider for better organization
    st.divider()

    # Footer or signature
    st.caption('Developed by David Palacio')

# Check if the key `clicked`` exists in `st.session_state`. If it doesn't
# exist (for instance, the case when the application it is uploaded the 
# first time), it starts with a dictionary with just one input: `1: False`
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

# When this function is called, change the value associated with the
# key `button`, inside the dictionary `st.session_state.clicked` to `True`
# This function is created to be used as a `callback` when the button is
# pressioned
def clicked(button):
    st.session_state.clicked[button] = True

# Button to start the inference process
st.button("Let's get started", on_click = clicked, args = [1])

# Logic for uploading and processing the CSV file once the button is clicked
if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader('Choose a CSV file', type = 'csv')

    if uploaded_file is not None:
        # Load the uploaded CSV into a DataFrame
        df = pd.read_csv(uploaded_file, low_memory=True)

        # Display a sample of the uploaded data
        st.header('Uploaded data sample')
        st.write(df.head())

        # Load the pre-trained model
        model = joblib.load('model/model.joblib')

        # Perform prediction using the model
        predictions = model.predict_proba(df)

        # Convert predictions into a DataFrame
        predictions_df = pd.DataFrame(predictions, columns=[
            'setosa_probability', 
            'versicolour_probability', 
            'virginica_probability'
        ])


        # Display a sample of the predictions
        st.header('Predicted value')
        st.write(predictions_df.head())

        # Prepare predictions for download as a CSV file
        csv_data = predictions_df.to_csv(index=False).encode('utf-8')

        # Button to download the predictions
        st.download_button(
            label='Download predictions',
            data=csv_data,
            file_name='predictions.csv',
            mime='text/csv',
            key='download-csv'
        )