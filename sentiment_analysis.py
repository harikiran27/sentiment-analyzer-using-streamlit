import tensorflow as tf
import streamlit as st
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import pandas as pd
import plotly.graph_objs as go



loaded_model = TFDistilBertForSequenceClassification.from_pretrained("sentiment")

def analyse_sentiment(test_sentence):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    predict_input = tokenizer.encode(test_sentence,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="tf")

    tf_output = loaded_model.predict(predict_input)[0]


    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    labels = ['negative','positive']
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    print(labels[label[0]])
    return labels[label[0]]


import pymongo
connect_string = 'mongodb+srv://admin:admin@cluster0.n7nsv.mongodb.net/myFirstDatabase?retryWrites=true&w=majority' 

from django.conf import settings
my_client = pymongo.MongoClient(connect_string)

# First define the database name
dbname = my_client['reviewsDB']

# Now get/create collection name (remember that you will see the database in your mongodb cluster only after you create a collection
collection_name = dbname["reviews"]


fig = go.Figure()
st.write("""
# Sentiments Analysis App ✌

""")

st.write('Sentiment analysis is the interpretation and classification of emotions (positive, and neutral) within text data using text analysis techniques. Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.')
st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.header('User Input(s)')
st.sidebar.subheader('Single Review Analysis')
single_review = st.sidebar.text_input('Enter single review below:')
st.sidebar.subheader('Mutiple Reviews Analysis')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
value = st.sidebar.button("Feed Database Reviews")

count_positive = 0
count_negative = 0

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    for i in range(input_df.shape[0]):
        result = analyse_sentiment(str(input_df.iloc[i]))
        if result=='positive':
            count_positive+=1
        elif result=='negative':
            count_negative+=1


    x = ["Positive", "Negative"]
    y = [count_positive, count_negative]

    if count_positive>count_negative:
        st.write("""# Great Work there! Majority of people liked your product 😃""")
    elif count_negative>count_positive:
        st.write("""# Try improving your product! Majority of people didn't find your product upto the mark 😔""")
    else:
        st.write("""# Good Work there, but there's room for improvement! Majority of people have neutral reactions to your product 😶""")
        
    layout = go.Layout(
        title = 'Multiple Reviews Analysis',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Number of reviews'),)
    
    fig.update_layout(dict1 = layout, overwrite = True)
    fig.add_trace(go.Bar(name = 'Multi Reviews', x = x, y = y))
    st.plotly_chart(fig, use_container_width=True)

elif single_review:
    result = analyse_sentiment(single_review)
    data = {
        "text" : single_review,
        "score" : result.lower()
    }
    collection_name.insert(data)

    if result.lower()=='positive':
        st.write("""# Great Work there! You got a Positive Review 😃""")
    elif result.lower()=='negative':
        st.write("""# Try improving your product! You got a Negative Review 😔""")

elif value:
    dbData = collection_name.find()
    for data in dbData:
        string = data["score"]
        
        if string.lower()=='positive':
            count_positive+=1
        elif string.lower()=='negative':
            count_negative+=1 

    x = ["Positive", "Negative"]
    y = [count_positive, count_negative]

    if count_positive>count_negative:
        st.write("""# Great Work there! Majority of people liked your product 😃""")
    elif count_negative>count_positive:
        st.write("""# Try improving your product! Majority of people didn't find your product upto the mark 😔""")
   
    layout = go.Layout(
        title = 'Database Reviews Analysis',
        xaxis = dict(title = 'Category'),
        yaxis = dict(title = 'Number of reviews'),)
    
    fig.update_layout(dict1 = layout, overwrite = True)
    fig.add_trace(go.Bar(name = 'Multi Reviews', x = x, y = y))
    st.plotly_chart(fig, use_container_width=True)
    

else:
    st.write("# ⬅ Enter user input from the sidebar to see the nature of the review.")


