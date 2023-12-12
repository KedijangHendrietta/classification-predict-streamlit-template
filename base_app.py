"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
from PIL import Image
import string 

# Model dependencies
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC

# Vectorizer
news_vect = open("models/vector.pickle","rb")
# loading your vectorizer from the pkl file
tweet = joblib.load(news_vect) 

# Load your raw data
raw = pd.read_csv("data/train.csv")
	
# The main function 
def main():
	"""Tweet Classifier App with Streamlit """
	st.sidebar.image('images//EnviroTech.png')
	st.sidebar.markdown('EnviroTech is a consultancy firm that uses data science processes to solve your everyday marketing problems')
	st.sidebar.markdown('    ')

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "Project Summary", "Explore The Data", "Prediction", "Additional Information","Team",  "Contact Us"]
	st.sidebar.subheader("Navigation")
	selection = st.sidebar.selectbox("Choose an option", options)

	# Building our the "Home" page
	if selection == "Home":
		st.title("Climate change tweet classification")
		st.title("Welcome!")
		header_image = Image.open('images/global-disaster.webp')
		st.image(header_image, use_column_width=True)
		st.subheader("  ")

	# Building the "Information" page
	if selection == "Additional Information":
		st.subheader("Additional Information")
		st.subheader("Model description")
		st.markdown("* **Linear Support Vector Classifier**:The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide, returning a best fit hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the predicted class is. ")
		st.markdown("* **Bernoulli Naive Bayers Classifier**: Bernoulli Naive Bayes is a variant of the Naive Bayes algorithm used for discrete data where features are only in binary form.")
		st.markdown("* **K-Neighbours Classifier**: KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood. ")
		st.markdown('For more indepth information on these models, visit: https://monkeylearn.com/blog/classification-algorithms/')


	# Building the "EDA" page
	if selection == "Explore The Data":
		st.title("Climate change tweet classification")
		st.subheader("**Exploratory data analysis visuals**")
		st.markdown('    ')
		st.markdown("**The distribution of the four possible sentiments**")
		st.markdown("This visual representation illustrates how tweets are categorized across four sentiment categories—pro, anti, neutral, and news-related—related to climate change. It provides an overview of the sentiments expressed in the dataset, showcasing the distribution or frequency of each sentiment category. This helps in understanding the prevalence of different viewpoints or attitudes towards climate change among the tweets analyzed.")
		st.image('images//Sentiment.png')
		st.markdown(" ")
		st.markdown("   ")
		st.markdown("**The top 10 most porpular Hashtags.**")
		st.markdown("This visual display highlights the top 10 hashtags used in the dataset related to climate change discussions. Hashtags play a significant role in social media by organizing and categorizing content. This visualization identifies the most frequently used hashtags, providing insight into trending topics, key themes, or subjects of interest within the realm of climate change discourse. It allows users to grasp the most prevalent topics or conversations occurring around this subject based on hashtag usage.")
		st.image('images//hashtags.png')
		st.markdown("   ")
		st.markdown("**Top 10 most common words in the Pro sentiment**")
		st.markdown("This analysis aims to identify and display the words that appear most often in the 'Pro' sentiment text messages. By counting the occurrences of each word and selecting the top ten based on their frequency, this visualization provides a quick insight into the most prevalent terms used in messages expressing a positive or supportive sentiment towards a particular topic, such as climate change.")
		st.image('images/Top 10 most common words.png')
		st.markdown("  ")

	# Building our the predication page
	if selection == "Prediction":
		st.title("Climate change tweet classification")
		st.info("Prediction with ML Models")
		# Understanding sentiment predictions
		st.image(Image.open('images/Matrix.png'), caption=None, use_column_width=True)

		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		
		model_choice = st.radio("Choose a model", ("LinearSVC","BNB"))   
	
		if model_choice == 'LinearSVC':
			vect_text = tweet.transform([tweet_text]).toarray()
			#load pkl file with model and make predictions
			predictor = joblib.load(open(os.path.join("models/LinearSVC.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			#when model has ran succefully, it will print out predictions
			if prediction[0] == -1:
				st.success('Text has been classified to show non believe in man made climate change')
			elif prediction[0] == 0:
				st.success('Text has been classified to being  neither belief nor non belief in man made climate change')
			elif prediction[0] == 1:
				st.success('Text has been classified to show belief in man made climate change')
			else:
				st.success('Text has ben classified as factual/news about climate change')
			st.success("Text Categorized as: {}".format(prediction))

		if model_choice == 'BNB':
			vect_text = tweet.transform([tweet_text]).toarray()
			#load pkl file with model and make predictions
			predictor = joblib.load(open(os.path.join("models/BNB.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			#when model has ran succefully, it will print out predictions
			if prediction[0] == -1:
				st.success('Text has been classified to show non believe in man made climate change')
			elif prediction[0] == 0:
				st.success('Text has been classified to being belief nor non belief in man made climate change')
			elif prediction[0] == 1:
				st.success('Text has been classified to show belief in man made climate change')
			else:
				st.success('Text has ben classified as factual/news about climate change')
			st.success("Text Categorized as:{}".format(prediction))
	
	# Building the "Contact Us" Page
	if selection == "Contact Us":
		st.image('images//EnviroTech.png')
		st.markdown('    ')
		st.markdown('Contact Us')
		st.markdown(' * Tel: 012 439 0000')
		st.markdown(' * Twitter: @EnviroTech')
		st.markdown(' * Address : ')
		st.markdown('11 Adriana Cres, Rooihuiskraal, Centurion, 0154')
 
	# Building the "Summary" page
	if selection == "Team":
		st.subheader("Meet The Team")
		st.markdown(" * **Kedijang Setsome** : CEO  ")
		st.image("images//kedi.webp")
		st.markdown("  ")
		st.markdown(" * **Desiree Malebana** : Project Manager ")
		st.image('images//Desiree.webp')
		st.markdown("  ")
		st.markdown(" * **Mashoto Kgasago** : Developer  ")
		st.image('images//mashoto.webp')
		st.markdown("  ")
		st.markdown(" * **Ninamukhovhe Tshivase** : Developer ")
		st.image('images//nina.jpeg')
		st.markdown("  ")
		st.markdown(" * **Destiny Owobu** : Data Scientist ")
		st.image('images//Destiny.webp')
		
  
	if selection == "Project Summary":
		st.title("Climate Change Belief Classifier")
		st.subheader("Project Overview")
		st.image('images//global-climate-change_500x468.webp')
		st.markdown("Our client  **Appian Way**  is a film production company that produced the film **Before The Floods**. Before the Flood is a documentary film featuring Leonardo DiCaprio as the narrator and advocate. It explores the pressing issue of climate change, examining its impacts on the environment, wildlife, and human civilization. DiCaprio travels across the globe, meeting with scientists, activists, and world leaders to discuss the causes and consequences of climate change, as well as possible solutions to mitigate its effects. The film aims to raise awareness about the urgent need for global action to address climate change before irreparable damage occurs to our planet.")
		st.markdown("EnviroTech Solutions was tasked to develop an app that will enable Appian Way Productions to identify their target market from a customer tweet database for a new documentary release. The assumptions made were that the poteintial viewers are pro climate change and that their tweets revealed this sentiment.")

		

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
