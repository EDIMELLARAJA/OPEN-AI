## Introduction
------------
The MultiDoc Chat App is a Python application that allows you to chat with multiple documents. You can ask questions about the Docs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded Docs.

## The application follows these steps to provide responses to your questions:

1. Doc Loading: The app reads multiple Doc documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the Docs.



# To create an Environment 
"python -m venv venv"- use this command

# To Activate the environment in CMD OR Linux
".\venv\Scripts\activate" -use this command for CMD
"source venv/bin/activate" - use this command for Linux

# To Intialize Git to project 
"git init" - use this command
"git add" - to add git to project

# Create Repository in GitHub
Go to GitHub and log in.
Click the + icon in the top-right corner and select New repository.
Name your repository and choose visibility (public or private).
Do not initialize the repository with a README, .gitignore, or license (you've already initialized it locally).
Click Create repository.

# Connect the Local Repository to GitHub
"git remote add origin https://github.com/<your-username>/<repository-name>.git" - Repalce <your-username>/<repository-name> to connect. 

# To push commands to branches
"git push" - To commit the code and changes

