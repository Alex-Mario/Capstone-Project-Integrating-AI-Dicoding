from flask import Flask, render_template, request, flash, redirect, make_response, session
from werkzeug.utils import secure_filename
import tensorflow as tf
import openai
import time
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, TFAutoModelForSequenceClassification
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langdetect import detect
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from openai import OpenAI
import os

api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
openai_api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
api_key_pdf = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

secret_key = os.urandom(24)
app = Flask(__name__)
app.secret_key = secret_key

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
client_pdf = OpenAI(api_key=api_key_pdf)

# Summarization model
summarization_tokenizer = AutoTokenizer.from_pretrained('tokenizer_summarize')
summarization_model = TFAutoModelForSeq2SeqLM.from_pretrained("model_summarize")

# Typo classifier model
typo_classifier_tokenizer = AutoTokenizer.from_pretrained("tokenizer_mispel")
typo_classifier_model = TFAutoModelForSequenceClassification.from_pretrained("model_mispel")

@app.route('/summarize', methods=['GET', 'POST'])
def summarize():
    if request.method == 'POST':
        document = request.form['document']
        inputs = summarization_tokenizer(document, return_tensors="tf")

        word_count = len(document.split())

        if word_count <= 256:
            max_length = 256
            min_length = 0
        else:
            max_length = 1024
            min_length = 512

        outputs = summarization_model.generate(inputs.input_ids, max_length=max_length, min_length=min_length)
        summary = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return render_template('summarize.html', document=document, summary=summary)
    return render_template('summarize.html', document='', summary='')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Typo Classifier
        inputs = typo_classifier_tokenizer(user_input, truncation=True, max_length=512, padding="max_length", return_tensors="tf")
        inputs_dict = {key: inputs[key].numpy() for key in inputs}
        logits = typo_classifier_model.predict(inputs_dict)
        probs = tf.nn.softmax(logits.logits, axis=-1)
        label_indices = tf.argmax(probs, axis=-1).numpy()
        id2label = {0: 'Correct', 1: 'Misspelled'}
        predicted_label = id2label[label_indices[0]]

        return render_template('classify.html', user_input=user_input, predicted_label=predicted_label)
    return render_template('classify.html', user_input='', predicted_label='')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    # Kode untuk recommendation (sama seperti sebelumnya)
    response = ''
    if request.method == 'POST':
        prompt = request.form['prompt']


        client = openai.OpenAI(api_key=api_key)

        assistant = client.beta.assistants.create(
            name="Dico Reference",
            instructions="Provide minimum 10 learning material references based on specified prompts, language output based on the input language",
            model="gpt-3.5-turbo-1106"
        )

        thread = client.beta.threads.create()

        # User's message
        user_message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Provide minimum 10 learning material references based on specified prompts, language output based on the input language"
        )

        while True:
            time.sleep(5)

            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                )

                # Append user's message with role and newline
                for msg in messages.data:
                    if msg.role == "user":
                        role = msg.role
                        content = msg.content[0].text.value
                        response += f"{role.capitalize()}:\n{content}\n"

                # Then add a newline for spacing
                response += "\n"

                # Append assistant's message with role, assistant name, and newline
                for msg in messages.data:
                    if msg.role == "assistant":
                        role = msg.role
                        content = msg.content[0].text.value
                        response += f"{assistant.name} ({role.capitalize()}):\n{content}\n"

                break

            else:
                time.sleep(5)

    return render_template('recommendation.html', response=response)

@app.route('/content_generation', methods=['GET', 'POST'])
def content_generation():
    # Kode untuk recommendation (sama seperti sebelumnya)
    response = ''
    if request.method == 'POST':
        prompt = request.form['prompt']

        client = openai.OpenAI(api_key=api_key)

        assistant = client.beta.assistants.create(
            name="Dico Generate",
            instructions="Generate minimum 500 words text-based Bahasa Indonesia educational content based on specified prompts and data sources.",
            model="gpt-3.5-turbo-1106"
        )

        thread = client.beta.threads.create()

        # User's message
        user_message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Generate minimum 500 words  text-based Bahasa Indonesia educational content based on specified prompts and data sources."
        )

        while True:
            time.sleep(5)

            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            if run_status.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                )

                # Append user's message with role and newline
                for msg in messages.data:
                    if msg.role == "user":
                        role = msg.role
                        content = msg.content[0].text.value
                        response += f"{role.capitalize()}:\n{content}\n"

                # Then add a newline for spacing
                response += "\n"

                # Append assistant's message with role, assistant name, and newline
                for msg in messages.data:
                    if msg.role == "assistant":
                        role = msg.role
                        content = msg.content[0].text.value
                        response += f"{assistant.name} ({role.capitalize()}):\n{content}\n"

                break

            else:
                time.sleep(5)

    return render_template('content_generation.html', response=response)

#Compare Code

# to get documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_document:
            for page in pdf_document.pages:  # Iterate through each page
                text += page.extract_text()
    return text

# to process our documents
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# embeddings the documents
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# conversation based on documents
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# For handle user input
def handle_userinput(user_question, conversation):
    if conversation is None:
        return "Please upload documents and click the 'Process' button first."

    response = conversation({'question': user_question})
    chat_history = response['chat_history']
    result = []

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            result.append(f"{message.content}")
        else:
            result.append(f"{message.content}")

    return result

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    load_dotenv()

    if request.method == 'POST':
        pdf_docs = request.files.getlist('pdf_docs')
        user_question = request.form['user_question']

        if not pdf_docs:
            return "Please upload your document first."
        else:
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
            conversation = get_conversation_chain(vectorstore)

            result = handle_userinput(user_question, conversation)

            return render_template('compare.html', result=result, success_message="Documents processing successfully")

    return render_template('compare.html', result=None, success_message=None)

# PDF Summarize Code

# Define the allowed extension
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load and process PDF
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

# Summarize text
def summarize_text(texts):
    llm = ChatOpenAI(openai_api_key=api_key_pdf, temperature=0, model_name='gpt-3.5-turbo')
    chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True)
    output_summary = chain.run(texts)
    return output_summary

# Function to translate text to Indonesian
def translate_to_indonesian(text):
    response = client_pdf.completions.create(
        model="text-davinci-003",
        prompt="Translate the following English text to Indonesian:\n\n" + text,
        temperature=0.4,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Function to detect language
def detect_language(texts):
    combined_text = " ".join([text.page_content for text in texts[:5]])  # Check the first 5 pages
    try:
        return detect(combined_text)
    except:
        return None

@app.route('/pdf_summarize', methods=['GET', 'POST'])
def pdf_summarize():

    if request.method == 'POST':
        # Check if the post request has the file part
        file = request.files.get('pdf_file')
        if not file or file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PDF file.')  # Menampilkan pesan kesalahan jika file bukan PDF
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process and summarize the PDF
            chunks = process_pdf(file_path)
            summary = summarize_text(chunks)
            language = detect_language(chunks)

            if language == 'id':
                final_summary = translate_to_indonesian(summary)
            else:
                final_summary = summary

            # Store the summary in session
            session['final_summary'] = final_summary

            # Hapus file setelah selesai
            os.remove(file_path)

            return redirect(request.url)  # Redirect to GET after POST to prevent resubmission

    # GET request or initial page load
    final_summary = session.pop('final_summary', None)  # Retrieve and remove summary from session
    response = make_response(render_template('pdf_summarize.html', summary=final_summary))
    # Set headers to disable caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
