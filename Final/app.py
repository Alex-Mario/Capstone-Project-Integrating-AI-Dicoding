from flask import Flask, render_template, request, flash, redirect, make_response, session
from werkzeug.utils import secure_filename
import tensorflow as tf
import openai
import time
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, TFAutoModelForSequenceClassification
import pdfplumber
# from dotenv import load_dotenv
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

api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxx'
openai_api_key = 'sk-xxxxxxxxxxxxxxxxxxxxxxxx'
api_key_pdf = 'sk-xxxxxxxxxxxxxxxxxxxxxxxx'

secret_key = os.urandom(24)
app = Flask(__name__)
app.secret_key = secret_key

# Define the UPLOAD_FOLDER variable outside the app.route so that it is accessible globally
upload_folder = os.path.join('/tmp', 'uploads')

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['UPLOAD_FOLDER'] = upload_folder

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
            instructions="Provide minimum 10 learning material references based on specified prompts",
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
            instructions="Provide minimum 10 learning material references based on specified prompts"
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

#compare code

def extract_text_from_pdf(pdf_stream):
    with pdfplumber.open(pdf_stream) as pdf:
        text = ""
        for page in pdf.pages:
            # Reduce the token limit on each PDF page
            text += page.extract_text()[:1000]  # Example: Token limit 1000
    return text

def translate_to_indonesian2(text):
    response = openai.completions.create(
        model="text-davinci-003",
        prompt="Translate the following English text to Indonesian:\n\n" + text,
        temperature=0.4,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Fungsi untuk membandingkan dua teks menggunakan OpenAI API
def compare_texts(text1, text2):
    # Detect the language of the first text
    lang = detect(text1)

    openai.api_key = openai_api_key
    prompt = f"Compare the following two texts:\n\nPDF 1:\n{text1[:2000]}\n\nPDF 2:\n{text2[:2000]}\n\nComparison:"

    # Using the 'openai.Completion.create' method
    response = openai.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=1000,  # Reduce the token limit for the completion
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    if lang == "id":
        # Translate the response to the desired language
        translated_response = translate_to_indonesian2(response.choices[0].text.strip())
    else:
        translated_response = response.choices[0].text.strip()

    return translated_response

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    success_message = None
    result = None

    if request.method == 'POST':
        pdf_files = request.files.getlist('pdf_docs')
        # user_question = request.form['user_question']

        # Use binary mode ('rb') when reading PDF files
        texts = [extract_text_from_pdf(pdf.stream) for pdf in pdf_files]

        result = compare_texts(*texts)

        success_message = "Comparison successful!"

    return render_template('compare.html', success_message=success_message, result=result)

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

            # Hapus semua file dalam folder UPLOAD_FOLDER
            for uploaded_file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path_to_delete = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
                try:
                    if os.path.isfile(file_path_to_delete):
                        os.remove(file_path_to_delete)
                except Exception as e:
                    print(f"Error deleting file {file_path_to_delete}: {str(e)}")

            return redirect(request.url)  # Redirect to GET after POST to prevent resubmission

    # GET request or initial page load
    final_summary = session.pop('final_summary', None)  # Retrieve and remove summary from session
    response = make_response(render_template('pdf_summarize.html', summary=final_summary))
    # Set headers to disable caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/about_us.html')
def about_us():
    return render_template('about_us.html')

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
