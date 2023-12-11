import os
from flask import Flask, request, render_template, send_file, flash, redirect, make_response, session
from werkzeug.utils import secure_filename
from openai import OpenAI
from langdetect import detect
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')   
app.secret_key = 'super secret key'  # Needed for flash messages

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
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True)
    output_summary = chain.run(texts)
    return output_summary

# Function to translate text to Indonesian
def translate_to_indonesian(text):
    response = client.completions.create(
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

@app.route('/', methods=['GET', 'POST'])
def index():
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
            return redirect(request.url)  # Redirect to GET after POST to prevent resubmission

    # GET request or initial page load
    final_summary = session.pop('final_summary', None)  # Retrieve and remove summary from session
    response = make_response(render_template('index.html', summary=final_summary))
    # Set headers to disable caching
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)
