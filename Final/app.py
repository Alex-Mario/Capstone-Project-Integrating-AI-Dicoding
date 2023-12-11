from flask import Flask, render_template, request
import tensorflow as tf
import openai
import time
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, TFAutoModelForSequenceClassification

api_key = 'sk-xxxxxxxxxxxxx'

app = Flask(__name__)

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
        outputs = summarization_model.generate(inputs.input_ids, max_length=256)
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

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
