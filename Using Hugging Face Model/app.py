from flask import Flask, render_template, request
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, TFAutoModelForSequenceClassification

app = Flask(__name__)

chk_summ = "Alex034/t5-small-indosum-summary-freeze"
chk_typo = "Alex034/typo_classifier_2023"

# Summarization model
summarization_tokenizer = AutoTokenizer.from_pretrained(chk_summ)
summarization_model = TFAutoModelForSeq2SeqLM.from_pretrained(chk_summ)

# Typo classifier model
typo_classifier_tokenizer = AutoTokenizer.from_pretrained(chk_typo)
typo_classifier_model = TFAutoModelForSequenceClassification.from_pretrained(chk_typo)

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
        id2label = {0: 'Correct', 1: 'Typo'}
        predicted_label = id2label[label_indices[0]]

        return render_template('classify.html', user_input=user_input, predicted_label=predicted_label)
    return render_template('classify.html', user_input='', predicted_label='')

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
