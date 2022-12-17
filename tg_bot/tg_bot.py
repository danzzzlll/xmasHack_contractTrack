import telebot
import torch
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizerFast, BertForSequenceClassification
import re
import nltk
from docx2pdf import convert
import aspose.words as aw
from wordcloud import WordCloud
from nltk.corpus import stopwords
import PyPDF2
import pymorphy2

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

morph = pymorphy2.MorphAnalyzer()

Token = '_____________'

bot = telebot.TeleBot(Token)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

@bot.message_handler(content_types=['text', 'document'])
def handle_message(message):
    if message.content_type == 'text':
        bot.reply_to(message, "Пожалуйста, отправьте документ который необходимо обработать.")
    elif message.content_type == 'document':
        doc = message.document
        if doc.file_name.split('.')[1] in ['doc', 'docx', 'pdf', 'rtf']:
            raw = message.document.file_id
            file_info = bot.get_file(raw)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(doc.file_name, 'wb') as new_file:
                new_file.write(downloaded_file)
                new_file.close()
            convert_file_to_pdf(doc.file_name)
            text = extract_file_text(doc.file_name)
            clean_text = cleanText(text)
            text = lemmatized(clean_text)
            sentences = get_words(transformer, clean_text, 5, 15)
            print(sentences)
            phrases = 'Фразы:\n' + '\n'.join(sentences[:5])
            prediction_str = 'Предикт:\n' + str(get_prediction(text)) + '\n\n'
            image = cloud_maker(sentences)
            image.to_file("cloud_text.png")
            bot.reply_to(message, prediction_str + phrases)
            with open ('cloud_text.png', 'rb') as image:
            	bot.send_photo(message.chat.id, image)
        else:
            bot.reply_to(message, "Принимаются только файлы форматов doc, docx, pdf, rtf.")



labels = {
    'Договоры оказания услуг': 0,
    'Договоры купли-продажи': 1,
    'Договоры аренды': 2,
    'Договоры подряда': 3,
    'Договоры поставки': 4,
}

target_names = [
    'Договоры оказания услуг',
    'Договоры купли-продажи',
    'Договоры аренды',
    'Договоры подряда',
    'Договоры поставки',
]

model_name = "sberbank-ai/ruBert-base"
max_length = 512

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
transformer = SentenceTransformer(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names))
model.load_state_dict(torch.load('pytorch_model.bin', map_location=torch.device(device)))
model.to(device)


def convert_file_to_pdf(path):
    if path.endswith('.rtf') or path.endswith('.doc'):
        doc = aw.Document(path)
        doc.save(path.split('.')[0] + '.docx')
    if not path.endswith('.pdf'):
        convert(path.split('.')[0] + '.docx')


def extract_file_text(path):
    path = path.split('.')[0] + '.pdf'

    pdf_file_obj = open(path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    pages_count = pdf_reader.numPages
    text = ''
    for i in range(pages_count):
        page_obj = pdf_reader.getPage(i)
        text += '\n' + page_obj.extractText()
    return text


def lemmatized(text):
    splited = text.split(' ')
    for i in range(len(splited)):
        if len(splited[i]) == 1 or len(splited[i]) == 2:
            splited[i] = ''
        splited[i] = morph.parse(splited[i])[0].normal_form
    text = listToString(splited)
    return text


def listToString(s):
 
    str1 = " "
 
    for ele in s:
        str1 += ele + " "
 
    return str1


def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    max1 = 0
    max2 = 0
    for i in range(len(probs[0])):
        if probs[0][i] > probs[0][max1]:
            max2 = max1
            max1 = i
    if probs[0][max2] > 0.2 and probs[0][max1] != probs[0][max2]:
        return f"""
№1 : {target_names[max1]}: {str(probs[0][max1])[7:13]},
№2 : {target_names[max2]}: {str(probs[0][max2])[7:13]},
"""
    return target_names[max1] + ': ' + str(probs[0][max1])[7:13]


def cleanText(text):
    text = text.lower()
    
    text = re.sub('(-)', '', text)
    
    text = re.sub('\n', '', text)
            
    text = re.sub(r'[_\n/ _–<\*+«,\#+\№\"\-+\_+\=+{\?+\»%!+\&\}^\+\;\+\>«+"\(\)\/+\:\\+.]', r' ', text)
    
    text = re.sub(r'[abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ]', r' ', text)
    
    text = re.sub(r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)', r' ', text)
    
    text = re.sub(r'(\d+\s\d+)|(\d+)',' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\uf06c\uf020', ' ', text)

    return text


def get_words(model, text, size=5, how_many=5):
    n_gram_range = (size, size)
    count = CountVectorizer(ngram_range=n_gram_range).fit([text])
    candidates = count.get_feature_names_out()
    doc_embedding = model.encode([text])
    candidate_embeddings = model.encode(candidates)
    top_n = how_many
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords


russian_stopwords = stopwords.words("russian")

word_cloud = WordCloud(
    background_color='white',
    stopwords=russian_stopwords, 
    height=400,
    width=300
)


def cloud_maker(text_list):
    text_split = ''
    for i in text_list:
        for j in i.split(' '):
            text_split += j+' '
    word_cloud.generate(text_split)
    return word_cloud.to_file('cloud_text.png')


print('ready')
bot.polling(none_stop=True, interval=1)
