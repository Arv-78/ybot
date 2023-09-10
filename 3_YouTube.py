import streamlit as st
import time
import openai
import shutil
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import json
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import YoutubeLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from markdownify import markdownify as md
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from pydub import AudioSegment
import pathlib
load_dotenv()
LANG_MODEL = 'gpt-3.5-turbo-16k'
EMBED_MODEL = 'text-embedding-ada-002'

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
def text_splitter():
    return RecursiveCharacterTextSplitter(separators=['\n\n','\n','##','. ',', '], chunk_size=4096, chunk_overlap=500)


splitter = text_splitter()



#ytVidsMethod module dependancie
def store_docs_as_json(docs, text_path, source=''):
    if not source:
        source = docs[0].metadata['source']
    json_data = {}
    for i in range(len(docs)):
        json_data[str(i)] = {'page_content': docs[i].page_content, 'metadata':docs[i].metadata}

    with open(f"{text_path}/{source}.json",'w') as f:
        json.dump(json_data, f, indent=4)
    return


def split_audio_file(input_file, output_directory, audio_format='mp3', chunk_duration=10*60*1000, overlap_duration=5*1000):
    shutil.rmtree(output_directory, ignore_errors=True)

    # Load the audio file
    audio = AudioSegment.from_file(input_file, audio_format)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Calculate the number of chunks
    if len(audio) > chunk_duration:
        num_chunks = 0
    else:
        num_chunks = len(audio) // (chunk_duration - overlap_duration)

    # Split the audio into chunks
    for i in range(num_chunks + 1):
        start_time = i * (chunk_duration - overlap_duration)
        end_time = start_time + chunk_duration
        if end_time > len(audio):
            end_time = len(audio)

        # Extract the chunk
        chunk = audio[start_time:end_time]

        # Save the chunk to a new file
        output_file = os.path.join(output_directory, f"chunk_{i}.{audio_format}")
        chunk.export(output_file, format=audio_format)

    print(f"{num_chunks+1} chunks created successfully.")
    return len(audio)


def transcribe_by_whisper(filename, audio_path, audio_format='mp3'):
    TEMP_AUDIO_PATH = f'{audio_path}/temp'
    length = split_audio_file(input_file=f'{audio_path}/{filename}.{audio_format}', output_directory=TEMP_AUDIO_PATH)
    transcripts = {}
    try:
        for i, filename in tuple(enumerate(os.listdir(TEMP_AUDIO_PATH))):
            audio_file= open(f"{TEMP_AUDIO_PATH}/{filename}", "rb")
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcripts[i] = dict(transcript)['text']

        return transcripts, length
    except:
        print('\nSome Error occured while transcribing using Whisper.')


def extract_yt_audio(url, audio_path, format='mp3'):
    filename = extract_video_id(url)
    ydl_opts = {
        "format": "m4a/bestaudio/best",
        "noplaylist": True,
        "outtmpl": f"{audio_path}/{filename}",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": format,
            }
        ],
    }

    # Download file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
    return 


def transcript_to_doc_formatter(transcript, metadata):
    t_list = transcript.fetch()
    text = ''.join([t['text'] for t in t_list])
    return [Document(page_content=text, metadata=metadata)]


def transcribe_yt(URL, metadata, audio_path):
    video_id = extract_video_id(URL)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        manually_created_lang_codes = list(map(lambda x: x.language_code,
                            filter(lambda x: not x.is_generated and 'en' in x.language_code, 
                                    transcript_list)
                                    ))
        if manually_created_lang_codes:
            print(f'\nSubtitle found for {URL}. Extracting...')
            transcript = transcript_list.find_manually_created_transcript(manually_created_lang_codes[:1])
            metadata_embed_docs = transcript_to_doc_formatter(transcript, metadata)
        else:
            print(f'\nexception found')
            raise Exception
    
    except:    
        print(f'\nNo existing transcription or embedding of {URL}. Extracting Audio...')
        # loader = GenericLoader(YoutubeAudioLoader([URL], AUDIO_PATH), OpenAIWhisperParser())
        # metadata_embed_docs = loader.load()
        extract_yt_audio(URL, audio_path)

        # audio_file= open(f"{AUDIO_PATH}/{video_id}.mp3", "rb")
        # transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcripts, _ = transcribe_by_whisper(video_id, audio_path=audio_path)

        metadata_embed_docs = []
        for t in transcripts:
            metadata_embed_docs.append(Document(page_content=transcripts[t], metadata=metadata))

        # for doc in metadata_embed_docs:
        #     doc.metadata=metadata
    return metadata_embed_docs


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        return True
    else:
        return False


def store_embeddings_mod(docs, store_name, embed_path):
    if not check_and_create_directory(f"{embed_path}/{store_name}") and os.listdir(f"{embed_path}/{store_name}"):
        return load_embeddings(store_name, path=embed_path)

    vectorStore = FAISS.from_documents(docs, load_model())
    vectorStore.save_local(f"{embed_path}/{store_name}")
    return vectorStore


def jsonLoader(filename, text_path, metadata={}):
    docs_ = []
    with open(f'{text_path}/{filename}.json','r') as f:
        json_data = json.load(f)
        
    for element in json_data:
        if not metadata:
            docs_.append(Document(page_content=json_data[element]['page_content'],metadata=json_data[element]['metadata']))
        else:
            docs_.append(Document(page_content=json_data[element]['page_content'],metadata=metadata))
    
    with st.expander('Show Transcription'):
        st.write(md(f'\n#### Transcription\n{docs_[0].page_content}'))

    return docs_


def load_model(type=''):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    if type=='llm':
        return ChatOpenAI(temperature=0.1, model=LANG_MODEL)
    else:
        return OpenAIEmbeddings(model=EMBED_MODEL)


def load_embeddings(sotre_name, path, embedding_model = load_model()):
    return FAISS.load_local(f"{path}/{sotre_name}", embedding_model)


def extract_video_id(URL):
    loader = YoutubeLoader(URL)
    return loader.extract_video_id(URL)


def generate_embed_dict(urls, main_embed_path, text_path, audio_path):
    embeddings = {}

    for url in urls:
        loader = YoutubeLoader(url)
        video_id = extract_video_id(url)
        metadata = loader._get_video_info()
        metadata['source'] = video_id

        if video_id in os.listdir(main_embed_path):
            print(f'\nEmbedding already exists for {url} at {main_embed_path}/{video_id}. Loading...')
            embeddings[video_id] = load_embeddings(video_id, main_embed_path, load_model())
            continue
        elif f'{video_id}.json' in os.listdir(text_path):
            print('\nJSON file exists. Creating embedding...')
            docs_whole = jsonLoader(f'{video_id}.json', text_path=text_path)
            docs = splitter.split_documents(docs_whole)
            embeddings[video_id] = store_embeddings_mod(docs, video_id, main_embed_path)
            continue

        print(f'\nNo existing embedding for {url}. Transcribing...')
        metadata_embed_docs = transcribe_yt(url, metadata, audio_path)

        print(f'\nCreating {video_id}.json')
        store_docs_as_json(metadata_embed_docs, text_path=text_path, source=video_id)
        docs = splitter.split_documents(metadata_embed_docs)

        print(f'\nStoring embedding of {url} at {main_embed_path}/{video_id}')
        embeddings[video_id] = store_embeddings_mod(docs, video_id, main_embed_path)

    return embeddings


def yt_to_embed(urls, main_embed_path, temp_embed_path, text_path, audio_path):
    embed_dict = generate_embed_dict(urls, main_embed_path, text_path, audio_path)
    video_ids = list(embed_dict.keys())
    initiated = False

    shutil.rmtree(temp_embed_path)

    for id in video_ids:
        shutil.copytree(f'{main_embed_path}/{id}',f'{temp_embed_path}/{id}')

        if not initiated:
            initiated = True
            merged_embed = FAISS.load_local(f'{temp_embed_path}/{id}', load_model())
        else:
            embed_temp = FAISS.load_local(f'{temp_embed_path}/{id}', load_model())
            merged_embed.merge_from(embed_temp)
        
    return merged_embed


#helper module functions
def parse_urls(urls_string):
    """Split the string by comma and strip leading/trailing whitespaces from each URL."""
    return [url.strip() for url in urls_string.split(',')]


def load_model(type=''):
    if type=='llm':
        return ChatOpenAI(temperature=0.1, model=LANG_MODEL)
    else:
        return OpenAIEmbeddings(model=EMBED_MODEL)


def qa_chain(db_Embedd, llm=load_model(type='llm')):
    retriever = db_Embedd.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa_chain

#Sagar's code
DATA_PATH = 'data'
OUTPUT_PATH = 'model'
EMBEDDING_PATH = f'{OUTPUT_PATH}/embedding'
VIDEO_PATH = f'{DATA_PATH}/video'
AUDIO_PATH = f'{DATA_PATH}/audio'
TEXT_PATH = f'{DATA_PATH}/text'
TEMP_EMBEDDING_PATH = f'{OUTPUT_PATH}/temp'
TEMP_AUDIO_PATH = f'{AUDIO_PATH}/temp'



for file in os.listdir(TEMP_EMBEDDING_PATH):
    shutil.rmtree(f'{TEMP_EMBEDDING_PATH}/{file}')
os.makedirs(TEMP_EMBEDDING_PATH, exist_ok=True)


def get_initial_message():
    messages=[
            {"role": "system", "content": "You are a helpful AI Tutor. Who anwers brief questions about AI."},
            {"role": "user", "content": "I want to learn AI"},
            {"role": "assistant", "content": "Thats awesome, what do you want to know aboout AI"}
        ]
    return messages


def get_chatgpt_response(query, chain = qa_chain):
    instruction = f"""
        % QUERY
        {query}

        % INSTRUCTION
        If answer has more than 30 words, format it in pointwise manner with point header in bold letters.
        """
    response = chain(instruction)
    return response['result']


def update_chat(messages, role, content):
    messages.append({"role": role, "content": content})
    return messages


def url_input(vid_urls):
    urls = parse_urls(vid_urls)
    merged_yt_embedding = yt_to_embed(urls, main_embed_path=EMBEDDING_PATH, temp_embed_path=TEMP_EMBEDDING_PATH, text_path=TEXT_PATH, audio_path=AUDIO_PATH)
            
    qa_chain_yt = qa_chain(merged_yt_embedding)
    return qa_chain_yt


## Fronend code from here
css = """
    <style>
        .title {
            position: fixed;
            top: 45px;
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            height: 70px;
            z-index: 1;
            background-color: #0E1117;
        }
        .backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 100;
            background-color: white;
            opacity: 1%;
        }
        @media (prefers-color-scheme: light) {
            .title {
            position: fixed;
            top: 45px;
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            height: 70px;
            z-index: 1;
            background-color: #ffff;
        }
        
    </style>
"""

overlay='''
    <div class="backdrop">
        Hello World
    </div>
'''

st.markdown(css, unsafe_allow_html=True)

st.markdown("<h1 class='title'>Youtube Assistant</h1>", unsafe_allow_html=True)


if "youtube_chat_history" not in st.session_state:
        st.session_state.youtube_chat_history = []
if "youtube_qa" not in st.session_state:
    st.session_state.youtube_qa = None


def youtube_chatbot():

    
    youtube_videos = st.sidebar.text_area(label="YouTube URLs", placeholder="Ex: https://www.youtube.com/watch?v=c_hO_fjmMnk", key="youtube_user_input")
    sub_btn = st.sidebar.button("submit")
    if youtube_videos and sub_btn:
        # st.markdown(overlay, unsafe_allow_html=True)
        overlay = st.warning('⭕ Please wait for the process to complete')
        st.session_state.youtube_qa = url_input(youtube_videos)
        overlay.empty()
        
        # time.sleep(10)
        st.experimental_rerun()
         


    for message in st.session_state.youtube_chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    if query := st.chat_input():
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.youtube_chat_history.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("Generating the response..."):
                response = ""
                if st.session_state.youtube_qa:
                    response = get_chatgpt_response(query, st.session_state.youtube_qa)
                else:
                    response = "Please provide the video links"
                st.markdown(response)
                st.session_state.youtube_chat_history.append({"role": "assistant", "content": response})
    
    if len(st.session_state.youtube_chat_history) > 0:
        clr_btn = st.button("Clear history")
        if clr_btn:
            st.session_state.youtube_chat_history = []
            st.experimental_rerun()


youtube_chatbot()
