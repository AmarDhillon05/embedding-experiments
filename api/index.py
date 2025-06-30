import base64
import numpy as np
import soundfile as sf
import random
import string
import time
import torch
import torchaudio
import os 
import sys
import tempfile
import cv2
import flask
import gc
from torch import nn
from PIL import Image
from io import BytesIO
from nomic import embed
from memory_profiler import profile
from transformers import CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import DataLoader
from flask import Flask, request

os.system("clear")

#Models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")



#Other utils
def file_to_base64(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
    


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string



pool = nn.AvgPool1d(kernel_size=2, stride=3, padding=0)






#Embedding functions
def embed_text(text, batch_size = None):
    
    start = time.time()

    bs = batch_size
    if bs == None:
        bs = len(text)

    embeddings = []
    while len(text) != 0:
        e = embed.text(text[:bs])['embeddings']
        embeddings += e
        text = text[bs:]

        del e 
        gc.collect()
    
    end = time.time()
    print(f"Ran in {end - start}s")

    return np.array(embeddings).flatten().tolist()




def embed_image(images_b64, batch_size = 1):
    print("Requested to embed an image")
    start = time.time()

    all_embeddings = []
    clip_model.eval()  # just to be sure

    with torch.no_grad():
        for i in range(0, len(images_b64), batch_size):
            batch_b64 = images_b64[i:i+batch_size]

            # Decode base64 and load PIL images only for this batch
            batch_images = [Image.open(BytesIO(base64.b64decode(b64))).convert("RGB") for b64 in batch_b64]
            print("Retreived batch images")

            # Process batch through CLIP
            inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
            outputs = clip_model.get_image_features(**inputs)
            all_embeddings.append(outputs)

            del batch_b64; del batch_images; del inputs; del outputs
            gc.collect()


    all_embeddings = torch.cat(all_embeddings, dim=0)

    end = time.time()
    print(f"Ran in {end - start}s")

    return np.array(all_embeddings).flatten().tolist()






def embed_audio(audio, batch_size = 1):

    start = time.time()

    #Doesn't use batch size rn, will fix later (prolly not)
    all_e = []

    for b64_audio_str in audio:
        # Step 1: Decode base64 to bytes
        audio_bytes = base64.b64decode(b64_audio_str)
        audio_buffer = BytesIO(audio_bytes)

        waveform, sample_rate = torchaudio.load(audio_buffer)

        # Resample if needed (Wav2Vec2 expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Mono channel only
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs = w2v_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

        w2v_model.eval()
        with torch.no_grad():
            outputs = w2v_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pool across time
            all_e.append(torch.flatten(embeddings))
            
            del outputs
            del embeddings

        del audio_buffer; del audio_bytes; del waveform
        gc.collect()


    end = time.time()
    print(f"Ran in {end - start}s")

    return np.array(all_e).flatten().tolist()




def embed_video(base64_videos, batch_size=12, every_n_frames=10): #Batch size is patch size here my bad
    
    start = time.time()

    clip_model.eval()
    video_embeddings = []

    with torch.no_grad():
        for base64_str in base64_videos:
            # Decode to temp file
            video_bytes = base64.b64decode(base64_str)
            with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
                temp_video.write(video_bytes)
                temp_video.flush()

                # Load video with OpenCV
                cap = cv2.VideoCapture(temp_video.name)
                frames = []
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % every_n_frames == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                    count += 1
                cap.release()

                # Batch frames → embeddings
                all_embeddings = []
                for i in range(0, len(frames), batch_size):
                    batch = frames[i:i+batch_size]
                    inputs = clip_processor(images=batch, return_tensors="pt", padding=True)
                    feats = clip_model.get_image_features(**inputs)
                    all_embeddings.append(feats)

                # Mean-pool over frames
                if all_embeddings:
                    video_tensor = torch.cat(all_embeddings, dim=0)
                    pooled = video_tensor.mean(dim=0)
                    video_embeddings.append(pooled)
                else:
                    video_embeddings.append(torch.zeros(512))  # fallback if video is empty

                del temp_video; del all_embeddings;
                gc.collect()

    end = time.time()
    print(f"Ran in {end - start}s")

    return np.array(video_embeddings).flatten().tolist() # List of [512]-dim tensors




#Downsampling by pooling and padding 
def downsample_to_size(array, target_length):
    while len(array) > target_length:
        array = pool(
            torch.tensor([array]).float()
        ).tolist()[0]
    
    array += [0 for _ in range(target_length - len(array))]
    return array



#Splitting lengths for embeddings and padding each (might not be the best approach)
#Should keep same order and length for embeddings for diff categories tho
def fit_embeddings_to_size(array, target_length):
    
    individual_length = target_length // len(array)
    fit_embeddings = [
        downsample_to_size(e, individual_length)
         for e in array[:-1]]
    fit_embeddings.append(
        downsample_to_size(array[-1], target_length - (individual_length * (len(array) - 1)))
    )

    flattened = []
    for i in fit_embeddings:
        flattened = flattened + i


    return flattened



#Single test
@profile
def test():
    text = [generate_random_string(1024) for _ in range(3)]
    audio = [file_to_base64("samples/city.m4a") for _ in range(2)]
    video = [file_to_base64("samples/blood.mp4") for _ in range(2)]
    image = [file_to_base64("samples/penguin.jpg") for _ in range(2)]

    text_embed = embed_text(text); print(f"Text embed done, size {np.array(text_embed).shape}\n")
    audio_embed = embed_audio(audio); print(f"Audio embed done, size {np.array(audio_embed).shape}\n")
    video_embed = embed_video(video); print(f"Video embed done, size {np.array(video_embed).shape}\n")
    image_embed = embed_image(image); print(f"Image embed done, size {np.array(image_embed).shape}\n")

    complete = text_embed + audio_embed + video_embed + image_embed
    complete = fit_embeddings_to_size(complete, 1024)
    print(f"Complete done, shape {np.array(complete).shape}")




try:
    #test()
    print("Skipping test for now")
except Exception as e:
    print(f"Ran into testing error: {e}")





#Flask endpoint

app = Flask(__name__)

@app.route('/')
def home():
    return "hi from ap embed"


@app.route("/embed", methods = ['POST'])
def embed_one():
    try:
        json_data = request.get_json() 
        dtype = json_data.get("type")
        data = json_data.get("data")  #b64 str
        bs = json_data.get("batch_size")
        funcs = {"image" : embed_image, "video" : embed_video, "audio" : embed_audio, "text" : embed_text}

        
        if bs == None:
            bs = 1

        f = funcs[dtype]

        return {
            "embeddings" : f([data], bs)
        }

    except Exception as e:
      
        print(f"Ran into {e}")
        return {
            "error" : f"{e}"
        }


@app.route("/embed_mult", methods = ['POST'])
def embed_mult():
    try:
        json_data = request.get_json()
        data = json_data.get("data")  #[ {"data" : ..., "type" : image}, ...]
        funcs = {"image" : embed_image, "video" : embed_video, "audio" : embed_audio, "text" : embed_text}

        data = [
            funcs[el['type']]([el['data']])
        for el in data]

        for i in data:
            print(f"Have an embedding of size {len(i)}")
        fitted = fit_embeddings_to_size(data, 1024)

        return {
            "embeddings" : fitted
        }

    except Exception as e:
    
        print(f"Ran into {e}")
        return {
            "error" : f"{e}"
        }

