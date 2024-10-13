import gradio as gr
from transformers import pipeline
from mistralai import Mistral
import cv2



# Initialize pipelines
pipe_tts = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits") # Or another suitable TTS model

messages = []

def process_video(video):
    images = []
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    while success:
        if count % (fps*5) == 0: # process a frame every 5 seconds to reduce processing
            try:
                images.append(image)
            except Exception as e:
                print(f"Error generating caption for frame: {e}")

        success,image = vidcap.read()
        count += 1
    vidcap.release()

    return images




def respond(message, image, video):
    global messages
    processed_video = None  # Initialize outside of the conditional
    if video is not None:
        processed_video = process_video(video)
        for image in processed_video:
            messages.append({"role": "user", "content": image})


    if image is not None:
        messages.append({"role": "user", "content": image})



    if isinstance(message, str) and message.strip() != "":
        messages.append({"role": "user", "content": message})


    response = generate_response(messages)
    messages.append({"role": "assistant", "content": response})

    audio_output = pipe_tts(response, voice="en-us_low")

    chat_history = ""
    for msg in messages:
        if msg["role"] == "user":
            chat_history += "User: " + msg["content"] + "\n"
        else:
            chat_history += "Assistant: " + msg["content"] + "\n"

    return chat_history, audio_output


def generate_response(messages):
    if not messages:
        return "Hello! How can I help you today?"
    try:
        pixtral_response = pixtral_client.chat.completions.create(messages=messages)
        response_text = pixtral_response.choices[0].message.content
        return response_text

    except Exception as e:
        print(f"Error with Pixtral API: {e}")
        return "I'm having some trouble right now. Please try again later."



with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Pixtral Multimodal Chat")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Enter text", placeholder="Type your message here...")
            img = gr.Image(type="pil", label="Upload image")
            video = gr.Video(type="filepath", label="Upload video") # added video input
            btn = gr.Button("Send")

    btn.click(respond, inputs=[txt, img, video], outputs=[chatbot, gr.Audio()])


demo.launch()