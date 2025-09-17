
# Importando Bibliotecas
import whisper
import gradio as gr
import static_ffmpeg
import pandas as pd



def transcribe(audio):
    
    #time.sleep(3)
    # Carrega o áudio e faz ele ter 30s no máximo
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # Cria um espectograma na escala log-Mel e passa para o mesmo dispositivo do modelo
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Decodifica e configura o áudio
    options = whisper.DecodingOptions(language="pt")
    result = whisper.decode(model, mel, options)
    
    # Retorna a transcrição
    return result.text

# Adicionando ffmpeg aos paths do sistema
static_ffmpeg.add_paths()

# Define o modelo da I.A como "base"
model = whisper.load_model("base")

# Cria interface web
gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(),
    title="OpenAI Whisper ASR Gradio Web UI",
    live=True
).launch()

# Manipulação da planilha de estoque
df = pd.read_excel('estoque.xlsx')

# Salvar de volta
df.to_excel("estoque.xlsx", index=False)