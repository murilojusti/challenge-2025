# Importando Bibliotecas
import whisper

# Define o modelo da I.A como "base"
model = whisper.load_model("base")

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