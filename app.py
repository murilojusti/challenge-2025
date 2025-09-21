
# Importando Bibliotecas
import whisper
import gradio as gr
import static_ffmpeg
import pandas as pd
from google import genai
import os
from dotenv import load_dotenv
import json

load_dotenv() # Carrega o env

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


def interpretar_comando(transcrito):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents= f"""Você é um assistente que interpreta pedidos de controle de estoque hospitalar.
                    Corrija eventuais erros de transcrição e retorne apenas um JSON no seguinte formato:

                    {{
                      "acao": "adicionar ou retirar",
                      "quantidade": número inteiro,
                      "item": "nome do insumo"
                    }}

                    Exemplo:
                    Entrada: "dar baixa em duas seringas"
                    Saída:
                    {{
                      "acao": "retirar",
                      "quantidade": 2,
                      "item": "seringa"
                    }}

                    Para 'acao' você deve retornar exatamente 'adicionar' ou 'retirar' dependendo do contexto.
                    Para 'quantidade' você deve retornar a quantidade exata que entender no contexto
                    Para 'item' você deve retornar uma dessas opções exatamente como está escrito a seguir: 'SERINGAS', 'GAZES', 'LUVAS', 'TUBOS_DE_COLETA'.
                    Tudo o que você deve retornar é o JSON, nem uma palavra a mais.

                    Agora, interprete o seguinte {transcrito}
                    """,
    )

    # pegar texto cru
    resposta_texto = response.candidates[0].content.parts[0].text.strip()

    # remover crases e "json" do bloco markdown
    if resposta_texto.startswith("```"):
        resposta_texto = resposta_texto.strip("`")  # tira as crases
        resposta_texto = resposta_texto.replace("json", "", 1).strip()  # tira o "json" inicial, se tiver

    # carregar como json
    resposta_json = json.loads(resposta_texto)
    print( resposta_json)

    return resposta_json

interpretar_comando('colocar cinco seringas')

# # Adicionando ffmpeg aos paths do sistema
# static_ffmpeg.add_paths()

# # Define o modelo da I.A como "base"
# model = whisper.load_model("base")

# # Cria interface web
# gr.Interface(
#     fn=transcribe,
#     inputs=gr.Audio(type="filepath"),
#     outputs=gr.Textbox(),
#     title="OpenAI Whisper ASR Gradio Web UI",
#     live=True
# ).launch()

# # Manipulação da planilha de estoque
# df = pd.read_excel('estoque.xlsx')



# # Salvar de volta
# df.to_excel("estoque.xlsx", index=False)