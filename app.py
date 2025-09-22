
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
    
    transcricao = result.text
    
    # Retorna a transcrição
    return transcricao


def interpretar_comando(transcrito):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents= f"""Você é um assistente que interpreta pedidos de controle de estoque hospitalar.
                    Corrija eventuais erros de transcrição e retorne apenas uma **LISTA de JSONs** no seguinte formato:

                    [
                      {{
                        "acao": "adicionar ou retirar",
                        "quantidade": número inteiro,
                        "item": "nome do insumo"
                      }},
                      {{
                        "acao": "...",
                        "quantidade": ...,
                        "item": "..."
                      }}
                    ]

                    Exemplo:
                    Entrada: "tirei duas luvas e uma gaze"
                    Saída:
                    [
                      {{
                        "acao": "retirar",
                        "quantidade": 2,
                        "item": "LUVAS"
                      }},
                      {{
                        "acao": "retirar",
                        "quantidade": 1,
                        "item": "GAZES"
                      }}
                    ]

                    Para 'acao' você deve retornar exatamente 'adicionar' ou 'retirar'.
                    Para 'quantidade' sempre um inteiro.
                    Para 'item' use exatamente uma dessas opções: 'SERINGAS', 'GAZES', 'LUVAS', 'TUBOS_DE_COLETA'.
                    Retorne apenas a lista JSON, nada além disso.

                    Agora, interprete o seguinte: {transcrito}
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

def pipeline(audio):
    transcricao = transcribe(audio)
    comandos = interpretar_comando(transcricao)  # agora vem lista de dicts

    # Carrega estoque
    df = pd.read_excel("estoque.xlsx")

    for comando in comandos:
        item = comando["item"].upper()

        if item not in df.columns:
            continue  # ignora item inválido

        if comando["acao"] == "adicionar":
            df.loc[0, item] += comando["quantidade"]
        elif comando["acao"] == "retirar":
            df.loc[0, item] -= comando["quantidade"]

    # Salva de volta
    df.to_excel("estoque.xlsx", index=False)

    return transcricao, comandos

# Adicionando ffmpeg aos paths do sistema
static_ffmpeg.add_paths()

# Define o modelo da I.A como "base"
model = whisper.load_model("base")

# Cria interface web
gr.Interface(
    fn=pipeline,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Textbox(label="Transcrição"), gr.JSON(label="Interpretação")],
    title="OpenAI Whisper + Gemini Estoque",
    live=True
).launch()