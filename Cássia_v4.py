import requests
import json
import speech_recognition as sr
import pyttsx3

cassia = pyttsx3.init()
cassia.setProperty('voice', r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_PT-BR_MARIA_11.0')
cassia.setProperty('volume', 1.0)
cassia.setProperty('rate', '160')
recognizer = sr.Recognizer()

# for voz in cassia.getProperty('voices'):
#     print(voz)

url = "http://localhost:11434/api/generate"

def recognize_command():
    with sr.Microphone() as mic:
        try:
            recognizer.adjust_for_ambient_noise(mic, duration=2)
            print('Estou te ouvindo...')
            audio = recognizer.listen(mic)
            text = recognizer.recognize_google(audio, language='pt')
            return text.lower()
        except sr.UnknownValueError:
            cassia.say('Desculpe, não entendi')
            print('Desculpe, não entendi')
            cassia.runAndWait()
            return None
        except sr.RequestError:
            cassia.say('Erro ao conectar com o serviço de reconhecimento de voz. ')
            print('Erro ao conectar com o serviço de reconhecimento de voz. ')
            cassia.runAndWait()
            return None

def process_llama_query(pergunta):
    input_json = {
        'model' : 'llama3.1',
        'prompt' : 'Responda em português' + pergunta
    }
    response = requests.post(url, json=input_json)

    linhas = response.text.strip().split('\n')
    valores_response = []

    for linha in linhas:
        obj = json.loads(linha)
        resposta = obj.get('response')
        valores_response.append(resposta)

    return ''.join(valores_response)

def processing_command():
    res = True
    while res:
        print("Iniciando...")
        comando = recognize_command()

        if comando in ['ok cássia', 'okcássia', 'ok cássias', 'ok, cássia',
                       'ok cassia', 'okcassia', 'ok cassias', 'ok cassia']:
            cassia.say('A disposição')
            print('A disposição')
            cassia.runAndWait()

            while res:
                comando = recognize_command()

                if comando:
                    if 'cadastrar evento' in comando:
                        cassia.say('Ok, qual evento devo cadastrar ?')
                        print('Ok, qual evento devo cadastrar ?')
                        cassia.runAndWait()
                        evento = recognize_command()
                        if evento:
                            with open('agenda.txt', 'a', encoding='utf-8') as arquivo:
                                arquivo.write(evento + '\n')
                            cassia.say('Evento cadastrado com sucesso. ')
                            print('Evento cadastrado com sucesso')
                            cassia.runAndWait()
                        pass
                    elif 'minha agenda' in comando:
                        try:
                            with open('agenda.txt', 'r') as arquivo:
                                agenda = arquivo.read()
                            cassia.say('Aqui estão os eventos cadastrados: ')
                            cassia.say(agenda)
                            cassia.runAndWait()
                        except FileNotFoundError:
                            cassia.say('A agenda está vazia ou não foi encontrada. ')
                            print('A agenda está vazia ou não foi encontrada. ')
                            cassia.runAndWait()
                        pass
                    elif 'pergunta' in comando:
                        cassia.say('O que você gostaria de saber ? ')
                        print('O que você gostaria de saber ? ')
                        cassia.runAndWait()
                        pergunta = recognize_command()
                        if pergunta:
                            resposta = process_llama_query(pergunta)
                            cassia.say(resposta)
                            print(resposta)
                            cassia.runAndWait()
                        pass
                    elif comando in ['obrigado', 'obrigados', 'ok, obrigado', 'ok, obrigados'
                                     'okobrigados', 'okobrigado']: #
                        cassia.say("De nada! Até logo.")
                        print("De nada! Até logo.")
                        cassia.runAndWait()
                        res = False  # Encerra o processo
                else:
                    cassia.say("Não entendi o que você disse. Por favor, repita.")
                    print("Não entendi o que você disse. Por favor, repita.")
                    cassia.runAndWait()

        else:
            print("Diga, 'ok cássia'...")

if __name__ == "__main__":
    processing_command()
