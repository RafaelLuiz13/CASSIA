# --------------------------------------------
# CASSIA_v40
#CÓDIGO FONTE ESCRITO EM PYCHARM NO PYTHON 3.9 EM AMBIENTE SO WIN10-PRO x64

import requests
import json
import speech_recognition as sr
import pyttsx3
import logging
import datetime
import cv2
import pytesseract
import face_recognition
import time
import numpy as np

#------------------------------------------------------------------------------------------------------------------------
#PREPARANDO A MONSTRUOSA CÁSSIA:
cassia = pyttsx3.init()
cassia.setProperty('voice', r'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_PT-BR_MARIA_11.0')
cassia.setProperty('volume', 1.0)
cassia.setProperty('rate', 160)
recognizer = sr.Recognizer()

#------------------------------------------------------------------------------------------------------------------------

yolo_weights = "yolov3.weights"
yolo_cfg = "yolov3.cfg"
yolo_classes = "coco.names"

#------------------------------------------------------------------------------------------------------------------------
#PRIMEIRA FUNÇÃO, SERVE PARA DESBLOQUEAR A CÁSSIA
def reconhecer_rosto():
    video_capture = cv2.VideoCapture(0)

    imagem_referencia = face_recognition.load_image_file("meu-rosto.jpg")
    referencia_encoding = face_recognition.face_encodings(imagem_referencia)[0]

    reconhecido = False
    tempo_inicial = 0

    while True:
        ret, frame = video_capture.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            match = face_recognition.compare_faces([referencia_encoding], face_encoding)

            if match[0]:
                cor = (0, 255, 0)
                label = "Rosto Reconhecido"
                if not reconhecido:
                    reconhecido = True
                    tempo_inicial = time.time()
            else:
                cor = (0, 0, 255)
                label = "Rosto Nao Reconhecido"

            cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

        cv2.imshow('Video', frame)

        if reconhecido and (time.time() - tempo_inicial) > 3:
            print('CÁSSIA: Olá meu rei')
            cassia.say('Olá meu rei')
            cassia.runAndWait()
            break

        # Encerra o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera a captura e fecha a janela
    video_capture.release()
    cv2.destroyAllWindows()

# Executa a função
reconhecer_rosto()

#------------------------------------------------------------------------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = r'E:\arquivos de RAFAELLUIZ\challenge2024\Sprint 3 - Deep Learning\tesseract.exe'

#------------------------------------------------------------------------------------------------------------------------
#CONFIGURAÇÃO DO .LOG PARA MELHOR VISUALIZÇÃO EM CASO DE ERROS E/OU WARNINGS
logging.basicConfig(filename='cassia.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#------------------------------------------------------------------------------------------------------------------------
#API DO LLAMA 3.1
url = "http://localhost:11434/api/generate"

#------------------------------------------------------------------------------------------------------------------------
# RECOGNIZE COMMAND É O RECONHECEDOR
def recognize_command():
    with sr.Microphone() as mic:
        try:
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            print("CÁSSIA: Estou te ouvindo...")
            audio = recognizer.listen(mic)
            text = recognizer.recognize_google(audio, language='pt')
            print(text.lower())
            return text.lower()
        except sr.UnknownValueError:
            cassia.say("Desculpe, não entendi.")
            print("CÁSSIA: Desculpe, não entendi.")
            cassia.runAndWait()
            return None
        except sr.RequestError:
            print("CÁSSIA: Erro ao conectar com o serviço de reconhecimento de voz.")
            return None
        except Exception as e:
            cassia.say(f"Ocorreu um erro: {e}")
            print(f"CÁSSIA: Ocorreu um erro: {e}")
            cassia.runAndWait()
            return None

# A FUNÇÃO DETECTAR OBJETOS USA OS MODELOS PRÉ TREINADOS DO YOLO E A BIBLIOTECA CV2, PARA RECONHECIMENTO A PARTIR DA WEBCAM
def detectar_objetos():
    cassia.say("Iniciando detecção de objetos...")
    print('CÁSSIA: Iniciando detecção de objetos...')
    cassia.runAndWait()

    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    with open(yolo_classes, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            print("CÁSSIA: Tempo limite de 60 segundos atingido. Encerrando a detecção.")
            cassia.say("Tempo limite de 60 segundos atingido. Encerrando a detecção.")
            cassia.runAndWait()
            break

        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Detecção de Objetos", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cassia.say("Detecção de objetos concluída. ")
    print("CÁSSIA: Detecção de objetos concluída. ")
    cassia.runAndWait()

#CAPTURA IMAGEM E PRINTA APÓS 10 SEGUNDOS (TEMPO MODIFICAVEL)
def captura_imagem():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cassia.say('Não foi possivel acessar a câmera. ')
        print('CÁSSIA: Não foi possivel acessar a câmera. ')
        cassia.runAndWait()
        return None

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mostra a imagem da webcam em uma janela chamada "Captura de Imagem"
        cv2.imshow('Captura de Imagem', frame)

        # Verifica se o tempo limite de 20 segundos foi atingido
        elapsed_time = time.time() - start_time
        if elapsed_time > 10:
            print("CÁSSIA: Tempo limite de 10 segundos atingido. Capturando a imagem.")
            cassia.say("Tempo limite de 10 segundos atingido. Capturando a imagem.")
            cassia.runAndWait()
            break

        # Fecha a visualização ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Captura e salva a última imagem mostrada
    if ret:
        imagem_capturada = 'imagem_capturada.jpg'
        cv2.imwrite(imagem_capturada, frame)
        cassia.say('Imagem capturada com sucesso.')
        print('CÁSSIA: Imagem capturada com sucesso.')
    else:
        cassia.say('Erro ao capturar a imagem.')
        print('CÁSSIA: Erro ao capturar a imagem.')
        imagem_capturada = None

        # Libera a captura e fecha a janela
    cap.release()
    cv2.destroyAllWindows()
    cassia.runAndWait()
    return imagem_capturada

#APÓS A FUNÇÃO captura_imagem TUDO QUE FOR POSSIVEL LER, SERÁ LIDO
def extrair_texto_imagem(imagem_path):
    try:
        texto = pytesseract.image_to_string(imagem_path, lang='eng') #
        return texto
    except Exception as e:
        cassia.say(f"Erro ao realizar OCR: {e} ")
        print(f"CÁSSIA: Erro ao realizar OCR: {e} ")
        cassia.runAndWait()
        return None

#FUNÇÃO QUE CHAMA O LLAMA E DEIXA A DISPOSIÇÃO
def process_llama_query(pergunta):
    input_json = {
        'model': 'llama3.1',
        'prompt': f'responda sucitamente em português em poucas palavras em um parágrafo: {pergunta}'
    }
    response = requests.post(url, json=input_json)

    if response.status_code == 200:
        linhas = response.text.strip().split('\n')
        valores_response = []

        for linha in linhas:
            obj = json.loads(linha)
            resposta = obj.get('response', '')
            valores_response.append(resposta)

        return ''.join(valores_response)
    else:
        return "Erro ao processar a consulta."

#FUNÇÃO COM API DO CLIMA, CHAMA E DEIXA A DISPOSIÇÃO
def clima_atual(cidade):
    API_KEY = "12e89494e8d6d12e0b02689a27e65814"
    link = f"https://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={API_KEY}&lang=pt_br"
    requisicao = requests.get(link)
    requisicao_dic = requisicao.json()

    descricao = requisicao_dic['weather'][0]['description']
    temperatura = requisicao_dic['main']['temp'] - 273.15  # Convertendo de Kelvin para Celsius
    return descricao, temperatura

#FUNÇÃO COM API DA COTAÇÃO ATUAL, CHAMA E DEIXA A DISPOSIÇÃO
def cotacao(moeda):
    url = "https://economia.awesomeapi.com.br/last/USD-BRL,EUR-BRL,BTC-BRL"
    response = requests.get(url)

    if response.status_code == 200:
        cotacao_atual = float(response.json()[moeda]['bid'])
        reais = int(cotacao_atual)
        centavos = int(round((cotacao_atual - reais) * 100))
        return reais, centavos
    else:
        return None, None

#SIMPLES FUNÇÃO COM A BIBLIOTECA DATETIME PARA REVELAR O HORARIO ATUAL
def hora_data_atual():
    agora = datetime.datetime.now()
    horas = agora.hour
    minutos = agora.minute
    return horas, minutos

#SIMPLES FUNÇÃO COM A BIBLIOTECA DATETIME PARA REVELAR O DIA ATUAL
def data_atual():
    agora = datetime.datetime.now()
    dia = agora.day
    mes = agora.month
    ano = agora.year
    meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho",
             "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
    return dia, meses[mes - 1], ano

#AQUI O BIXO PEGA, O processing_command É RESPONSAVEL POR FAZER TODAS AS FUNÇÕES DECLARADAS A CIMA FUNCIONAR.
def processing_command():
    res = True
    while res:
        print("CÁSSIA: Iniciando...")
        comando = recognize_command()

        if comando in ['ok cássia', 'okcássia', 'ok cássias', 'ok, cássia',
                       'ok cassia', 'okcassia', 'ok cassias', 'ok cassia']:
            cassia.say('Sim, o que posso fazer ?')
            print('CÁSSIA: Sim, o que posso fazer ?')
            cassia.runAndWait()

            while res:
                comando = recognize_command()

                if comando:
                    if 'cadastrar evento' in comando:
                        cassia.say('Ok, qual evento devo cadastrar ?')
                        print('CÁSSIA: Ok, qual evento devo cadastrar ?')
                        cassia.runAndWait()
                        evento = recognize_command()
                        if evento:
                            with open('agenda.txt', 'a', encoding='utf-8') as arquivo:
                                arquivo.write(evento + '\n')
                            cassia.say('Evento cadastrado com sucesso. ')
                            print('CÁSSIA: Evento cadastrado com sucesso')
                            cassia.runAndWait()
                        pass

                    elif 'minha agenda' in comando:
                        try:
                            with open('agenda.txt', 'r', encoding='utf-8') as arquivo:
                                agenda = arquivo.read()
                            cassia.say('Aqui estão os eventos cadastrados: ')
                            print('CÁSSIA: Aqui estão os eventos cadastrados: ')
                            cassia.say(agenda)
                            print(agenda)
                            cassia.runAndWait()
                        except FileNotFoundError:
                            cassia.say('A agenda está vazia ou não foi encontrada. ')
                            print('CÁSSIA: A agenda está vazia ou não foi encontrada. ')
                            cassia.runAndWait()
                        pass

                    elif 'remover evento' in comando:
                        cassia.say('Qual evento você gostaria de remover?')
                        print('CÁSSIA: Qual evento você gostaria de remover?')
                        cassia.runAndWait()
                        evento_a_remover = recognize_command()
                        if evento_a_remover:
                            try:
                                with open('agenda.txt', 'r', encoding='utf-8') as arquivo:
                                    eventos = arquivo.readlines()

                                with open('agenda.txt', 'w', encoding='utf-8') as arquivo:
                                    evento_encontrado = False
                                    for evento in eventos:
                                        if evento.strip().lower() != evento_a_remover.lower():
                                            arquivo.write(evento)
                                        else:
                                            evento_encontrado = True

                                if evento_encontrado:
                                    cassia.say(f'O evento "{evento_a_remover}" foi removido com sucesso.')
                                    print(f'CÁSSIA: O evento "{evento_a_remover}" foi removido com sucesso.')
                                else:
                                    cassia.say(f'O evento "{evento_a_remover}" não foi encontrado na agenda.')
                                    print(f'CÁSSIA: O evento "{evento_a_remover}" não foi encontrado na agenda.')

                                cassia.runAndWait()
                            except FileNotFoundError:
                                cassia.say('A agenda está vazia ou não foi encontrada.')
                                print('CÁSSIA: A agenda está vazia ou não foi encontrada.')
                                cassia.runAndWait()
                        pass

                    elif 'detectar objeto' in comando:
                        detectar_objetos()

                        pass

                    elif 'ler texto' in comando or 'leia o texto' in comando:
                        cassia.say('Fazendo a captura...')
                        print('CÁSSIA: Fazendo a captura...')
                        cassia.runAndWait()

                        imagem_path = captura_imagem()

                        if imagem_path:
                            cassia.say('Lendo o texto da imagem...')
                            print('CÁSSIA: Lendo o texto da imagem...')
                            cassia.runAndWait()

                            texto = extrair_texto_imagem(imagem_path)
                            if texto:
                                cassia.say(f"O texto da imagem é: {texto}")
                                print(f"CÁSSIA: O texto da imagem é: {texto}")
                                cassia.runAndWait()
                            else:
                                cassia.say('Não foi possivel extrair o texto da imagem. ')
                                print('CÁSSIA: Não foi possivel extrair o texto da imagem. ')
                                cassia.runAndWait()
                        pass

                    elif 'pergunta' in comando:
                        cassia.say('O que você gostaria de saber ? ')
                        print('CÁSSIA: O que você gostaria de saber ? ')
                        cassia.runAndWait()
                        pergunta = recognize_command()
                        if pergunta:
                            resposta = process_llama_query(pergunta)
                            cassia.say(resposta)
                            print(resposta)
                            cassia.runAndWait()
                        pass

                    elif 'hora' in comando:
                        horas, minutos = hora_data_atual()
                        cassia.say(f"Agora são {horas} horas e {minutos} minutos.")
                        print(f"CÁSSIA: Agora são {horas} horas e {minutos} minutos.")
                        cassia.runAndWait()

                        pass

                    elif 'dia' in comando:
                        dia, mes, ano = data_atual()
                        cassia.say(f"Hoje é dia {dia} de {mes} de {ano}.")
                        print(f"CÁSSIA: Hoje é dia {dia} de {mes} de {ano}.")
                        cassia.runAndWait()

                        pass

                    elif 'clima' in comando:
                        cassia.say("De qual cidade você quer saber o clima?")
                        print("CÁSSIA: De qual cidade você quer saber o clima?")
                        cassia.runAndWait()
                        cidade = recognize_command()
                        if cidade:
                            descricao, temperatura = clima_atual(cidade)
                            cassia.say(f"O clima agora em {cidade} é {descricao}, com temperatura de {temperatura:.2f} graus Celsius.")
                            print(f"CÁSSIA: O clima agora em {cidade} é {descricao}, com temperatura de {temperatura:.2f}ºC.")
                            cassia.runAndWait()

                        pass

                    elif 'cotação' in comando:
                        cassia.say('Qual moeda você gostaria de consultar? (dólar, euro ou bitcoin)')
                        print('CÁSSIA: Qual moeda você gostaria de consultar? (dólar, euro ou bitcoin)')
                        cassia.runAndWait()
                        comando = recognize_command()
                        moeda = ''
                        if 'dólar' in comando:
                            moeda = "USDBRL"

                        elif 'euro' in comando:
                            moeda = "EURBRL"
                        elif 'bitcoin' in comando:
                            moeda = "BTCBRL"
                        if moeda:
                            reais, centavos = cotacao(moeda)

                            cassia.say(f"No momento, {moeda} está custando {reais} reais e {centavos} centavos.")
                            print(f"CÁSSIA: No momento, {moeda} está custando {reais} reais e {centavos} centavos.")
                            cassia.runAndWait()

                        pass

                    elif 'obrigado' in comando:
                        cassia.say("De nada! Até logo.")
                        print("CÁSSIA: De nada! Até logo.")
                        cassia.runAndWait()
                        res = True  #
        else:
            print("Diga, 'ok cássia'...")

#DEVE SER USADO PARA ENCAPSULAMENTO DO CÓDIGO (CASO O AMBIENTE SO FOR LINUX, NÃO SERÁ NECESSÁRIO)
if __name__ == "__main__":
    processing_command()

