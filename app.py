# Importamos librerias
from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import openai
from dotenv import load_dotenv
from transcriber import Transcriber
from llm import LLM
from weather import Weather
from tts import TTS

# Cargar las llaves del archivo .env
load_dotenv()
openai.api_key = "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"
elevenlabs_key = "cfdd1251dd6daecf75c8cc08cd020f82"

# Creamos nuestra funcion de dibujo
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos la malla facial
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=4)

# Guardamos en variables los pesos, la arquitectura y las clases del modelo previamente entrenado
model = 'frozen_inference_graph_V2.pb'
config = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
clases = 'coco_labels.txt'

# Extraemos las etiquetas del archivo
with open(clases) as cl:
    labels = cl.read().split("\n")
print(labels)

# Leemos el modelo con TensorFlow
net = cv2.dnn.readNetFromTensorflow(model, config)

# Creamos la app
app = Flask(__name__)

# Creamos una función de detección de objetos
def object_detect(net, img):
    # Dimensiones
    dim = 300

    # Preprocesamos nuestra imagen
    blob = cv2.dnn.blobFromImage(img, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)

    # Pasamos nuestra imagen preprocesada a la red
    net.setInput(blob)

    # Extraemos los objetos detectados
    objetos = net.forward()

    return objetos

# Creamos una función de mostrar texto
def text(img, text, x, y):
    # Extraemos el tamaño del texto
    sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    # Extraemos el tamaño
    dim = sizetext[0]
    baseline = sizetext[1]

    # Creamos un rectangulo negro con el tamaño apropiado
    cv2.rectangle(img, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)
    # Mostramos el texto
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

# Creamos una función de mostrar objetos
def dibujar_objetos(img, objects, umbral=0.5):

    # Extraemos info
    filas = img.shape[0]
    colum = img.shape[1]

    # Para los objetos detectados
    for i in range(objects.shape[2]):
        # Buscamos su clase y confianza de detección
        clase = int(objects[0, 0, i, 1])
        puntaje = float(objects[0, 0, i, 2])

        # Extraemos sus coordenadas y las normalizamos a pixeles
        x = int(objects[0, 0, i, 3] * colum)
        y = int(objects[0, 0, i, 4] * filas)
        w = int(objects[0, 0, i, 5] * colum - x)
        h = int(objects[0, 0, i, 6] * filas - x)

        # Revisamos si superamos el umbral
        if puntaje > umbral:
            # Mostramos la clase
            text(img, "{}".format(labels[clase]), x, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Creamos la Video Captura
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Creamos una función para procesar el frame
def process_frame(frame):
    # Realizamos las detecciones de la malla facial
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = MallaFacial.process(frameRGB)

    # Si tenemos rostros
    if resultados.multi_face_landmarks:
        # Iteramos
        for rostros in resultados.multi_face_landmarks:
            # Dibujamos
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

    # Realizamos las detecciones de objetos
    detect = object_detect(net, frame)
    # Mostramos las detecciones
    dibujar_objetos(frame, detect)

    return frame

# Mostramos el video en RT
def gen_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)

        # Codificamos nuestro video en Bytes
        suc, encode = cv2.imencode('.jpg', frame)
        frame = encode.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta de aplicación 'principal'
@app.route('/')
def index():
    return render_template('index.html')

# Ruta del video
@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Definir la ruta "/audio" para manejar las solicitudes POST
@app.route("/audio", methods=["POST"])
def audio():
    # Obtener el archivo de audio de la solicitud
    audio = request.files.get("audio")
    
    # Transcribir el audio a texto
    text = Transcriber().transcribe(audio)
    print(f"Transcripción del audio: {text}")

    # Consultas a chatGPT
    prompt_usuario = text

    # Obtener la respuesta de ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente robótico que contesta en español con un máximo de 150 caracteres, respondiendo explícitamente lo que te preguntaron en cuestión del qué, cómo, cuándo y dónde. Eres servicial y NUNCA RESPONDES DE QUE NO PUEDES HACER ALGO solo porque eres un robot, das una respuesta diplomática y q la persona sienta que habala con un humano, es último es tu propósito. Cuando tienes que decir o contestar un número NUNCA lo dices poniendo el número en cifras como un 4, sino que lo escribes cuatro"},
            {"role": "user", "content": prompt_usuario},
        ]
    )

    respuesta_chatgpt = response.choices[0].message["content"]
    print(f"Respuesta de ChatGPT: {respuesta_chatgpt}")

    # Puedes procesar la respuesta de ChatGPT según tus necesidades
    llm = LLM()
    function_name, args, message = llm.process_functions(respuesta_chatgpt)

    if function_name is not None:
        # Aquí puedes manejar funciones específicas
        if function_name == "get_weather":
            # Generar archivo de audio utilizando TTS con la respuesta de ChatGPT
            tts_file = TTS().process(respuesta_chatgpt)
    else:
        # Respuesta por defecto
        # Generar archivo de audio utilizando TTS con la respuesta de ChatGPT
        tts_file = TTS().process(respuesta_chatgpt)
        return {"result": "ok", "text": respuesta_chatgpt, "file": tts_file}

# Ejecutar la aplicación si este script es el principal
if __name__ == "__main__":
    app.run(debug=True)
