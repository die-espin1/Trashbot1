# Importar el módulo openai y json
import openai
import json

# Clase para utilizar cualquier LLM para procesar un texto
# y regresar una función a llamar con sus parámetros
# Utiliza el modelo 0613, pero puedes cambiarlo según tus necesidades
class LLM():
    def __init__(self):
        pass
    
    # Método para procesar funciones en el texto y determinar si se debe llamar a alguna función
    def process_functions(self, text):
        
        # Realizar una solicitud a la API de ChatGPT con funciones predefinidas
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                # Descripción del asistente
                {"role": "system", "content": "Eres un asistente malhablado"},
                # Mensaje del usuario (input de texto)
                {"role": "user", "content": text},
            ], functions=[
                # Definición de funciones disponibles
                {
                    "name": "get_weather",
                    "description": "Obtener el clima actual",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ubicacion": {
                                "type": "string",
                                "description": "La ubicación, debe ser una ciudad",
                            }
                        },
                        "required": ["ubicacion"],
                    },
                },
                # Otras funciones similares...
            ],
            function_call="auto",  # Configuración para llamar automáticamente a las funciones
        )
        
        message = response["choices"][0]["message"]
        
        # Verificar si la respuesta incluye una llamada a función
        if message.get("function_call"):
            # Sí, obtener información sobre la función
            function_name = message["function_call"]["name"]  # Nombre de la función
            args = message.to_dict()['function_call']['arguments']  # Argumentos de la función en formato JSON
            args = json.loads(args)  # Convertir argumentos a un diccionario de Python
            print("Función a llamar: " + function_name)
            return function_name, args, message
        
        return None, None, message
    
    # Método para procesar la respuesta de una función y obtener una respuesta en lenguaje natural
    def process_response(self, text, message, function_name, function_response):
        # Realizar una solicitud a la API de ChatGPT incluyendo la respuesta de la función
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                # Descripción del asistente
                {"role": "system", "content": "Eres un asistente malhablado"},
                # Mensaje original del usuario
                {"role": "user", "content": text},
                # Mensaje de la función llamada y su respuesta
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        # Obtener y devolver la respuesta en lenguaje natural
        return response["choices"][0]["message"]["content"]
