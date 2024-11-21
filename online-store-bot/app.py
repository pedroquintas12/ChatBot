from flask import Flask, request, jsonify
from flask_cors import CORS 
from model import encontrar_resposta_mais_similar 

app = Flask(__name__)
CORS(app)
# Função de exemplo para processar a mensagem do usuário
def process_message(user_input):
    # Aqui você pode usar qualquer lógica para processar a entrada
    # Vamos responder com uma resposta simples por enquanto
    return f"Você disse: {user_input}"

@app.route('/get_response', methods=['POST'])
def get_responser():
    # Obtendo a mensagem enviada no corpo da requisição
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400  # Responde com erro 400 caso a mensagem não seja fornecida

    user_input = data['message']
    
    # Processa a entrada do usuário
    response = encontrar_resposta_mais_similar(user_input)
    
    # Retorna a resposta em formato JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
