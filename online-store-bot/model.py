import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Inicialização do spaCy para pré-processamento de texto
nlp = spacy.load("pt_core_news_sm")

# Dataset de perguntas e respostas sobre manutenção de motos
faq = [
    ("Quais os dias de funcionamento?", "De Segunda a sexta."),
    ("Qual o prazo de entrega?", "O prazo de entrega depende da sua localização, mas geralmente é de 3 a 7 dias úteis."),
    ("Como posso pagar?", "Você pode pagar com cartão de crédito, boleto bancário ou PayPal."),
    ("Quais produtos você vende?", "Vendemos uma variedade de produtos eletrônicos, roupas, e acessórios."),
    ("Qual é o horário de atendimento?", "Nosso horário de atendimento é de segunda a sexta-feira, das 9h às 18h."),
    ("Onde fica a loja física?", "Nossa loja física está localizada na Rua X, número 123, Centro."),
    ("Como ajustar a suspensão da moto?", "A suspensão deve ser ajustada conforme o peso do piloto e o tipo de terreno. Isso pode ser feito no garfo e no amortecedor traseiro."),
    ("Qual a importância de fazer a revisão de moto?", "A revisão garante a segurança do piloto, melhora o desempenho da moto e aumenta a durabilidade do veículo."),
    ("O que pode causar falha no sistema de alimentação da moto?", "Problemas no carburador, filtro de ar entupido ou bomba de combustível defeituosa podem causar falhas no sistema de alimentação."),
    ("Como limpar a parte elétrica da moto?", "A limpeza deve ser feita com cuidado, utilizando um pano seco e produtos adequados para não danificar os componentes elétricos."),
    ("O que fazer se minha moto estiver com o escapamento vazando?", "Verifique as juntas de vedação e a parte do escapamento danificada. Se necessário, substitua ou faça o reparo."),
    ("Como trocar as luzes da moto?", "A troca das luzes pode ser feita retirando a lâmpada antiga e substituindo por uma nova, de acordo com as especificações do fabricante."),
    ("Como substituir o cabo da embreagem da moto?", "A substituição envolve desconectar o cabo antigo e instalar um novo, ajustando a tensão conforme necessário."),
    ("O que é a regulagem da mistura de combustível?", "É o ajuste da quantidade de combustível e ar no carburador, importante para a eficiência do motor e o consumo de combustível."),
    ("Como fazer a manutenção preventiva em uma moto?", "A manutenção preventiva inclui a troca de óleo, verificação de pneus, ajuste da corrente e a inspeção do sistema de freios e elétrica."),
    ("Qual é a função do filtro de combustível?", "O filtro de combustível impede que impurezas do combustível cheguem ao motor, garantindo melhor desempenho e vida útil do motor."),
    ("Como ajustar os espelhos da moto?", "Os espelhos devem ser ajustados para garantir a melhor visibilidade, geralmente alinhando-os com a linha da moto."),
    ("Como fazer a manutenção do sistema de direção da moto?", "Verifique se há folgas ou ruídos na direção, especialmente no rolamento da coluna de direção."),
    ("O que é necessário para fazer a manutenção de uma moto de competição?", "Além da revisão periódica, é necessário ajustar a suspensão, o sistema de ignição e o carburador conforme as necessidades da competição."),
    ("Como faço a manutenção da corrente da moto?", "A corrente deve ser lubrificada regularmente e ajustada para garantir o funcionamento adequado da transmissão."),
    ("Como posso verificar a pressão dos pneus da moto?", "A pressão dos pneus pode ser verificada utilizando um manômetro específico para motocicletas."),
    ("O que causa o desgaste irregular dos pneus?", "O desgaste irregular pode ser causado por pressão inadequada, desalinhamento da suspensão ou condução incorreta."),
    ("Quando devo trocar o óleo da moto?", "O óleo da moto deve ser trocado de acordo com as especificações do fabricante ou a cada 3.000 a 5.000 km, dependendo do modelo."),
    ("Qual é a importância da calibragem dos pneus?", "Manter a calibragem correta dos pneus aumenta a segurança, melhora o desempenho e prolonga a vida útil dos pneus."),
    ("Como ajustar a tensão da corrente da moto?", "A tensão da corrente deve ser ajustada para que fique com uma folga adequada, nem muito frouxa nem muito apertada."),
    ("Como identificar um problema no sistema de freios?", "Ruídos, vibrações ou dificuldades para frear são sinais de que pode haver problemas no sistema de freios."),
    ("O que fazer se o motor da moto não pegar?", "Verifique o nível de combustível, a bateria e o sistema de ignição para identificar a causa do problema."),
    ("Como substituir o filtro de ar da moto?", "O filtro de ar pode ser substituído retirando a peça antiga e instalando a nova, garantindo que esteja bem encaixada."),
    ("Quando devo trocar a vela de ignição?", "A vela de ignição deve ser trocada regularmente, conforme recomendado pelo fabricante, geralmente a cada 10.000 a 20.000 km."),
    ("Como fazer a regulagem do carburador?", "A regulagem do carburador envolve ajustar a mistura de ar e combustível, de acordo com as necessidades do motor."),
    ("O que é o balanceamento da roda da moto?", "O balanceamento é feito para garantir que a roda gire de forma equilibrada, sem causar vibrações durante a condução."),
    ("Como manter a bateria da moto em bom estado?", "A bateria deve ser mantida limpa e com o nível de eletrólito adequado, além de ser recarregada regularmente."),
    ("Como saber se a suspensão da moto está com defeito?", "Se houver dificuldade em absorver impactos ou ruídos anormais, é possível que a suspensão precise de manutenção."),
    ("Como trocar o fluido de freio da moto?", "O fluido de freio deve ser trocado periodicamente, seguindo as especificações do fabricante para garantir a eficiência do sistema."),
    ("O que é o sistema de injeção eletrônica?", "É um sistema que controla a quantidade de combustível injetado no motor, melhorando a eficiência e o consumo de combustível."),
    ("Como faço para limpar o carburador da moto?", "O carburador deve ser retirado e limpo com produtos adequados para remover resíduos e impurezas acumuladas."),
    ("O que é a válvula de alívio da moto?", "A válvula de alívio controla a pressão interna do motor e evita danos causados por pressão excessiva no sistema."),
    ("Como identificar vazamentos de óleo na moto?", "Verifique se há manchas de óleo no chão ou no motor, especialmente ao redor do filtro de óleo e da junta do motor."),
    ("Como ajustar o jogo de válvulas?", "O ajuste das válvulas deve ser feito com a moto fria, verificando o clearance correto conforme as especificações do fabricante."),
    ("Como regular a embreagem?", "A embreagem deve ser ajustada para garantir que não haja folga excessiva, proporcionando um acionamento suave."),
    ("Como substituir o radiador de óleo?", "O radiador de óleo deve ser substituído se estiver danificado ou entupido, garantindo a lubrificação adequada do motor."),
    ("Como evitar o superaquecimento do motor?", "Verifique o sistema de arrefecimento, garantindo que o fluido de arrefecimento esteja no nível correto e sem impurezas."),
    ("O que fazer se a moto estiver com falha de ignição?", "Verifique a vela de ignição, o sistema de bobina e os cabos de ignição para identificar possíveis falhas."),
    ("Como fazer a regulagem do ponto de ignição?", "A regulagem do ponto de ignição é feita com ferramentas específicas para ajustar o momento de disparo da vela."),
    ("Quando devo trocar a pastilha de freio?", "As pastilhas de freio devem ser trocadas quando estiverem desgastadas até o limite indicado pelo fabricante."),
    ("O que é o sistema de lubrificação da moto?", "O sistema de lubrificação distribui óleo pelo motor, reduzindo o atrito e evitando o superaquecimento."),
    ("Como identificar um problema no sistema de transmissão?", "Ruídos anormais ou dificuldade para engatar marchas são sinais de que há algo errado na transmissão."),
    ("Como fazer o alinhamento das rodas?", "O alinhamento das rodas é feito ajustando os componentes da suspensão para garantir que as rodas fiquem paralelas entre si."),
    ("O que é o cabeçote da moto?", "O cabeçote é a parte superior do motor, responsável por controlar as válvulas e o fluxo de gases."),
    ("Como verificar a pressão da corrente?", "A pressão da corrente deve ser verificada verificando se há folga excessiva ou se está apertada demais."),
    ("Como fazer a manutenção do sistema de ignição?", "A manutenção do sistema de ignição envolve verificar a bobina, a vela e o cabo de ignição regularmente."),
    ("O que é o sistema de escape da moto?", "O sistema de escape dirige os gases de escape do motor para fora, ajudando a controlar a emissão e o desempenho."),
    ("Como substituir a bomba de combustível?", "A bomba de combustível deve ser substituída se estiver apresentando falhas no fornecimento de combustível ao motor."),
    ("O que é o sistema de refrigeração da moto?", "O sistema de refrigeração mantém a temperatura do motor dentro da faixa ideal para evitar o superaquecimento."),
    ("Como ajustar o carburador da moto?", "O carburador deve ser ajustado para garantir uma mistura de combustível e ar adequada ao desempenho do motor."),
    ("quanto tempo demora uma revisão?", "O Prazo de entrega varia de acordo com a moto, mas em geral demora de 1 dia a 1 dia e mio")

]

# Pre-processamento das perguntas
def preprocess(text):
    return ' '.join([token.text.lower() for token in nlp(text) if not token.is_stop and not token.is_punct])

# Separar perguntas e respostas
perguntas = [p[0] for p in faq]
respostas = [p[1] for p in faq]

# Pré-processar as perguntas
perguntas_processadas = [preprocess(p) for p in perguntas]

# Criando o modelo de Machine Learning
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Treinando o modelo
model.fit(perguntas_processadas, respostas)

# Função para responder à pergunta
def responder_pergunta(pergunta):
    pergunta_processada = preprocess(pergunta)
    # Usando o modelo para prever a resposta mais adequada
    return model.predict([pergunta_processada])[0]

# Função para encontrar a pergunta mais similar
def encontrar_resposta_mais_similar(pergunta_usuario, limiar_similaridade=0.5, resposta_padrao="Desculpe, não entendi a sua pergunta. Pode reformular?"):
    pergunta_usuario_processada = preprocess(pergunta_usuario)
    # Transformando as perguntas processadas para a mesma representação numérica
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(perguntas_processadas + [pergunta_usuario_processada])
    
    # Calculando a similaridade entre a pergunta do usuário e todas as perguntas do FAQ
    similaridades = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    
    # Encontrar a pergunta mais similar
    indice_pergunta_similar = np.argmax(similaridades)
    similaridade_maxima = similaridades[0][indice_pergunta_similar]
    
    # Se a similaridade for abaixo do limiar, retorna a resposta padrão
    if similaridade_maxima < limiar_similaridade:
        return resposta_padrao
    
    return respostas[indice_pergunta_similar]