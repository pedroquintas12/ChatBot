const express = require('express');
const bodyParser = require('body-parser');
const OpenAI = require('openai');
const mysql = require('mysql2');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
app.use(bodyParser.json());
app.use(express.static('public'));

const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: '123456',
  database: 'loja',
});

db.connect((err) => {
  if (err) throw err;
  console.log('Conectado ao banco de dados');
});

// Recupera a chave da API OpenAI do banco de dados
let openaiApiKey;

const getOpenAIKey = () => {
  return new Promise((resolve, reject) => {
    db.query('SELECT value FROM config WHERE chave = "openai_api_key"', (err, results) => {
      if (err) return reject(err);
      if (results.length > 0) {
        resolve(results[0].value);
      } else {
        reject(new Error('Chave da API OpenAI não encontrada no banco de dados.'));
      }
    });
  });
};

getOpenAIKey()
  .then(apiKey => {
    openaiApiKey = apiKey;
    const openai = new OpenAI({ apiKey });

    // Estado da conversa
    const userSessions = {};

    const isAskingAboutProducts = (question) => {
      const lowerCaseQuestion = question.toLowerCase();
      return lowerCaseQuestion.includes('produto') || 
             lowerCaseQuestion.includes('produtos') || 
             lowerCaseQuestion.includes('quais os produtos disponíveis');
    };

    // Endpoint para perguntas do chatbot
    app.post('/chat', async (req, res) => {
      const { userId, question } = req.body;

      // Inicializa a sessão do usuário se não existir
      if (!userSessions[userId]) {
        userSessions[userId] = { step: 'askCategory' };
      }

      const session = userSessions[userId];

      if (session.step === 'askCategory') {
        if (isAskingAboutProducts(question)) {
          db.query('SELECT DISTINCT categoria FROM produtos', (err, result) => {
            if (err) throw err;

            let categoryResponse = 'Por favor, escolha uma categoria:\n';
            result.forEach((row, index) => {
              categoryResponse += `${index + 1} - ${row.categoria}\n`;
            });

            categoryResponse += '\nEscolha um produto pelo ID:\n';

            session.categories = result.map(row => row.categoria);
            session.step = 'showProducts';

            res.json({ answer: categoryResponse });
          });
        } else {
          res.json({ answer: 'Não entendi. Se você quer saber sobre produtos, diga algo como "quais os produtos disponíveis?"' });
        }
      } else if (session.step === 'showProducts') {
        const categoryIndex = parseInt(question, 10) - 1;
        const category = session.categories[categoryIndex];

        if (category) {
          session.category = category;
          session.step = 'selectProduct';

          db.query('SELECT id, nome, preco FROM produtos WHERE categoria = ? LIMIT 5', [session.category], (err, result) => {
            if (err) throw err;

            let productResponse = `Aqui estão alguns produtos da categoria ${session.category}:\n`;
            result.forEach(product => {
              productResponse += `ID: ${product.id} - ${product.nome}: R$ ${product.preco}\n`;
            });

            productResponse += '\nEscolha um produto pelo ID:\n';

            res.json({ answer: productResponse });
          });
        } else {
          res.json({ answer: 'Categoria não encontrada. Por favor, escolha uma categoria válida.' });
        }
      } else if (session.step === 'selectProduct') {
        const productId = parseInt(question, 10);
        
        if (!isNaN(productId)) {
          // Verifica se o produto ID está na categoria selecionada
          db.query('SELECT id, nome, descricao FROM produtos WHERE id = ? AND categoria = ?', [productId, session.category], (err, result) => {
            if (err) throw err;

            if (result.length > 0) {
              const product = result[0];
              res.json({ answer: `Descrição do produto "${product.nome}":\n${product.descricao}` });
            } else {
              res.json({ answer: 'Produto não encontrado na categoria selecionada. Por favor, forneça um ID válido dentro da categoria escolhida.' });
            }
          });
        } else {
          res.json({ answer: 'Por favor, forneça um ID de produto válido.' });
        }
      } else if (session.step === 'awaitingNextQuestion') {
        if (isAskingAboutProducts(question)) {
          db.query('SELECT id, nome, preco FROM produtos WHERE categoria = ? LIMIT 5', [session.category], (err, result) => {
            if (err) throw err;

            let productResponse = `Aqui estão alguns produtos da categoria ${session.category}:\n`;
            result.forEach(product => {
              productResponse += `ID: ${product.id} - ${product.nome}: R$ ${product.preco}\n`;
            });

            res.json({ answer: productResponse + '\nEscolha um produto pelo ID:\n' });
          });
        } else {
          try {
            const completion = await openai.chat.completions.create({
              model: "gpt-3.5-turbo", 
              messages: [{ role: "user", content: question }],
            });
            res.json({ answer: completion.choices[0].message.content });
          } catch (error) {
            res.status(500).json({ error: 'Erro ao acessar a API do OpenAI' });
          }
        }
      }
    });

    // Servir a interface HTML
    app.get('/', (req, res) => {
      res.sendFile(__dirname + '/public/index.html');
    });

    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
      console.log(`Servidor rodando na porta ${PORT}`);
    });
  })
  .catch(err => {
    console.error('Erro ao recuperar a chave da API OpenAI:', err.message);
    process.exit(1);
  });
