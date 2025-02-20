Roblox AI Trainer
==================

# 🤖 Roblox AI Controller 🎮

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.4+](https://img.shields.io/badge/TensorFlow-2.4%2B-orange.svg)](https://www.tensorflow.org/install)
[![OpenCV 4.5+](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://pypi.org/project/opencv-python/)
[![PyAutoGUI 0.9+](https://img.shields.io/badge/PyAutoGUI-0.9%2B-yellow.svg)](https://pypi.org/project/PyAutoGUI/)
[![PyWin32 300+](https://img.shields.io/badge/PyWin32-300%2B-blue.svg)](https://pypi.org/project/pywin32/)
[![Keyboard 0.13+](https://img.shields.io/badge/Keyboard-0.13%2B-orange.svg)](https://pypi.org/project/keyboard/)
[![Mouse 0.7+](https://img.shields.io/badge/Mouse-0.7%2B-green.svg)](https://pypi.org/project/mouse/)
[![Matplotlib 3.3+](https://img.shields.io/badge/Matplotlib-3.3%2B-yellow.svg)](https://pypi.org/project/matplotlib/)
[![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)](https://github.com/seuusuario/roblox-ai-controller)

**Uma IA "avançada" para controle automático de personagens no Roblox com aprendizado profundo e recuperação de colisões**

![Interface Preview](screenshots/preview.png)

## 📑 Descrição

Este projeto é um controlador AI inteligente para Roblox que utiliza redes neurais convolucionais (CNN) para aprender e replicar comportamentos de jogo. Com recursos como detecção de colisões, sistema de recuperação automática e interface visual intuitiva, ele permite:

- 🧠 Treinamento do modelo em tempo real
- 🕹️ Controle automático do personagem
- 🚨 Sistema anti-travamento
- 📊 Monitoramento em tempo real das ações
- 🖥️ Visualização dupla da tela e previsões

## ✨ Funcionalidades Principais

- **Detecção Inteligente de Colisões**
  - Sistema de recuperação com sequência personalizável
  - Escape automático de loops de colisão
- **Modos de Operação**
  - 🏋️ Modo Treino (coleta dados e treina o modelo)
  - 🤖 Modo Predict (controle automático do jogo)
- **Interface Amigável**
  - Tema escuro moderno
  - Visualização em tempo real do jogo
  - Gráfico interativo de previsões
- **Controles Avançados**
  - 🚨 Parada de emergência (Ctrl+Shift+L)
  - Suporte a múltiplos dispositivos de entrada
  - Smoothing de movimentos do mouse
- **Gestão de Modelos**
  - Carregar/Salvar modelos treinados
  - Monitoramento do treino com gráficos
  - Auto-salvamento periódico

## Exemplo de plotagem de previsões 📊📈

![Interface Preview](screenshots/plotexample.png)

## 📋 Pré-requisitos

- Python 3.8 ou superior
- Janela do Roblox em execução (com título "Roblox")
- Resolução mínima de tela: 1280x720
- Espaço em disco: ~5GB (para modelos e datasets)

As dependências podem ser instaladas com:

```bash
pip install tensorflow opencv-python numpy pyautogui pywin32 keyboard mouse matplotlib
```

## Configuração do Ambiente ⚙️💻🔧

O projeto faz algumas configurações iniciais para garantir um melhor desempenho:

- Suprime mensagens desnecessárias do TensorFlow.
- Configura a GPU para utilizar alocação dinâmica de memória.
- Define limites de *threads* para processamento paralelo.

## Estrutura do Projeto 📂📌📋

- **`RobloxAITrainer`**: Classe principal que gerencia a captura de tela, treinamento e inferência.
- **`init_model`**: Inicializa a rede neural convolucional e carrega pesos de um modelo salvo.
- **`setup_gui`**: Configura a interface gráfica para controle do treinamento.
- **`capture_screen`**: Captura a tela da janela do Roblox e processa a imagem.
- **`get_current_actions`**: Obtém o estado atual das teclas e do mouse.
- **`training_worker`**: Gerencia o processo de treinamento da IA.
- **`capture_and_preview_loop`**: Exibe visualmente a captura de tela e acumula dados para treinamento.

## Treinamento 🏋️📈🧠

O modelo recebe capturas de tela da janela do Roblox e as associa às ações do jogador (teclas pressionadas e movimentação do mouse). O treinamento é realizado em *batches* e os resultados são armazenados para análise.

Para iniciar o treinamento, execute o script principal e utilize a interface para configurar os *epochs* e iniciar a captura de dados.

## Inferência (Predição) 🔍🎯🤔

Após o treinamento, o modelo pode ser utilizado para prever as próximas ações do jogador, baseado na imagem da tela.

## Salvamento do Modelo 💾📂🔄

O modelo treinado é salvo automaticamente no diretório `models/` e pode ser recarregado para uso futuro.

## Encerramento e Limpeza 🛑🚮✅

- O programa registra um *hook* para limpar os recursos ao sair.
- Todas as janelas *OpenCV* abertas são fechadas ao encerrar.
- Os arquivos de log de treinamento são salvos automaticamente para revisão posterior.

## Controles Rápidos ⌨️🎮🖱️

- Pressionar `ctrl+shift+l` interrompe todas as atividades e fecha o programa.
- O botão "Stop" na interface também interrompe o treinamento ou a predição.
