Roblox AI Trainer
==================

## Descrição
O **Roblox AI Trainer** é um sistema de treinamento e predição baseado em redes neurais, projetado para capturar e interpretar entradas do jogador no jogo *Roblox*. Utiliza *TensorFlow* para criar e treinar um modelo de *deep learning* que aprende a partir das ações do jogador.

## Dependências
Este projeto requer as seguintes bibliotecas:

- Python 3
- TensorFlow
- OpenCV
- NumPy
- PyAutoGUI
- Win32API (pywin32)
- Keyboard
- Mouse
- Tkinter
- Matplotlib

As dependências podem ser instaladas com:
```bash
pip install tensorflow opencv-python numpy pyautogui pywin32 keyboard mouse matplotlib
```

## Configuração do Ambiente
O projeto faz algumas configurações iniciais para garantir um melhor desempenho:
- Suprime mensagens desnecessárias do TensorFlow.
- Configura a GPU para utilizar alocação dinâmica de memória.
- Define limites de *threads* para processamento paralelo.

## Estrutura do Projeto
- **`RobloxAITrainer`**: Classe principal que gerencia a captura de tela, treinamento e inferência.
- **`init_model`**: Inicializa a rede neural convolucional e carrega pesos de um modelo salvo.
- **`setup_gui`**: Configura a interface gráfica para controle do treinamento.
- **`capture_screen`**: Captura a tela da janela do Roblox e processa a imagem.
- **`get_current_actions`**: Obtém o estado atual das teclas e do mouse.
- **`training_worker`**: Gerencia o processo de treinamento da IA.
- **`capture_and_preview_loop`**: Exibe visualmente a captura de tela e acumula dados para treinamento.

## Treinamento
O modelo recebe capturas de tela da janela do Roblox e as associa às ações do jogador (teclas pressionadas e movimentação do mouse). O treinamento é realizado em *batches* e os resultados são armazenados para análise.

Para iniciar o treinamento, execute o script principal e utilize a interface para configurar os *epochs* e iniciar a captura de dados.

## Inferência (Predição)
Após o treinamento, o modelo pode ser utilizado para prever as próximas ações do jogador, baseado na imagem da tela.

## Salvamento do Modelo
O modelo treinado é salvo automaticamente no diretório `models/` e pode ser recarregado para uso futuro.

## Encerramento e Limpeza
- O programa registra um *hook* para limpar os recursos ao sair.
- Todas as janelas *OpenCV* abertas são fechadas ao encerrar.
- Os arquivos de log de treinamento são salvos automaticamente para revisão posterior.

## Controles Rápidos
- Pressionar `L` interrompe todas as atividades e fecha o programa.
- O botão "Stop" na interface também interrompe o treinamento ou a predição.

## Contribuição
Sinta-se à vontade para modificar o código, testar novos hiperparâmetros e compartilhar melhorias.

## Licença
Este projeto é de uso livre e pode ser modificado conforme necessário.

