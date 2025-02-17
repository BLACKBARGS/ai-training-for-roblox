# Imports da biblioteca padrão
import sys
import os
import time
import threading
import atexit
from threading import Thread

# Imports de terceiros
import cv2
import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow import keras
import win32gui
import win32con
import keyboard
import mouse
import tkinter as tk

# Configuração do Matplotlib (deve ser feita antes de importar pyplot)
import matplotlib
matplotlib.use('Agg')  # Usa backend não interativo para evitar problemas em threads
import matplotlib.pyplot as plt


# Suprime mensagens desnecessárias do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurações da GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_gpus = tf.config.list_physical_devices('GPU')
if physical_gpus:
    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Limita o número de threads usados pelo TensorFlow
tf.config.threading.set_intra_op_parallelism_threads(3)
tf.config.threading.set_inter_op_parallelism_threads(3)


class RobloxAITrainer:
    def __init__(self):
        self.threads = []
        self.roblox_hwnd = None
        self.stop_event = threading.Event()
        self.windows_to_close = []
        self.batch_size = 32

        # Dicionário com as ações
        self.action_keys = {
            'w': 0, 's': 1, 'a': 2, 'd': 3,
            'space': 4, 'shift': 5, 'e': 6, 'q': 7,
            'r': 8, 'f': 9, 'mouse_left': 10, 'mouse_right': 11,
        }

        # Buffer para treinamento e lock para acesso concorrente
        self.training_buffer = []
        self.buffer_lock = threading.Lock()
        self.training_history = {}  # Armazena métricas (loss, accuracy, etc.)

        # Cria pastas para salvar o modelo e os gráficos, se não existirem
        os.makedirs("models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        self.init_model()
        self.setup_gui()
        keyboard.add_hotkey('l', self.stop_all)
        atexit.register(self.cleanup_resources)

    def init_model(self):
        """Inicializa e compila o modelo. Carrega pesos, se disponíveis."""
        self.model = keras.Sequential([
            keras.layers.Input(shape=(240, 320, 1)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(len(self.action_keys) + 2, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Atualize o nome do arquivo para terminar com .weights.h5
        model_path = os.path.join("models", "roblox_ai_model.weights.h5")
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                print("Modelo carregado com sucesso!")
            except Exception as e:
                print(f"Não foi possível carregar o modelo: {e}")

    def setup_gui(self):
        """Configura a interface gráfica."""
        self.root = tk.Tk()
        self.root.title("Roblox AI Trainer")
        self.root.geometry("400x300")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_label = tk.Label(self.root, text="Status: Idle")
        self.status_label.pack(pady=10)

        epoch_frame = tk.Frame(self.root)
        epoch_frame.pack(pady=5)
        tk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epoch_var = tk.StringVar(value="1")
        tk.Entry(epoch_frame, textvariable=self.epoch_var, width=5).pack(side=tk.LEFT)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Start Prediction", command=self.start_prediction).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Stop", command=self.stop_all).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Model", command=self.save_model).pack(side=tk.LEFT, padx=5)

    def setup_cv_window(self, window_name):
        """Cria e configura uma janela OpenCV."""
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 320, 240)
        self.windows_to_close.append(window_name)

    def cleanup_windows(self):
        """Fecha todas as janelas OpenCV abertas."""
        for window in self.windows_to_close:
            cv2.destroyWindow(window)
        self.windows_to_close.clear()

    def capture_screen(self):
        """Captura a tela da janela do Roblox e retorna uma imagem em escala de cinza normalizada."""
        try:
            self.roblox_hwnd = win32gui.FindWindow(None, "Roblox")
            if not self.roblox_hwnd:
                print("Janela do Roblox não encontrada.")
                return None

            if win32gui.IsIconic(self.roblox_hwnd):
                win32gui.ShowWindow(self.roblox_hwnd, win32con.SW_RESTORE)

            rect = win32gui.GetWindowRect(self.roblox_hwnd)
            if (rect[2] - rect[0] <= 0) or (rect[3] - rect[1] <= 0):
                print("Dimensões inválidas da janela.")
                return None

            screenshot = pyautogui.screenshot(region=(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]))
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (320, 240))
            return frame.astype(np.float32) / 255.0

        except Exception as e:
            print(f"Erro na captura da tela: {e}")
            return None

    def get_current_actions(self):
        """Retorna um vetor de ações atuais (teclas e posição do mouse)."""
        actions = np.zeros(len(self.action_keys) + 2)
        for key, index in self.action_keys.items():
            if key.startswith('mouse'):
                actions[index] = float(mouse.is_pressed(key.split('_')[1]))
            else:
                actions[index] = float(keyboard.is_pressed(key))
        try:
            rect = win32gui.GetWindowRect(self.roblox_hwnd)
            window_width, window_height = rect[2] - rect[0], rect[3] - rect[1]
            mouse_x, mouse_y = pyautogui.position()
            actions[-2] = np.clip((mouse_x - rect[0]) / window_width, 0, 1)
            actions[-1] = np.clip((mouse_y - rect[1]) / window_height, 0, 1)
        except Exception:
            actions[-2:] = 0.5  # Posição central padrão se ocorrer erro
        return actions

    # --- MODO TREINAMENTO ---
    def capture_and_preview_loop(self):
        """Captura a tela, atualiza o preview e acumula amostras para treinamento."""
        self.status_label.config(text="Status: Training")
        print("Treinamento iniciado...")
        window_name = "Training View"
        self.setup_cv_window(window_name)
        while not self.stop_event.is_set():
            frame = self.capture_screen()
            if frame is not None:
                with self.buffer_lock:
                    self.training_buffer.append((frame, self.get_current_actions()))
                # Atualiza preview
                display_frame = (frame * 255).astype(np.uint8)
                actions = self.get_current_actions()
                active_actions = [key for key, idx in self.action_keys.items() if actions[idx] > 0]
                cv2.putText(display_frame, f"Actions: {', '.join(active_actions)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
                cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup_windows()
        self.status_label.config(text="Status: Idle")

    def training_worker(self):
        """Realiza o treinamento a partir dos dados acumulados e interrompe quando o número de epochs for atingido."""
        self.current_total_epochs = 0
        max_total_epochs = int(self.epoch_var.get())
        while not self.stop_event.is_set():
            # Se atingiu o número total de epochs, encerra o treinamento
            if self.current_total_epochs >= max_total_epochs:
                print("Número total de epochs atingido. Encerrando treinamento.")
                self.stop_all()  # Cuidado: evitar join da thread atual dentro do stop_all
                break

            batch_data = None
            with self.buffer_lock:
                if len(self.training_buffer) >= self.batch_size * 5:
                    batch_data = self.training_buffer.copy()
                    self.training_buffer.clear()

            if batch_data:
                X = np.array([item[0] for item in batch_data])
                y = np.array([item[1] for item in batch_data])
                print(f"Treinando com {len(batch_data)} amostras - Epoch {self.current_total_epochs + 1}/{max_total_epochs}")
                history = self.model.fit(X, y, epochs=1, batch_size=self.batch_size, verbose=1)
                self.current_total_epochs += 1
                self.update_training_history(history.history)

            time.sleep(0.1)

    def update_training_history(self, history_dict):
        """Acumula e plota as métricas de treinamento."""
        for key, values in history_dict.items():
            if key not in self.training_history:
                self.training_history[key] = []
            self.training_history[key].extend(values)
        try:
            plt.figure()
            if "loss" in self.training_history:
                plt.plot(self.training_history["loss"], label="loss")
            if "accuracy" in self.training_history:
                plt.plot(self.training_history["accuracy"], label="accuracy")
            plt.legend()
            plt.xlabel("Iterações")
            plt.ylabel("Métrica")
            plt.title("Histórico de Treinamento")
            plot_path = os.path.join("plots", "training_history.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Gráfico de treinamento salvo em {plot_path}")
        except Exception as e:
            print("Erro ao atualizar o gráfico de treinamento:", e)

    def start_training(self):
        """Inicia os processos de captura/preview e treinamento."""
        self.stop_event.clear()
        t_preview = Thread(target=self.capture_and_preview_loop, daemon=True)
        t_training = Thread(target=self.training_worker, daemon=True)
        t_preview.start()
        t_training.start()
        self.threads.extend([t_preview, t_training])

    # --- MODO PREDIÇÃO ---
    def prediction_loop(self):
        """Captura a tela, realiza a predição e executa as ações correspondentes."""
        self.status_label.config(text="Status: Predicting")
        print("Predição iniciada...")
        window_name = "Prediction View"
        self.setup_cv_window(window_name)
        while not self.stop_event.is_set():
            frame = self.capture_screen()
            if frame is not None:
                prediction = self.model.predict(np.expand_dims(frame, axis=0))[0]
                self.execute_actions(prediction)
                display_frame = (frame * 255).astype(np.uint8)
                active_actions = [key for key, idx in self.action_keys.items() if prediction[idx] > 0.5]
                cv2.putText(display_frame, f"Predictions: {', '.join(active_actions)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
                cv2.imshow(window_name, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup_windows()
        self.status_label.config(text="Status: Idle")

    def execute_actions(self, prediction):
        """Executa ações (teclas e comandos do mouse) com base na predição do modelo."""
        # Processa teclas
        for key, idx in self.action_keys.items():
            if not key.startswith('mouse'):
                if prediction[idx] > 0.5:
                    keyboard.press(key)
                else:
                    keyboard.release(key)

        # Processa posição e cliques do mouse
        if self.roblox_hwnd:
            try:
                win32gui.SetForegroundWindow(self.roblox_hwnd)
                rect = win32gui.GetWindowRect(self.roblox_hwnd)
                mouse_x = int(prediction[-2] * (rect[2] - rect[0]) + rect[0])
                mouse_y = int(prediction[-1] * (rect[3] - rect[1]) + rect[1])
                mouse.move(mouse_x, mouse_y)
            except Exception as e:
                print(f"Erro ao definir a janela em primeiro plano ou mover o mouse: {e}")

            for action, idx in [('left', self.action_keys['mouse_left']),
                                ('right', self.action_keys['mouse_right'])]:
                if prediction[idx] > 0.5:
                    mouse.press(action)
                else:
                    mouse.release(action)

    def start_prediction(self):
        """Inicia o modo de predição em uma thread separada."""
        self.stop_event.clear()
        t_pred = Thread(target=self.prediction_loop, daemon=True)
        t_pred.start()
        self.threads.append(t_pred)

    def save_model(self):
        """Salva os pesos do modelo na pasta 'models'."""
        try:
            model_path = os.path.join("models", "roblox_ai_model.weights.h5")
            self.model.save_weights(model_path)
            print("Modelo salvo com sucesso em", model_path)
        except Exception as e:
            print(f"Erro ao salvar o modelo: {e}")

    def stop_all(self):
        """Interrompe todos os processos e realiza a limpeza necessária."""
        print("Interrompendo todos os processos...")
        self.stop_event.set()

        # Libera teclas pressionadas
        for key in self.action_keys:
            if not key.startswith('mouse'):
                keyboard.release(key)
        mouse.release(button='left')
        mouse.release(button='right')

        # Aguarda o término das threads, exceto a thread corrente
        current_thread = threading.current_thread()
        for t in self.threads:
            if t is current_thread:
                continue
            if t.is_alive():
                t.join(timeout=1)
        self.threads = []
        self.cleanup_windows()
        self.status_label.config(text="Status: Idle")
        print("Todos os processos foram interrompidos.")

    def cleanup_resources(self):
        """Realiza a limpeza final ao sair."""
        self.stop_all()
        cv2.destroyAllWindows()
        if self.root.winfo_exists():
            self.root.destroy()

    def on_close(self):
        """Manipulador do fechamento da janela principal."""
        self.cleanup_resources()
        sys.exit(0)

    def run(self):
        """Inicia o loop principal da GUI."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()


if __name__ == "__main__":
    trainer = RobloxAITrainer()
    trainer.run()
