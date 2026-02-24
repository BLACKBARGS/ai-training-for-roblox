import sys
import os
import time
import threading
import atexit
import random
from threading import Thread
from datetime import datetime
from PIL import Image, ImageTk

import cv2
import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow import keras
from tensorflow import losses, metrics
import win32gui
import win32con
import keyboard
import mouse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dqn_agent import DQNAgent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')
tf.config.run_functions_eagerly(True)


class CollisionDetector:
    def __init__(self, action_keys, threshold=0.20, stuck_time=4, recovery_sequence=[('w', 0.5), ('space', 0.2), ('mouse_right', 0.1)]):
        self.frame_history = []
        self.action_keys = action_keys
        self.threshold = threshold
        self.stuck_time = stuck_time
        self.recovery_sequence = recovery_sequence
        self.reset_state()

    def reset_state(self):
        self.stuck_start_time = None
        self.in_recovery = False
        self.recovery_step = 0
        self.recovery_start = 0
        self.recovery_attempts = 0

    def check_recovery_loop(self):
        if self.recovery_attempts > 10:
            self.reset_state()
            return True
        return False

    def update(self, current_frame, current_actions):
        if self.in_recovery:
            return False

        self.frame_history.append(current_frame)
        if len(self.frame_history) > 5:
            self.frame_history.pop(0)

        diff = np.mean(np.abs(self.frame_history[-1] - self.frame_history[0]))
        movement_attempt = any(current_actions[self.action_keys.index(k)] > 0.5 for k in ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl', 'mouse_left', 'mouse_right', 'mouse_middle', 'mouse_x', 'mouse_x2',])

        if diff < self.threshold and movement_attempt:
            if self.stuck_start_time is None:
                self.stuck_start_time = time.time()
            elif time.time() - self.stuck_start_time >= self.stuck_time:
                self.recovery_attempts += 1
                return True
        else:
            self.stuck_start_time = None

        return False

    def get_recovery_action(self):
        if not self.in_recovery:
            self.in_recovery = True
            self.recovery_step = 0
            self.recovery_start = time.time()

        if self.recovery_step >= len(self.recovery_sequence):
            self.reset_state()
            return None

        current_step = self.recovery_sequence[self.recovery_step]
        if time.time() - self.recovery_start >= current_step[1]:
            self.recovery_step += 1
            self.recovery_start = time.time()
            return current_step[0]

        return current_step[0]


class RobloxAITrainer:
    def __init__(self):
        self.threads = []
        self.stop_event = threading.Event()
        self.training_history = {'loss': [], 'accuracy': []}
        self.current_total_epochs = 0
        self.last_frame = None
        self.batch_size = 32
        self.preview_update_interval = 0.033
        self.last_preview_update = 0
        self.active_actions = []
        self.mouse_smoothing = 0.3
        self.key_smoothing = 0.25
        self.valid_mouse_buttons = {'left', 'right', 'middle', 'x', 'x2'}
        self.prediction_plot_path = "temp/prediction_plot.png"
        self.running = False
        self.training_mode = 'supervised'
        self.agent = None

        self.action_keys = [
            'w', 'a', 's', 'd', 'space', 'shift', 'ctrl',
            'shift', 'ctrl', 'tab', 'q', 'e',  
            'r', 'f', '1', '2', '3', '4',
            'mouse_left', 'mouse_right', 
            'mouse_middle', 'mouse_x', 'mouse_x2' 
        ]

        self.mouse_axes = 2
        self.total_outputs = len(self.action_keys) + self.mouse_axes
        self.prev_mouse_position = None
        self.max_mouse_delta = 20
        self.collision_detector = CollisionDetector(self.action_keys)
        self.init_model()
        self.setup_gui()
        keyboard.add_hotkey('ctrl+shift+l', self.emergency_stop)
        atexit.register(self.cleanup_resources)

    def calculate_reward(self, state, collision):
        reward = 0
        if collision:
            reward -= 10

        if self.last_frame is not None and state is not None:
            diff = np.mean(np.abs(state - self.last_frame))
            reward += diff * 5

        if not any(self.active_actions):
            reward -= 0.1

        return reward

    def dqn_action_to_predictions(self, action_index):
        predictions = np.zeros(self.total_outputs)
        predictions[-2:] = 0.5

        if 0 <= action_index < len(self.action_keys):
            predictions[action_index] = 1.0

        return predictions

    def init_model(self):
        try:
            self.model = keras.Sequential([
                keras.layers.Input(shape=(240, 320, 1)),
                keras.layers.Conv2D(64, (5, 5), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(256, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(1024, activation='relu'),
                keras.layers.Dropout(0.6),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dropout(0.4),
                keras.layers.Dense(self.total_outputs, activation='sigmoid')
            ])
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss=losses.MeanSquaredError(),
                metrics=[metrics.MeanAbsoluteError()]
            )

        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao inicializar modelo: {str(e)}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Roblox AI Controller")
        self.root.geometry("1200x800")
        self.set_dark_theme()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.prediction_preview_label = ttk.Label(preview_frame)
        self.prediction_preview_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(status_frame, text="Status: Inativo")
        self.status_label.pack(side=tk.LEFT)

        self.action_display = ttk.Label(status_frame, text="Ações Ativas: ")
        self.action_display.pack(side=tk.LEFT, padx=10)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        model_frame = ttk.LabelFrame(control_frame, text="Modelo")
        model_frame.pack(side=tk.LEFT, padx=5)

        ttk.Button(model_frame, text="Carregar Modelo", command=self.load_model).pack(side=tk.LEFT, padx=2)
        ttk.Button(model_frame, text="Salvar Modelo", command=self.save_model).pack(side=tk.LEFT, padx=2)

        train_frame = ttk.LabelFrame(control_frame, text="Treinamento")
        train_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(train_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epoch_entry = ttk.Entry(train_frame, width=5)
        self.epoch_entry.insert(0, "10")
        self.epoch_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(train_frame, text="Iniciar Treino", command=self.start_training).pack(side=tk.LEFT, padx=2)

        predict_frame = ttk.LabelFrame(control_frame, text="Prediction")
        predict_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(predict_frame, text="Iniciar Prediction", command=self.start_prediction).pack(side=tk.LEFT, padx=2)
        ttk.Button(predict_frame, text="Parar Tudo", command=self.stop_all).pack(side=tk.LEFT, padx=2)

        self.console = tk.Text(main_frame, height=10, state=tk.DISABLED, bg='#333333', fg='white')
        self.console.pack(fill=tk.BOTH, expand=True)

    def log_message(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def update_preview(self, frame):
        if time.time() - self.last_preview_update < self.preview_update_interval or frame is None:
            return

        try:
            img = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img)
            img.thumbnail((800, 600))
            img = ImageTk.PhotoImage(image=img)
            self.preview_label.configure(image=img)
            self.preview_label.image = img
            self.last_preview_update = time.time()
        except Exception as e:
            self.log_message(f"Erro no preview: {str(e)}")

    def update_predictions_preview(self, predictions):
        try:
            plt.figure(figsize=(6, 3))
            plt.bar(range(len(self.action_keys)), predictions[:len(self.action_keys)])
            plt.xticks(range(len(self.action_keys)), self.action_keys, rotation=45)
            plt.ylim(0, 1)
            plt.title('Probabilidades das Ações')

            os.makedirs("temp", exist_ok=True)
            plt.savefig(self.prediction_plot_path, bbox_inches='tight')
            plt.close()

            img = Image.open(self.prediction_plot_path)
            img = ImageTk.PhotoImage(img)
            self.prediction_preview_label.configure(image=img)
            self.prediction_preview_label.image = img

        except Exception as e:
            self.log_message(f"Erro na preview de previsões: {str(e)}")

    def capture_screen(self):
        try:
            hwnd = win32gui.FindWindow(None, "Roblox")
            if not hwnd:
                return None

            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            rect = win32gui.GetWindowRect(hwnd)
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            if w <= 0 or h <= 0:
                return None

            screenshot = pyautogui.screenshot(region=(rect[0], rect[1], w, h))
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)
            return cv2.resize(frame, (320, 240)) / 255.0
        except Exception as e:
            self.log_message(f"Erro na captura: {str(e)}")
            return None
    
    def set_dark_theme(self):
        self.root.configure(bg='#2e2e2e')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', 
                        background='#2e2e2e', 
                        foreground='white',
                        fieldbackground='#3e3e3e',
                        borderwidth=1)

        style.configure('TButton', 
                        background='#4e4e4e', 
                        borderwidth=1,
                        relief='flat')

        style.map('TButton',
                  background=[('active', '#5e5e5e')],
                  relief=[('pressed', 'sunken')])

    def get_current_actions(self):
        """
        Captura as acoes atuais do teclado, cliques do mouse e o movimento relativo do mouse.
        O movimento relativo eh calculado como a diferenca entre a posição atual e a anterior,
        normalizado para o intervalo [0, 1], onde 0.5 representa nenhum movimento.
        """
        actions = np.zeros(len(self.action_keys) + self.mouse_axes)

        # Acoes do teclado e cliques do mouse
        for idx, key in enumerate(self.action_keys):
            if key.startswith('mouse_'):
                btn = key.split('_')[1]
                actions[idx] = 1.0 if mouse.is_pressed(btn) else 0.0
            else:
                actions[idx] = 1.0 if keyboard.is_pressed(key) else 0.0

        # Movimento relativo do mouse
        try:
            mx, my = pyautogui.position()
            if self.prev_mouse_position is None:
                self.prev_mouse_position = (mx, my)
                delta_x, delta_y = 0, 0
            else:
                delta_x = mx - self.prev_mouse_position[0]
                delta_y = my - self.prev_mouse_position[1]
                self.prev_mouse_position = (mx, my)

            # Normaliza o delta para [0, 1] com base em max_mouse_delta.
            norm_delta_x = np.clip((delta_x + self.max_mouse_delta) / (2 * self.max_mouse_delta), 0, 1)
            norm_delta_y = np.clip((delta_y + self.max_mouse_delta) / (2 * self.max_mouse_delta), 0, 1)

            actions[-2] = norm_delta_x
            actions[-1] = norm_delta_y
        except Exception as e:
            actions[-2:] = 0.5

        return actions

    def training_worker(self):
        try:
            target_epochs = int(self.epoch_entry.get())
        # Define o modo de treinamento: 'dqn' para DQNAgent ou 'supervised' para treinamento direto
            mode = getattr(self, 'training_mode', 'supervised')

            if mode == 'dqn':
                if self.agent is None:
                    self.agent = DQNAgent((240, 320, 1), len(self.action_keys))

                episode = 0
                while not self.stop_event.is_set() and episode < target_epochs:
                    state = self.capture_screen()
                    if state is None:
                        continue

                    total_reward = 0
                    steps = 0
                    while not self.stop_event.is_set() and steps < 500:
                        action = self.agent.act(state)
                        self.execute_actions(self.dqn_action_to_predictions(action))
                    
                        next_state = self.capture_screen()
                        if next_state is None:
                            break

                        collision = self.collision_detector.update(next_state, self.get_current_actions())
                        reward = self.calculate_reward(next_state, collision)
                        self.agent.remember(state, action, reward, next_state, collision)
                    
                        if len(self.agent.memory) >= self.agent.batch_size:
                            loss = self.agent.replay()
                            if steps % 10 == 0:
                                self.log_message(f"Episode: {episode}, Step: {steps}, Loss: {loss:.4f}")
                    
                        total_reward += reward
                        state = next_state
                        steps += 1
                        self.update_preview(state)
                    
                        if collision:
                            break
                    
                        time.sleep(0.05)
                
                    self.log_message(f"Episode {episode} finalizado - Reward: {total_reward:.2f}")
                    episode += 1
                
                    if episode % 10 == 0:
                        self.save_model(f"models/roblox_dqn_model_{episode}")
            
                self.log_message("Treinamento DQN concluído!")
        
            else:  # Modo 'supervised'
                buffer = []
                while not self.stop_event.is_set() and self.current_total_epochs < target_epochs:
                    frame = self.capture_screen()
                    if frame is not None:
                        actions = self.get_current_actions()
                        buffer.append((frame, actions))
                    
                        if len(buffer) >= self.batch_size:
                            X = np.array([x[0] for x in buffer])
                            y = np.array([x[1] for x in buffer])
                            history = self.model.train_on_batch(X[..., np.newaxis], y)
                            loss_val = float(history[0])
                            acc_val = float(history[1])
                            self.training_history['loss'].append(loss_val)
                            self.training_history['accuracy'].append(acc_val)
                            self.current_total_epochs += 1
                        
                            if self.current_total_epochs % 5 == 0:
                                self.log_message(f"Época {self.current_total_epochs} - Loss: {loss_val:.4f}")
                                self.save_training_plot()
                        
                            buffer = []
                
                    self.update_preview(frame)
                    time.sleep(0.05)
            
                self.save_model("models/roblox_ai_model.h5")
                self.log_message("Treino supervisionado concluído e modelo salvo!")
    
        except Exception as e:
            self.log_message(f"Erro no treino: {str(e)}")
    
        finally:
            self.stop_all()


    def save_training_plot(self):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.training_history['loss'], label='Loss')
            plt.plot(self.training_history['accuracy'], label='Accuracy')
            plt.title(f"Progresso do Treino - {datetime.now().strftime('%d/%m %H:%M')}")
            plt.legend()
            os.makedirs("plots", exist_ok=True)
            filename = f"plots/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            plt.close()
        except Exception as e:
            self.log_message(f"Erro ao salvar grafico: {str(e)}")

    def prediction_loop(self):
        try:
            self.running = True
            while self.running and not self.stop_event.is_set():
                start_time = time.time()

                if self.collision_detector.check_recovery_loop():
                    self.log_message("Loop detectado! Resetando sistema...")
                    self.emergency_stop()
                    return

                frame = self.capture_screen()
                self.update_preview(frame)

                collision = False
                if self.last_frame is not None and frame is not None:
                    collision = self.collision_detector.update(frame, self.get_current_actions())
                self.last_frame = frame.copy() if frame is not None else None
                recovery_action = self.collision_detector.get_recovery_action()
                if recovery_action:
                    self.execute_recovery_action(recovery_action)
                    time.sleep(0.1)
                    continue

                if collision:
                    self.log_message("Colisao detectada! Executando rotina de escape...")
                    self.collision_detector.in_recovery = True
                    continue

                if frame is not None:
                    pred = self.model.predict(frame[np.newaxis, ..., np.newaxis], verbose=0)[0]
                    self.execute_actions(pred)
                    self.update_action_display(pred)
                    self.update_predictions_preview(pred)

                elapsed = time.time() - start_time
                if elapsed < 0.033:
                    time.sleep(0.033 - elapsed)

        except Exception as e:
            self.log_message(f"Erro no predict: {str(e)}")
            self.stop_all()
        finally:
            self.running = False

    def execute_recovery_action(self, action):
        try:
            duration = np.random.uniform(0.1, 0.3)

            if action.startswith('mouse_'):
                btn = action.split('_')[1]
                mouse.click(btn)
                time.sleep(0.1)
            else:
                keyboard.press(action)
                time.sleep(duration)
                keyboard.release(action)

            self.log_message(f"Recuperacao: {action} ({duration:.2f}s)")
        except Exception as e:
            self.log_message(f"Erro na recuperacao: {str(e)}")

    def execute_actions(self, predictions):
        try:
            new_actions = []

            for idx, key in enumerate(self.action_keys):
                current_state = predictions[idx] > 0.5

                if key.startswith('mouse_'):
                    btn = key.split('_')[1]
                    if btn in self.valid_mouse_buttons:
                        if current_state:
                            mouse.press(btn)
                            new_actions.append(key)
                        else:
                            mouse.release(btn)
                else:
                    if current_state:
                        keyboard.press(key)
                        new_actions.append(key)
                    else:
                        keyboard.release(key)

            self.active_actions = new_actions

            delta_x = (predictions[-2] * 2 - 1) * self.max_mouse_delta
            delta_y = (predictions[-1] * 2 - 1) * self.max_mouse_delta
            move_x = int(delta_x * self.mouse_smoothing)
            move_y = int(delta_y * self.mouse_smoothing)
            pyautogui.moveRel(move_x, move_y)

        except Exception as e:
            self.log_message(f"Erro na execucao: {str(e)}")

    def update_action_display(self, predictions):
        action_text = "Ações Ativas: " + ", ".join(self.active_actions)
        self.action_display.config(text=action_text)

    def load_model(self, path=None):
        try:
            if not path:
                path = filedialog.askopenfilename(
                    initialdir="models",
                    title="Selecionar Modelo",
                    filetypes=(("HDF5 models", "*.h5"), ("Todos os arquivos", "*.*"))
                )

            if path and os.path.exists(path):
                self.model = keras.models.load_model(
                    path,
                    custom_objects={
                        'MeanSquaredError': losses.MeanSquaredError,
                        'MeanAbsoluteError': metrics.MeanAbsoluteError
                    }
                )
                self.log_message(f"Modelo carregado: {os.path.basename(path)}")

        except Exception as e:
            self.log_message(f"Erro ao carregar modelo: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao carregar modelo:\n{str(e)}")
            self.model = None

    def save_model(self, path=None):
        try:
            if not path:
                path = filedialog.asksaveasfilename(
                    initialdir="models",
                    title="Salvar Modelo",
                    defaultextension=".h5",
                    filetypes=(("HDF5 models", "*.h5"), ("Todos os arquivos", "*.*"))
                )

            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.model.save(path, save_format='h5')
                self.log_message(f"Modelo salvo: {os.path.basename(path)}")

        except Exception as e:
            self.log_message(f"Erro ao salvar modelo: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao salvar modelo:\n{str(e)}")

    def start_training(self):
        self.stop_event.clear()
        self.status_label.config(text="Status: Treinando")
        train_thread = Thread(target=self.training_worker)
        train_thread.start()
        self.threads.append(train_thread)

    def start_prediction(self):
        self.stop_event.clear()
        self.status_label.config(text="Status: Predicting")
        predict_thread = Thread(target=self.prediction_loop)
        predict_thread.start()
        self.threads.append(predict_thread)

    def stop_all(self):
        self.stop_event.set()
        self.running = False
        try:
            if self.root.winfo_exists():
                self.status_label.config(text="Status: Inativo")
        except tk.TclError:
            pass

        current_thread = threading.current_thread()
        for t in self.threads[:]:
            if t is not current_thread and t.is_alive():
                try:
                    t.join(timeout=1)
                except RuntimeError as e:
                    self.log_message(f"Erro ao parar thread: {str(e)}")
            if t in self.threads:
                self.threads.remove(t)

    def emergency_stop(self):
        self.stop_all()
        self.log_message("PARADA DE EMERGENCIA ATIVADA!")
        for key in self.action_keys:
            if not key.startswith('mouse_'):
                keyboard.release(key)
            else:
                btn = key.split('_')[1]
                if btn in self.valid_mouse_buttons:
                    try:
                        mouse.release(btn)
                    except Exception as e:
                        self.log_message(f"Erro ao liberar {key}: {str(e)}")
                else:
                    self.log_message(f"Botao de mouse invalido: {btn}")
        self.collision_detector.reset_state()

    def cleanup_resources(self):
        self.stop_all()
        cv2.destroyAllWindows()
        try:
            os.remove(self.prediction_plot_path)
        except:
            pass

    def on_close(self):
        if messagebox.askokcancel("Sair", "Deseja realmente sair?"):
            self.running = False
            self.cleanup_resources()
            self.root.destroy()

if __name__ == "__main__":
    ai_trainer = RobloxAITrainer()
    ai_trainer.root.mainloop()
