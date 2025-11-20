import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy import signal
import os

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

class AudioClassifier:
    def __init__(self):
        self.features = []
        self.labels = []
        self.model = None
        self.classes = []
        self.model_type = "random_forest"
        self.metrics = {}
        self.cv_metrics = {}
        
    def set_model_type(self, model_type):
        self.model_type = model_type
        
    def remove_silence_advanced(self, audio, sr, threshold_db=-40, min_silence_duration=0.2, min_speech_duration=0.3):
        if len(audio) == 0:
            return audio
            
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        rms_energy = np.sqrt(np.mean(frames**2, axis=0))
        
        rms_db = 20 * np.log10(rms_energy + 1e-10)

        if len(rms_db) > 0:
            noise_level = np.median(rms_db)
            speech_threshold = max(threshold_db, noise_level + 5)
        else:
            speech_threshold = threshold_db
        
        speech_frames = rms_db > speech_threshold
        
        speech_segments = self._find_speech_segments(speech_frames, min_silence_duration, min_speech_duration, hop_length, sr)
        
        if not speech_segments:
            return audio
            
        start_sample = speech_segments[0][0]
        end_sample = speech_segments[-1][1]
        
        buffer_samples = int(0.05 * sr)
        start_sample = max(0, start_sample - buffer_samples)
        end_sample = min(len(audio), end_sample + buffer_samples)
        
        if end_sample - start_sample < int(0.3 * sr):
            return audio
            
        return audio[start_sample:end_sample]
    
    def _find_speech_segments(self, speech_frames, min_silence_duration, min_speech_duration, hop_length, sr):
        segments = []
        in_speech = False
        speech_start = 0
        
        min_silence_frames = int(min_silence_duration * sr / hop_length)
        min_speech_frames = int(min_speech_duration * sr / hop_length)
        
        silence_count = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                in_speech = True
                speech_start = i * hop_length
                silence_count = 0
            elif not is_speech and in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    in_speech = False
                    speech_end = (i - silence_count) * hop_length + hop_length
                    if (speech_end - speech_start) / sr >= min_speech_duration:
                        segments.append((speech_start, speech_end))
                    silence_count = 0
            elif is_speech and in_speech:
                silence_count = 0
        
        if in_speech:
            speech_end = len(speech_frames) * hop_length
            if (speech_end - speech_start) / sr >= min_speech_duration:
                segments.append((speech_start, speech_end))
                
        return segments
    
    def apply_noise_reduction(self, audio, sr, noise_reduction_level=0.7):
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        noise_frames = int(0.1 * sr / (len(audio) / magnitude.shape[1]))
        noise_frames = min(noise_frames, magnitude.shape[1] // 3)
        
        if noise_frames > 1:
            noise_estimate = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            magnitude_reduced = magnitude - noise_reduction_level * noise_estimate
            magnitude_reduced = np.maximum(magnitude_reduced, 0.01 * magnitude)
            
            stft_reduced = magnitude_reduced * np.exp(1j * phase)
            audio_reduced = librosa.istft(stft_reduced)
            
            if len(audio_reduced) > len(audio):
                audio_reduced = audio_reduced[:len(audio)]
            else:
                padding = np.zeros(len(audio) - len(audio_reduced))
                audio_reduced = np.concatenate([audio_reduced, padding])
                
            return audio_reduced
        else:
            return audio
    
    def extract_features(self, audio, sr):
        try:
            audio_clean = self.remove_silence_advanced(audio, sr)
            audio_clean = self.apply_noise_reduction(audio_clean, sr)
            
            if len(audio_clean) < int(0.3 * sr):
                audio_clean = self.apply_noise_reduction(audio, sr)
                
            features_list = []
            
            mfcc = librosa.feature.mfcc(y=audio_clean, sr=sr, n_mfcc=13)
            features_list.extend(np.mean(mfcc, axis=1))
            features_list.extend(np.std(mfcc, axis=1))
            
            chroma = librosa.feature.chroma_stft(y=audio_clean, sr=sr)
            features_list.extend(np.mean(chroma, axis=1))
            features_list.extend(np.std(chroma, axis=1))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_clean, sr=sr)
            features_list.extend(np.mean(spectral_contrast, axis=1))
            
            rms = librosa.feature.rms(y=audio_clean)
            features_list.append(float(np.mean(rms)))
            features_list.append(float(np.std(rms)))
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_clean, sr=sr)
            features_list.append(float(np.mean(spectral_centroids)))
            features_list.append(float(np.std(spectral_centroids)))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_clean, sr=sr)
            features_list.append(float(np.mean(spectral_rolloff)))
            features_list.append(float(np.std(spectral_rolloff)))
            
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_clean)
            features_list.append(float(np.mean(zero_crossing_rate)))
            features_list.append(float(np.std(zero_crossing_rate)))
            
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_clean, sr=sr)
            features_list.append(float(np.mean(mel_spectrogram)))
            features_list.append(float(np.std(mel_spectrogram)))
            
            features_array = np.array(features_list, dtype=np.float64)
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features_array
            
        except Exception as e:
            print(f"Ошибка при извлечении признаков: {e}")
            return np.zeros(65, dtype=np.float64)
    
    def add_sample(self, audio, label, sr=22050):
        features = self.extract_features(audio, sr)
        self.features.append(features)
        self.labels.append(label)
        
    def remove_last_sample(self):
        if self.features and self.labels:
            self.features.pop()
            removed_label = self.labels.pop()
            return removed_label
        return None
        
    def train(self, use_cross_validation=True, cv_folds=5):
        if len(set(self.labels)) < 2:
            raise ValueError("Нужно как минимум 2 разных класса для обучения")
            
        X = np.array(self.features)
        y = np.array(self.labels)
            
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "catboost" and CATBOOST_AVAILABLE:
            self.model = CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            if self.model_type == "catboost" and not CATBOOST_AVAILABLE:
                raise ValueError("CatBoost не установлен. Установите его с помощью: pip install catboost")
            else:
                raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        
        if use_cross_validation and len(self.labels) >= cv_folds * 2:
            cv = StratifiedKFold(n_splits=min(cv_folds, len(set(self.labels))), shuffle=True, random_state=42)
            
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            cv_results = cross_validate(self.model, X, y, cv=cv, scoring=scoring, return_train_score=False)
            
            self.cv_metrics = {
                'accuracy': {
                    'mean': np.mean(cv_results['test_accuracy']),
                    'std': np.std(cv_results['test_accuracy'])
                },
                'precision': {
                    'mean': np.mean(cv_results['test_precision_weighted']),
                    'std': np.std(cv_results['test_precision_weighted'])
                },
                'recall': {
                    'mean': np.mean(cv_results['test_recall_weighted']),
                    'std': np.std(cv_results['test_recall_weighted'])
                },
                'f1': {
                    'mean': np.mean(cv_results['test_f1_weighted']),
                    'std': np.std(cv_results['test_f1_weighted'])
                },
                'fold_scores': {
                    'accuracy': cv_results['test_accuracy'],
                    'precision': cv_results['test_precision_weighted'],
                    'recall': cv_results['test_recall_weighted'],
                    'f1': cv_results['test_f1_weighted']
                }
            }
        
        self.model.fit(X, y)
        self.classes = self.model.classes_.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        temp_model = self._create_temp_model()
        temp_model.fit(X_train, y_train)
        y_pred = temp_model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'use_cross_validation': use_cross_validation
        }
        
        return {
            'standard_metrics': self.metrics,
            'cv_metrics': self.cv_metrics if use_cross_validation else None
        }
    
    def _create_temp_model(self):
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "catboost" and CATBOOST_AVAILABLE:
            return CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False
            )
        elif self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def predict(self, audio, sr=22050):
        if self.model is None:
            raise ValueError("Модель не обучена")
        features = self.extract_features(audio, sr)
        return self.model.predict([features])[0]