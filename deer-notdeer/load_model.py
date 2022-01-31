from tensorflow.keras.models import load_model

if __name__ == '__main__':
    m = load_model('./model_checkpoints')
    print(m.summary())