"""GPU detection script for TensorFlow and PyTorch."""


def check_tensorflow_gpu():
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        print('TensorFlow GPUs:', gpus)
        print('TensorFlow GPU count:', len(gpus))
    except Exception as e:
        print('TensorFlow error:', e)


def check_pytorch_gpu():
    try:
        import torch

        print('PyTorch CUDA available:', torch.cuda.is_available())
        print('PyTorch CUDA version:', torch.version.cuda)
        print('PyTorch GPU count:', torch.cuda.device_count())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f'GPU {i}:', torch.cuda.get_device_name(i))
    except Exception as e:
        print('PyTorch error:', e)


if __name__ == '__main__':
    print('=' * 50)
    print('GPU DETECTION')
    print('=' * 50)

    check_tensorflow_gpu()
    print()
    check_pytorch_gpu()
