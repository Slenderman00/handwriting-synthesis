"""
hand.py — self-contained Hand class for the package.
"""
import os
import logging
import numpy as np
from pathlib import Path

_PKG_DIR = Path(__file__).parent


class Hand:
    def __init__(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        import sys
        pkg = str(_PKG_DIR)
        if pkg not in sys.path:
            sys.path.insert(0, pkg)

        from rnn import rnn as RNN

        self.nn = RNN(
            log_dir=str(_PKG_DIR / "logs"),
            checkpoint_dir=str(_PKG_DIR / "checkpoints"),
            prediction_dir=str(_PKG_DIR / "predictions"),
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer="rms",
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10,
        )

        with self.nn.graph.as_default():
            with self.nn.session.as_default():
                self.nn.restore(17900)

    def _sample(self, lines, biases=None, styles=None):
        biases = biases or [0.5] * len(lines)
        styles = styles or [0]   * len(lines)

        import drawing

        samples = []
        for line, bias, style in zip(lines, biases, styles):
            style_strokes_path = _PKG_DIR / "styles" / f"style-{style}-strokes.npy"
            style_chars_path   = _PKG_DIR / "styles" / f"style-{style}-chars.npy"

            if style_strokes_path.exists():
                style_strokes = np.load(str(style_strokes_path))
                style_chars   = np.load(str(style_chars_path)).tobytes().decode("utf-8")
                prime         = True
                prime_length  = 500
            else:
                style_strokes = np.zeros([1, 3])
                style_chars   = ""
                prime         = False
                prime_length  = 0

            char_seq = style_chars + line
            encoded  = drawing.encode_ascii(char_seq)

            feed = {
                self.nn.prime:         prime,
                self.nn.x_prime:       np.expand_dims(style_strokes, 0),
                self.nn.x_prime_len:   np.array([prime_length]),
                self.nn.c:             np.expand_dims(encoded, 0),
                self.nn.c_len:         np.array([len(char_seq)]),
                self.nn.bias:          np.array([bias]),
                self.nn.sample_tsteps: 800,
                self.nn.num_samples:   1,
            }

            with self.nn.graph.as_default():
                with self.nn.session.as_default():
                    [sample] = self.nn.session.run([self.nn.sampled_sequence], feed)

            samples.append(sample[0])

        return samples
