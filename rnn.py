import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.vocab_size  = input_size
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(output_size, hidden_size) * 0.1
        self.bh  = np.zeros((hidden_size, 1))
        self.by  = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.last_hs = {-1: h}
        for t, idx in enumerate(inputs):
            x_one = np.zeros((self.vocab_size, 1))
            x_one[idx] = 1
            h = np.tanh(self.Wxh @ x_one + self.Whh @ h + self.bh)
            self.last_hs[t] = h
        y = self.Why @ h + self.by
        return y, self.last_hs

    def softmax(self, x):
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex, axis=0)

    def loss(self, scores, y_true):
        probs = self.softmax(scores)
        return -np.log(probs[y_true, 0] + 1e-9)

    def predict(self, inputs):
        scores, _ = self.forward(inputs)
        probs = self.softmax(scores)
        return np.argmax(probs)

    def backward(self, inputs, hs, y_true, lr=1e-2):
        scores, _ = self.forward(inputs)
        probs = self.softmax(scores)
        probs[y_true] -= 1
        dWhy = probs @ hs[len(inputs)-1].T
        dby  = probs
        dh   = self.Why.T @ probs
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh  = np.zeros_like(self.bh)
        for t in reversed(range(len(inputs))):
            h = hs[t]
            h_prev = hs[t-1] if t>0 else np.zeros_like(h)
            dt = (1 - h*h) * dh
            dbh  += dt
            idx = inputs[t]
            x_one = np.zeros((self.vocab_size,1)); x_one[idx]=1
            dWxh += dt @ x_one.T
            dWhh += dt @ h_prev.T
            dh    = self.Whh.T @ dt
        for param, grad in [(self.Wxh,dWxh),(self.Whh,dWhh),(self.Why,dWhy),(self.bh,dbh),(self.by,dby)]:
            param -= lr * grad