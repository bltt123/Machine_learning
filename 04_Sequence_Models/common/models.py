"""04 序列家族模型：HMM / CRF（字符级 BIO）+ RNN / LSTM / GRU 文本分类。

HMM/CRF 为纯 NumPy 教学实现；RNN 系为 PyTorch。
"""
import numpy as np
import torch
import torch.nn as nn

# ── HMM ──
class HMMSegmenter:
    def __init__(self, tags, vocab_size, alpha=1.0):
        self.tags = tags
        self.tag2id = {t: i for i, t in enumerate(tags)}
        self.n_tag = len(tags)
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.pi = None
        self.A = None
        self.B = None

    def fit(self, X_ids, Y_ids):
        n = self.n_tag
        V = self.vocab_size
        a = self.alpha
        pi_c = np.full(n, a)
        A_c = np.full((n, n), a)
        B_c = np.full((n, V), a)
        for x, y in zip(X_ids, Y_ids):
            pi_c[y[0]] += 1
            for t, tag in enumerate(y):
                B_c[tag, x[t]] += 1
                if t > 0:
                    A_c[y[t - 1], tag] += 1
        self.pi = np.log(pi_c / pi_c.sum())
        self.A = np.log(A_c / A_c.sum(axis=1, keepdims=True))
        self.B = np.log(B_c / B_c.sum(axis=1, keepdims=True))
        return self

    def viterbi(self, x_ids):
        T = len(x_ids)
        n = self.n_tag
        dp = np.full((T, n), -1e18)
        back = np.zeros((T, n), dtype=int)
        for s in range(n):
            dp[0, s] = self.pi[s] + self.B[s, x_ids[0]]
        for t in range(1, T):
            for s in range(n):
                scores = dp[t - 1] + self.A[:, s] + self.B[s, x_ids[t]]
                back[t, s] = int(np.argmax(scores))
                dp[t, s] = float(np.max(scores))
        path = [0] * T
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = int(back[t, path[t]])
        return path

# ── CRF ──
class LinearCRF:
    """线性链 CRF：score = sum_t emit[x_t, y_t] + sum_t trans[y_{t-1}, y_t]。"""
    def __init__(self, n_tag, vocab_size, seed=0):
        rng = np.random.default_rng(seed)
        self.n_tag = n_tag
        self.vocab_size = vocab_size
        self.emit = rng.normal(0, 0.1, (vocab_size, n_tag))
        self.trans = rng.normal(0, 0.1, (n_tag, n_tag))

    def _forward_log_partition(self, x_ids):
        T = len(x_ids)
        alpha = np.full((T, self.n_tag), -1e18)
        alpha[0] = self.emit[x_ids[0]]
        for t in range(1, T):
            for s in range(self.n_tag):
                vals = alpha[t - 1] + self.trans[:, s]
                m = np.max(vals)
                alpha[t, s] = self.emit[x_ids[t], s] + m + np.log(np.sum(np.exp(vals - m)))
        m = np.max(alpha[-1])
        return m + np.log(np.sum(np.exp(alpha[-1] - m)))

    def _score(self, x_ids, y_ids):
        s = float(self.emit[x_ids[0], y_ids[0]])
        for t in range(1, len(x_ids)):
            s += float(self.emit[x_ids[t], y_ids[t]])
            s += float(self.trans[y_ids[t - 1], y_ids[t]])
        return s

    def viterbi(self, x_ids):
        T = len(x_ids)
        n = self.n_tag
        dp = np.full((T, n), -1e18)
        back = np.zeros((T, n), dtype=int)
        dp[0] = self.emit[x_ids[0]]
        for t in range(1, T):
            for s in range(n):
                scores = dp[t - 1] + self.trans[:, s] + self.emit[x_ids[t], s]
                back[t, s] = int(np.argmax(scores))
                dp[t, s] = float(np.max(scores))
        path = [0] * T
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 1, 0, -1):
            path[t - 1] = int(back[t, path[t]])
        return path

    def fit(self, X_ids, Y_ids, epochs=80, lr=0.12, verbose=False):
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for x, y in zip(X_ids, Y_ids):
                T = len(x)
                logZ = self._forward_log_partition(x)
                s_gold = self._score(x, y)
                loss = logZ - s_gold
                total_loss += loss
                alpha = np.full((T, self.n_tag), -1e18)
                alpha[0] = self.emit[x[0]]
                for t in range(1, T):
                    for s in range(self.n_tag):
                        vals = alpha[t - 1] + self.trans[:, s]
                        m = np.max(vals)
                        alpha[t, s] = self.emit[x[t], s] + m + np.log(np.sum(np.exp(vals - m)))
                beta = np.full((T, self.n_tag), -1e18)
                beta[-1] = 0.0
                for t in range(T - 2, -1, -1):
                    for s in range(self.n_tag):
                        vals = self.trans[s, :] + self.emit[x[t + 1], :] + beta[t + 1]
                        m = np.max(vals)
                        beta[t, s] = m + np.log(np.sum(np.exp(vals - m)))
                for t in range(T):
                    for s in range(self.n_tag):
                        p = np.exp(alpha[t, s] + beta[t, s] - logZ)
                        grad = p - (1 if y[t] == s else 0)
                        self.emit[x[t], s] -= lr * grad
                for t in range(1, T):
                    for p_ in range(self.n_tag):
                        for c_ in range(self.n_tag):
                            logp = alpha[t - 1, p_] + self.trans[p_, c_] + self.emit[x[t], c_] + beta[t, c_] - logZ
                            prob = np.exp(logp)
                            indicator = 1 if (y[t - 1] == p_ and y[t] == c_) else 0
                            grad = prob - indicator
                            self.trans[p_, c_] -= lr * grad
            if verbose and ep % 20 == 0:
                print(f"epoch {ep:02d} loss {total_loss/len(X_ids):.4f}")
        return self

# ── 文本分类：MeanPool / RNN / LSTM / GRU ──
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hid_dim=32, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        e = self.emb(x)
        m = e.mean(dim=1)
        return self.fc(m)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hid_dim=32, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.rnn(e)
        return self.fc(h.squeeze(0))

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hid_dim=32, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        return self.fc(h.squeeze(0))

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hid_dim=32, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        e = self.emb(x)
        _, h = self.gru(e)
        return self.fc(h.squeeze(0))

# ── Seq2Seq with Attention（for 03）──
class Seq2SeqAttention(nn.Module):
    """Tiny copy/reverse：Encoder GRU + Bahdanau attention Decoder。BOS=vocal_size。"""
    def __init__(self, vocab_size=12, emb_dim=16, hid_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_id = vocab_size
        self.emb = nn.Embedding(vocab_size + 1, emb_dim)
        self.encoder = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.attn_W = nn.Linear(hid_dim * 2, hid_dim)
        self.attn_v = nn.Linear(hid_dim, 1, bias=False)
        self.decoder_cell = nn.GRUCell(emb_dim + hid_dim, hid_dim)
        self.out = nn.Linear(hid_dim * 2, vocab_size)

    def forward(self, src, tgt_input):
        """tgt_input 为解码器输入（已含 BOS 的移位序列），与 target 等长。"""
        B, T = src.shape
        _, S = tgt_input.shape
        enc_emb = self.emb(src)
        enc_out, h = self.encoder(enc_emb)
        dec_h = h.squeeze(0)
        logits = []
        attns = []
        for t in range(S):
            inp = self.emb(tgt_input[:, t])
            dec_exp = dec_h.unsqueeze(1).expand(-1, T, -1)
            energy = torch.tanh(self.attn_W(torch.cat([dec_exp, enc_out], dim=-1)))
            scores = self.attn_v(energy).squeeze(-1)
            attn = torch.softmax(scores, dim=-1)
            ctx = (attn.unsqueeze(-1) * enc_out).sum(dim=1)
            dec_in = torch.cat([inp, ctx], dim=-1)
            dec_h = self.decoder_cell(dec_in, dec_h)
            out_in = torch.cat([dec_h, ctx], dim=-1)
            logits.append(self.out(out_in))
            attns.append(attn)
        return torch.stack(logits, dim=1), torch.stack(attns, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=6):
        self.eval()
        B, T = src.shape
        enc_emb = self.emb(src)
        enc_out, h = self.encoder(enc_emb)
        dec_h = h.squeeze(0)
        cur = torch.full((B,), self.bos_id, dtype=torch.long, device=src.device)
        outs = []
        attns = []
        for _ in range(max_len):
            inp = self.emb(cur)
            dec_exp = dec_h.unsqueeze(1).expand(-1, T, -1)
            energy = torch.tanh(self.attn_W(torch.cat([dec_exp, enc_out], dim=-1)))
            scores = self.attn_v(energy).squeeze(-1)
            attn = torch.softmax(scores, dim=-1)
            ctx = (attn.unsqueeze(-1) * enc_out).sum(dim=1)
            dec_in = torch.cat([inp, ctx], dim=-1)
            dec_h = self.decoder_cell(dec_in, dec_h)
            logit = self.out(torch.cat([dec_h, ctx], dim=-1))
            cur = logit.argmax(dim=-1)
            outs.append(cur)
            attns.append(attn)
        return torch.stack(outs, dim=1), torch.stack(attns, dim=1)
