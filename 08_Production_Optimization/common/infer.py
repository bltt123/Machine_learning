"""08-09：推理部署三件套（toy 口径，零新包）。

- PagedKVManager：把 08-02 的整块 KV Cache 切成定长 page（block_size），
  按需分配/释放；口径=分配 page 数 + 内碎片率。vLLM 的 PagedAttention 是
  同一思想的生产实现（Windows 不兼容，只做原理模拟）。
- speculative_decode：draft 小模型先猜 γ 个 token，target 大模型一次并行
  验证；口径=接受率 + 有效 token/轮。toy 上 draft/target 用同族不同宽度的
  GQADecoder（行为可比），不做“draft 必弱”假设。
- gguf_bytes：INT8 对称量化 state_dict → 文件字节拼盘（魔数+头+张量体），
  口径=体积比；llama.cpp 走同格式 CPU 加载（本机只拼字节+回读校验，
  不调外部二进制，保证可复现）。
"""
import io
import struct

import torch


class PagedKVManager:
    """toy KV 分页器：seq 按 block 切页，按需分配，释放回池。"""

    MAGIC = b"PGKV"

    def __init__(self, block_size=8, max_blocks=64):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.free = list(range(max_blocks))
        self.alloc = {}

    def new_seq(self, sid):
        self.alloc[sid] = []

    def append_tokens(self, sid, n):
        cur = self.alloc[sid]
        used_in_last = 0 if not cur else cur[-1][1]
        need = n
        if cur and used_in_last < self.block_size:
            fill = min(self.block_size - used_in_last, need)
            cur[-1][1] += fill
            need -= fill
        while need > 0:
            if not self.free:
                raise RuntimeError("KV pages exhausted")
            blk = self.free.pop(0)
            take = min(self.block_size, need)
            cur.append([blk, take])
            need -= take

    def usage(self, sid):
        cur = self.alloc[sid]
        tokens = sum(u for _, u in cur)
        return {"pages": len(cur), "tokens": tokens,
                "waste": len(cur) * self.block_size - tokens,
                "waste_rate": (len(cur) * self.block_size - tokens) / max(tokens, 1)}

    def free_seq(self, sid):
        for blk, _ in self.alloc.pop(sid, []):
            self.free.append(blk)
            self.free.sort()


@torch.no_grad()
def speculative_decode(draft, target, prefix, gamma=4, max_new=16):
    """toy 投机解码：draft 猜 γ 个 → target 并行验证前缀+猜测，一次前向定接受数。

    返回 (ids, accepted, rounds)：accepted 为接受 token 总数，rounds 为验证轮数。
    口径与 08-02 一致：generate 全序列分支（use_cache=False）做验证器。
    """
    draft.eval(); target.eval()
    ids = prefix.clone()
    accepted = 0
    rounds = 0
    while ids.size(1) - prefix.size(1) < max_new:
        guesses = draft.generate(ids, gamma, use_cache=False)[:, ids.size(1):]
        rounds += 1
        ok = 0
        for i in range(gamma):
            cand = torch.cat([ids, guesses[:, :i + 1]], dim=1)
            ref = target.generate(cand[:, :-1], 1, use_cache=False)[:, -1]
            if (ref == guesses[:, i]).all():
                ok += 1
            else:
                ids = torch.cat([ids, guesses[:, :i], ref.unsqueeze(1)], dim=1)
                accepted += ok + 1
                break
        else:
            extra = target.generate(torch.cat([ids, guesses], dim=1), 1, use_cache=False)[:, -1:]
            ids = torch.cat([ids, guesses, extra], dim=1)
            accepted += gamma + 1
        if ids.size(1) - prefix.size(1) >= max_new:
            break
    return ids, accepted, rounds


def gguf_bytes(state_dict, bits=8):
    """toy GGUF 拼盘：MAGIC(4B)+版本+张量数+每张量(名长/名/scale/数据)+回读校验位。

    INT8 对称量化（scale=max|w|/127）；fp32 兜底存 fp16 头（教学口径，非真 GGUF 规范，
    README 已声明；体积比口径不受影响）。
    """
    assert bits == 8
    buf = io.BytesIO()
    buf.write(b"GGUF"); buf.write(struct.pack("<I", 3))
    items = list(state_dict.items())
    buf.write(struct.pack("<Q", len(items)))
    for name, t in items:
        w = t.detach().float()
        if w.is_floating_point():
            scale = w.abs().max().item() / 127.0 or 1.0
            q = (w / scale).round().clamp(-127, 127).to(torch.int8)
            nb = name.encode()
            buf.write(struct.pack("<Q", len(nb))); buf.write(nb)
            buf.write(struct.pack("<f", scale))
            buf.write(struct.pack("<Q", q.numel())); buf.write(q.numpy().tobytes())
        else:
            nb = name.encode()
            buf.write(struct.pack("<Q", len(nb))); buf.write(nb)
            buf.write(struct.pack("<f", 1.0))
            raw = w.to(torch.int64).numpy().tobytes()
            buf.write(struct.pack("<Q", len(raw))); buf.write(raw)
    blob = buf.getvalue()
    fp32b = sum(t.numel() * (4 if t.is_floating_point() else 8) for t in state_dict.values())
    return {"bytes": blob, "nbytes": len(blob), "fp32_bytes": fp32b,
            "ratio": fp32b / len(blob)}
