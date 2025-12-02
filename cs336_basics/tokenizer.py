import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.id_to_bytes = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self._re = re.compile(PAT)

    def _split_by_special(self, text):
        if not self.special_tokens:
            return [(text, False)]

        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped = [re.escape(s) for s in sorted_tokens]
        pattern = "(" + "|".join(escaped) + ")"
        parts = re.split(pattern, text)

        out = []
        for p in parts:
            if not p:
                continue
            if p in self.special_tokens:
                out.append((p, True))
            else:
                out.append((p, False))
        return out

    def _pretokenize(self, text):
        out = []
        for m in self._re.finditer(text):
            tok = m.group(0)
            if tok:
                out.append(tok.encode("utf-8"))
        return out

    def _bpe(self, token_bytes):
        tokens = [bytes([b]) for b in token_bytes]

        while True:
            best_rank = float('inf')
            best_idx = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
            
            if best_idx == -1:
                break
            
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        return tokens

    def encode(self, text):
        ids = []
        segments = self._split_by_special(text)

        for seg, is_special in segments:
            if is_special:
                tok_bytes = seg.encode("utf-8")
                tok_id = self.bytes_to_id[tok_bytes]
                ids.append(tok_id)
                continue

            for pre_b in self._pretokenize(seg):
                bpe_tokens = self._bpe(pre_b)
                for t in bpe_tokens:
                    tid = self.bytes_to_id[t]
                    ids.append(tid)

        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    def decode(self, ids):
        bs = b"".join(self.id_to_bytes[i] for i in ids)
        return bs.decode("utf-8", errors="replace")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens):
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tid_str, hex_str = line.split("\t")
                tid = int(tid_str)
                vocab[tid] = bytes.fromhex(hex_str)

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h1, h2 = line.split(" ")
                merges.append((bytes.fromhex(h1), bytes.fromhex(h2)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
