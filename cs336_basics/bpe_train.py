import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_text_on_special(text, special_tokens):
    if not special_tokens:
        return [text]

    escaped = [re.escape(t) for t in special_tokens]
    pattern = "(" + "|".join(escaped) + ")"
    parts = re.split(pattern, text)

    result = []
    for p in parts:
        if not p:
            continue
        if p in special_tokens:
            continue
        result.append(p)

    return result


def _pretokenize(text):
    tokens = []
    for match in re.finditer(PAT, text):
        tok = match.group(0)
        if tok:
            tokens.append(tok.encode("utf-8"))
    return tokens


def _split_into_byte_tokens(b):
    return tuple(bytes([x]) for x in b)


def _compute_pair_stats(vocab_words):
    stats = defaultdict(int)

    for word, freq in vocab_words.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq

    return stats


def _merge_pair(pair, vocab_words):
    a, b = pair
    new_token = a + b
    new_vocab = {}

    for word, freq in vocab_words.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        
        new_word_tuple = tuple(new_word)
        if new_word_tuple in new_vocab:
            new_vocab[new_word_tuple] += freq
        else:
            new_vocab[new_word_tuple] = freq

    return new_vocab


def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = _split_text_on_special(text, special_tokens)

    vocab_words = Counter()

    for chunk in chunks:
        token_bytes_list = _pretokenize(chunk)

        for tb in token_bytes_list:
            byte_tokens = _split_into_byte_tokens(tb)
            vocab_words[byte_tokens] += 1

    vocab = {}
    next_id = 0

    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        vocab[next_id] = tok_bytes
        next_id += 1

    for x in range(256):
        vocab[next_id] = bytes([x])
        next_id += 1

    merges = []

    while len(vocab) < vocab_size:
        pair_stats = _compute_pair_stats(vocab_words)
        if not pair_stats:
            break

        max_freq = max(pair_stats.values())
        candidates = [p for p, c in pair_stats.items() if c == max_freq]

        best_pair = max(candidates)

        vocab_words = _merge_pair(best_pair, vocab_words)

        merges.append(best_pair)

        a, b = best_pair
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        if len(vocab) >= vocab_size:
            break

    return vocab, merges
