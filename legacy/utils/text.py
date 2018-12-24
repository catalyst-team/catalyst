import numpy as np

PAD_TOKEN = "_PAD_"  # 0
BOS_TOKEN = "_BOS_"  # 1
EOS_TOKEN = "_EOS_"  # 2
UNK_TOKEN = "_UNK_"  # 3


def load_vocab(filepath, default_tokens=None):
    default_tokens = (
            default_tokens or [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN])
    tokens = []
    with open(filepath) as fin:
        for line in fin:
            line = line.replace("\n", "")
            token = line.split()[0]
            tokens.append(token)

    tokens = default_tokens + list(sorted(tokens))
    token2id = {t: i for i, t in enumerate(tokens)}
    id2token = {i: t for i, t in enumerate(tokens)}
    return token2id, id2token


def create_line_encode_fn(
        token2id, sep=" ",
        max_len=None, lowercase=True,
        bos_token=True, eos_token=True,
        strip=True):
    token2id[sep] = token2id[PAD_TOKEN]

    if len(sep) > 0:
        split_fn = lambda x: (  # noqa: E731
                ([BOS_TOKEN] if bos_token else [])
                + (x.strip().split(sep) if strip else x.split(sep))
                + ([EOS_TOKEN] if eos_token else []))
    else:
        split_fn = lambda x: (  # noqa: E731
                ([BOS_TOKEN] if bos_token else [])
                + (list(x.strip()) if strip else list(x))
                + ([EOS_TOKEN] if eos_token else []))

    def line_encode_fn(line):
        if lowercase:
            line = line.lower()
        enc = np.array(list(map(
            lambda x: token2id.get(x, token2id[UNK_TOKEN]), split_fn(line))),
            dtype=np.int64)

        if max_len is not None:
            enc = enc[:max_len]

            result = np.ones(
                shape=(max_len,), dtype=np.int64) * int(token2id[PAD_TOKEN])
            result[:len(enc)] = enc
        else:
            result = enc

        return result

    return line_encode_fn


def create_line_decode_fn(id2token, sep=" ", strip=True):

    def line_decode_fn(line):
        sampled_caption = []
        for token_id in line:
            token = id2token[token_id]
            if token == BOS_TOKEN:
                continue
            # @TODO: remove unk_token with something?
            if token == PAD_TOKEN or token == UNK_TOKEN:
                token = " "
            if token == EOS_TOKEN:
                break
            sampled_caption.append(token)
        result = sep.join(sampled_caption)

        if strip:
            result = result.strip()

        return result

    return line_decode_fn


def l2_normalize(vec):
    return vec / np.linalg.norm(vec)


def create_fasttext_encode_fn(model_bin, normalize=False):
    from fastText import load_model
    m = load_model(model_bin)

    def line_encode_fn(line):
        line = line.replace("\n", "")
        result = m.get_sentence_vector(line)
        if normalize:
            result = l2_normalize(result)
        return result

    return line_encode_fn


def create_gensim_encode_fn(model_bin, sep=" ", normalize=False):
    import gensim
    try:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(
            model_bin, binary=True)
    except UnicodeDecodeError:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(
            model_bin, binary=False)
    w2v_mean = np.mean(w2v.vectors, axis=0)

    def line_encode_fn(line):
        line = line.replace("\n", "")
        vectors = []
        for word in line.split(sep):
            try:
                embedding = w2v[word]
                vectors.append(embedding)
            except KeyError:
                continue
        if len(vectors) > 0:
            result = np.mean(np.array(vectors), axis=0)
        else:
            result = w2v_mean.copy()
        if normalize:
            result = l2_normalize(result)
        return result

    return line_encode_fn
