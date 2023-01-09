import os
import io
import pathlib
import subprocess as sp

import torch
from colbert.infra.config import ColBERTConfig
from colbert.modeling.colbert import ColBERT
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.utils.utils import load_checkpoint


#PATH = os.environ['HOME'] + '/.cloudnote/'
PATH = "/home/joey/.colbert/"

CHECKPOINT = PATH + "colbertv2.0/"

# download weights if they're not here and we're not in AWS
if str(pathlib.Path().resolve())[:5] != '/var/' and not os.path.isdir(CHECKPOINT):
    print("Collecting ColBERT weights to", CHECKPOINT)

    sp.run(f"mkdir -p {PATH}".split(' '))
    sp.run(f"wget -O {PATH}colbert_weights.tar.gz https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz".split(' '))
    sp.run(f"tar -xvzf {PATH}colbert_weights.tar.gz -C {PATH}".split(' '))

config = None
colbert = None
entry_tokenizer = None
query_tokenizer = None

# check if anything is unitialized
def need_init():
    global config
    global colbert
    global entry_tokenizer
    global query_tokenizer

    return config is None or colbert is None or entry_tokenizer is None or query_tokenizer is None

# initialize ColBERT
# TODO: docstring
def initialize_colbert():
    global config
    global colbert
    global entry_tokenizer
    global query_tokenizer

    print("Initializing ColBERT...")

    if config is None:
        maxlen = 300
        config = ColBERTConfig.load_from_checkpoint(CHECKPOINT)
        config.query_maxlen = maxlen
        config.doc_maxlen = maxlen
        config.nway = 1
        config.interaction = "colbert"

    if entry_tokenizer is None:
        entry_tokenizer = DocTokenizer(config)

    if query_tokenizer is None:
        query_tokenizer = QueryTokenizer(config)

    if colbert is None:
        colbert = ColBERT(CHECKPOINT, colbert_config=config)

    print("ColBERT initialized")
    print("PARAMETER COUNT:", sum(p.numel() for p in colbert.parameters() if p.requires_grad))

def pad_tensor(tensor, length, pad_value):
    if tensor.size(1) >= length:
        return tensor

    shape = [x for x in tensor.size()]
    shape[1] = length - tensor.size(1)

    return torch.cat([tensor, torch.full(shape, pad_value)], axis=1)

# TODO: docstring
def create_entry_embedding(entry_text):
    global colbert
    global entry_tokenizer

    if need_init():
        initialize_colbert()

    entry_tensors = entry_tokenizer.tensorize([entry_text])
    entry_embedding, entry_mask = colbert.doc(*entry_tensors, keep_dims='return_mask')

    maxlen = query_tokenizer.query_maxlen
    mask_token_id = query_tokenizer.mask_token_id

    entry_embedding = pad_tensor(entry_embedding, maxlen, mask_token_id).detach()
    entry_mask = pad_tensor(entry_mask, maxlen, False).detach()

    return entry_embedding.float(), entry_mask.float()

def get_score_matrix(embeddings, masks):
    def stack_squeeze(tensor_list):
        tensor = torch.stack(tensor_list)
        tensor = torch.squeeze(tensor, dim=1)

        return tensor

    all_embeddings = stack_squeeze(embeddings)
    masks = stack_squeeze(masks)

    scores = [colbert.score_embeddings_against_query(em, all_embeddings, masks) for em in embeddings]

    return scores

def get_entry_mask_from_bytesio(embeddings, masks, maxlen, mask_token_id):
    """
    Get the PyTorch tensor representations of embeddings and masks

    Args:
        embeddings: List of io.BytesIO buffers representing the embeddings
        masks: List of io.BytesIO buffers representing the embeddings
        maxlen: Integer representing the max length of the entry
        max_token_id: Integer representing the ID of the mask token

    Returns:
        embeddings: PyTorch tensor representation of embeddings
        masks: PyTorch tensor representation of masks
    """

    embeddings = torch.stack([pad_tensor(torch.load(e), maxlen, mask_token_id) for e in embeddings])
    embeddings = torch.squeeze(embeddings, dim=1)

    if masks is not None:
        masks = torch.stack([pad_tensor(torch.load(m), maxlen, False) for m in masks])
        masks = torch.squeeze(masks, dim=1)

    return embeddings, masks

def query_entries(query, entries, k=10):
    """
    Search and return the k most similar entries to the query.

    Args:
        query: String representing user's query
        entries: [[ index, embedding , mask       ]] for each Entry row matching the user's user_id
                 [[ int  , io.BytesIO, io.BytesIO ]]

    Returns:
        List of indices corresponding to the ranked entries
    """

    global colbert
    global query_tokenizer

    if need_init():
        initialize_colbert()

    query_tensors = query_tokenizer.tensorize([query])

    maxlen = query_tokenizer.query_maxlen
    mask_token_id = query_tokenizer.mask_token_id

    embeddings = []
    masks = []
    for e in entries:
        embeddings.append(e[1])
        masks.append(e[2])

    embeddings, masks = get_entry_mask_from_bytesio(embeddings, masks, maxlen, mask_token_id)

    rankings = colbert.compare_query_to_entries(query_tensors, embeddings, masks)

    return rankings

def average_embeddings(embeddings, masks, maxlen, mask_token_id):
    """
    Get the average of the given embeddings and the longest mask.

    Args:
        embeddings: list of io.BytesIO objects representing the embeddings
        masks: list of io.BytesIO objects representing the masks

    Returns:
        [ average_embedding, mask ] where both are reduced axis=0
    """

    embeddings, masks = get_entry_mask_from_bytesio(embeddings, masks, maxlen, mask_token_id)

    embedding = torch.mean(embeddings, dim=0, keepdim=True)
    mask = torch.min(masks, dim=0, keepdim=True)

    return embedding, mask

def query_entries_with_embedding(query_embedding, entries, k=10):
    """
    Search and return the k most similar entries to the query embedding.

    Args:
        query: io.BytesIO buffer representing PyTorch tensor embedding of query
        entries: [[ index, embedding , mask       ]] for each Entry row matching the user's user_id
                 [[ int  , io.BytesIO, io.BytesIO ]]

    Returns:
        List of indices corresponding to the ranked entries
    """

    global colbert
    global query_tokenizer

    if need_init():
        initialize_colbert()

    query_tensor = pad_tensor(torch.load(query_embedding), maxlen, mask_token_id)

    maxlen = query_tokenizer.query_maxlen
    mask_token_id = query_tokenizer.mask_token_id

    embeddings = []
    masks = []
    for e in entries:
        embeddings.append(e[1])
        masks.append(e[2])

    embeddings, masks = get_entry_mask_from_bytesio(embeddings, masks, maxlen, mask_token_id)

    rankings = colbert.score_embeddings_against_query(query_tensor, embeddings, masks)

    return rankings

# TODO: reformat other entries -> { E_embeddings, E_masks }
def threshold_query(query, E_embeddings, E_masks, threshold=0.4):
    """
    Filter a list of entries such that their similarity(query, entry) / basis < threshold

    Args:
        query: io.BytesIO buffer
        E_embeddings: list of io.BytesIO buffers
        E_masks: list of io.BytesIO buffers
        threshold: float value := 0 < threshold <= 1

    Returns:
        List of indices such that [similarity(query, entries[i]) / basis < threshold for i in indices]
    """

    global colbert
    global query_tokenizer

    if need_init():
        initialize_colbert()

    maxlen = query_tokenizer.query_maxlen
    mask_token_id = query_tokenizer.mask_token_id

    query, _ = get_entry_mask_from_bytesio([query], None, maxlen, False)
    query = query.float()

    embeddings, masks = get_entry_mask_from_bytesio(E_embeddings, E_masks, maxlen, mask_token_id)
    embeddings = embeddings.float()
    masks = masks.float()

    scores = colbert.score_embeddings_against_query(query, embeddings, masks, False)

    # all scores are scaled relative to the query's relevance to itself
    # TODO: scaling relative to the most similar entry's score?
    basis_id, basis = scores[0]
    scores = [(i, score / basis) for i, score in scores[1:]]
    filtered_scores = [s[0] for s in scores if s[1] >= threshold and s[0] != basis_id]

    return filtered_scores

def rank_with_average(Q_embed, Q_masks, entries):
    """
    Average the query embeddings and rank the given entries against the query.

    Args:
        Q_embed: List of io.BytesIO buffers
        Q_masks: List of io.BytesIO buffers
        entries: [[ id , embedding , mask       ]] for each Entry row matching e.user_id == user_id and e.id != query.id
                 [[ int, io.BytesIO, io.BytesIO ]]

    Returns:
        List of indices corresponding to the rankings of the entries
    """

    global colbert
    global query_tokenizer

    if need_init():
        initialize_colbert()

    maxlen = query_tokenizer.query_maxlen
    mask_token_id = query_tokenizer.mask_token_id

    Q_embed, Q_mask = average_embeddings(Q_embed, Q_masks, maxlen, False)

    embeddings = []
    masks = []
    for e in entries:
        embeddings.append(e[1])
        masks.append(e[2])

    embeddings, masks = get_entry_mask_from_bytesio(embeddings, masks, maxlen, mask_token_id)

    embeddings = embeddings.float()
    masks = masks.float()
    Q_embed = Q_embed.float()

    rankings = colbert.score_embeddings_against_query(Q_embed, embeddings, masks)

    return rankings

def entry_similarity_ranking(query, entries, k=10):
    """
    Compare a single query entry to the given entries using cosine similarity.

    Args:
        query: io.BytesIO representing PyTorch tensor embedding
        entries: [[ id , embedding , mask       ]] for each Entry row matching e.user_id == user_id and e.id != query.id
                 [[ int, io.BytesIO, io.BytesIO ]]

    Returns:
        List of ids corresponding to the ranked entries
    """

    global colbert
    global query_tokenizer

    if need_init():
        initialize_colbert()


    def pad_load(tensor_bytes, maxlen=query_tokenizer.query_maxlen):
        return pad_tensor(torch.load(tensor_bytes), maxlen, False)

    query = pad_load(query)

    embeddings = torch.stack([pad_load(e[1]) for e in entries])
    embeddings = torch.squeeze(embeddings, dim=1)

    masks = torch.stack([pad_load(e[2]) for e in entries])
    masks = torch.squeeze(masks, dim=1)

    ids = [e[0] for e in entries]

    rankings = colbert.compare_entry_to_entries(query, embeddings, masks, ids)

    return rankings
