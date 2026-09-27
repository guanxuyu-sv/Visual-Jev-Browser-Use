"""Candidate symbols that are provably one token each, in the exact context they are read.

The plan forbids assuming `17` or `12:3` is a single token. Element indices and
select targets are therefore never shown to the readout directly: each candidate
gets a local symbol from a pool verified against this tokenizer, in the exact
string position where the logit is read. Verification is per (tokenizer, context),
because BPE merges depend on the preceding characters.
"""

import string

# Ordered by preference. Single ASCII characters are the likeliest single tokens,
# but nothing here is trusted until `verify_pool` has checked it.
_RAW_POOL = [*string.digits[1:], "0", *string.ascii_uppercase, *string.ascii_lowercase]


class SymbolPool:
    """A verified set of one-token symbols, plus the map back to real targets."""

    def __init__(self, tokenizer, context, symbols, token_ids):
        self.tokenizer = tokenizer
        self.context = context
        self.symbols = symbols
        self.token_ids = token_ids
        self._id_of = dict(zip(symbols, token_ids))

    def __len__(self):
        return len(self.symbols)

    def token_id(self, symbol):
        return self._id_of[symbol]

    def assign(self, targets, order=None):
        """Map real target keys (element index, or `12:3` for a select option) to symbols.

        `order` is a permutation applied before assignment. Training and evaluation
        randomise it so the model cannot learn a fixed slot shortcut (plan 4.3).
        Returns (symbol -> target, target -> symbol). Raises when K exceeds the pool:
        the caller must handle that explicitly rather than truncating silently.
        """
        targets = list(targets)
        if len(targets) > len(self.symbols):
            raise CandidateOverflow(
                f"{len(targets)} candidates exceed the {len(self.symbols)} verified symbols"
            )
        if order is not None:
            if sorted(order) != list(range(len(targets))):
                raise ValueError("order must be a permutation of the candidate positions")
            targets = [targets[i] for i in order]
        to_target = {self.symbols[i]: t for i, t in enumerate(targets)}
        return to_target, {t: s for s, t in to_target.items()}


class CandidateOverflow(ValueError):
    """More candidates than the verified symbol pool can encode."""


def verify_pool(tokenizer, context, pool=None, limit=None):
    """Keep only symbols that add exactly one token when appended to `context`.

    `context` is the literal string the readout ends with, e.g. "...\nAnswer:".
    A symbol survives only if encode(context + symbol) == encode(context) + [id],
    which is the only thing that makes reading a single next-token logit valid.
    """
    pool = _RAW_POOL if pool is None else pool
    base = tokenizer.encode(context, add_special_tokens=False)
    symbols, token_ids = [], []
    for symbol in pool:
        extended = tokenizer.encode(context + symbol, add_special_tokens=False)
        if len(extended) != len(base) + 1 or extended[: len(base)] != base:
            continue
        token_id = extended[-1]
        if token_id in token_ids:  # two symbols must not collide on one id
            continue
        symbols.append(symbol)
        token_ids.append(token_id)
        if limit is not None and len(symbols) >= limit:
            break
    if not symbols:
        raise RuntimeError(f"No single-token symbol survived verification for context {context!r}")
    return SymbolPool(tokenizer, context, symbols, token_ids)


def verify_words(tokenizer, context, words):
    """Verify fixed-vocabulary answers (DONE, CLICK, ...) that must also be one token.

    Returns (kept, dropped). Operations that are not single tokens cannot be read
    from one logit; the caller decides whether to rename them or fall back.
    """
    base = tokenizer.encode(context, add_special_tokens=False)
    kept, dropped = {}, []
    for word in words:
        extended = tokenizer.encode(context + word, add_special_tokens=False)
        if len(extended) == len(base) + 1 and extended[: len(base)] == base:
            kept[word] = extended[-1]
        else:
            dropped.append(word)
    return kept, dropped
