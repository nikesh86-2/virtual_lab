"""
Grammar-based RNA generator

Produces foldable RNA sequences using structural rules:
- stem-loop (hairpins)
- bulges
- internal loops
- GC-biased stabilisation

Designed for NSGA-II integration
"""

import random


NUCLEOTIDES = ["A", "C", "G", "U"]

COMPLEMENT = {
    "A": "U",
    "U": "A",
    "C": "G",
    "G": "C"
}


# ----------------------------------------------------------
# BASIC BUILDING BLOCKS
# ----------------------------------------------------------

def random_base(gc_bias=0.5):
    if random.random() < gc_bias:
        return random.choice(["G", "C"])
    return random.choice(["A", "U"])


def complement(seq):
    return "".join(COMPLEMENT[b] for b in reversed(seq))


# ----------------------------------------------------------
# STEM GENERATOR
# ----------------------------------------------------------

def generate_stem(length, gc_bias=0.7):
    return "".join(random_base(gc_bias) for _ in range(length))


# ----------------------------------------------------------
# LOOP GENERATORS
# ----------------------------------------------------------

def generate_loop(length):
    return "".join(random.choice("AU") for _ in range(length))


def generate_bulge(max_length=3):
    length = random.randint(1, max_length)
    return "".join(random.choice("ACGU") for _ in range(length))


# ----------------------------------------------------------
# STRUCTURE TEMPLATES (GRAMMAR RULES)
# ----------------------------------------------------------

def hairpin(length=40):
    """
    Simple stem-loop
    """
    stem_len = random.randint(6, 10)
    loop_len = random.randint(4, 8)

    left = generate_stem(stem_len)
    loop = generate_loop(loop_len)
    right = complement(left)

    seq = left + loop + right

    return pad_sequence(seq, length)


def bulged_stem(length=40):
    """
    Stem with bulge (more realistic RNA)
    """
    stem_len = random.randint(6, 10)
    bulge_len = random.randint(1, 3)
    loop_len = random.randint(4, 8)

    left = generate_stem(stem_len)
    bulge = generate_bulge(bulge_len)
    loop = generate_loop(loop_len)
    right = complement(left)

    seq = left + bulge + loop + right

    return pad_sequence(seq, length)


def internal_loop(length=40):
    """
    Internal asymmetric loop
    """
    stem_len = random.randint(6, 10)
    loop1_len = random.randint(2, 5)
    loop2_len = random.randint(2, 5)

    left = generate_stem(stem_len)
    loop1 = generate_loop(loop1_len)
    loop2 = generate_loop(loop2_len)
    right = complement(left)

    seq = left[:3] + loop1 + left[3:] + loop2 + right

    return pad_sequence(seq, length)


def multi_branch(length=40):
    """
    Multi-branch motif (more complex folding)
    """
    stemA = generate_stem(6)
    stemB = generate_stem(6)
    stemC = generate_stem(6)

    loopA = generate_loop(4)
    loopB = generate_loop(4)
    loopC = generate_loop(4)

    seq = (
        stemA +
        loopA +
        complement(stemB) +
        loopB +
        complement(stemC) +
        loopC +
        complement(stemA)
    )

    return pad_sequence(seq, length)


# ----------------------------------------------------------
# GRAMMAR DRIVER
# ----------------------------------------------------------

def generate_structured_rna(length=40):
    """
    Randomly choose a grammar rule
    """
    generators = [
        hairpin,
        bulged_stem,
        internal_loop,
        multi_branch
    ]

    gen = random.choice(generators)
    return gen(length)


# ----------------------------------------------------------
# MUTATION (VERY IMPORTANT FOR NSGA)
# ----------------------------------------------------------

def mutate_sequence(seq, mutation_rate=0.05):
    seq = list(seq)

    for i in range(len(seq)):
        if random.random() < mutation_rate:
            seq[i] = random.choice(NUCLEOTIDES)

    return "".join(seq)


def structure_preserving_mutation(seq):
    """
    Mutate while preserving base-pair complementarity
    """
    seq = list(seq)
    n = len(seq)

    for i in range(n // 2):
        if random.random() < 0.05:
            base = random.choice(["G", "C", "A", "U"])
            seq[i] = base
            seq[n - i - 1] = COMPLEMENT[base]

    return "".join(seq)


# ----------------------------------------------------------
# UTILS
# ----------------------------------------------------------

def pad_sequence(seq, target_length):
    if len(seq) > target_length:
        return seq[:target_length]

    while len(seq) < target_length:
        seq += random.choice(NUCLEOTIDES)

    return seq


# ----------------------------------------------------------
# POPULATION INITIALISER (FOR NSGA)
# ----------------------------------------------------------

def generate_population(size=20, length=40):
    return [generate_structured_rna(length) for _ in range(size)]


# ----------------------------------------------------------
# DEBUG / TEST
# ----------------------------------------------------------

if __name__ == "__main__":
    for _ in range(5):
        s = generate_structured_rna(40)
        print(s)