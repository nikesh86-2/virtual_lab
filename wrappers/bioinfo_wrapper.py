from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any

from Bio import Entrez, SeqIO
#from Bio.SeqRecord import SeqRecord

log = logging.getLogger("virtual_lab.bioinfo")


class BioinfoWrapper:
    """
    Wrapper around fetch / MSA / conservation analysis.
    Now includes:
    - Structured conservation outputs
    - Fitness score for optimisation
    - Motif-level conserved region detection
    """

    def __init__(self, email: str | None = None):
        self.email = email or os.getenv("ENTREZ_EMAIL", "virtual.lab@example.com")
        Entrez.email = self.email

        self.mafft = shutil.which("mafft")
        self.muscle = shutil.which("muscle")
        self.clustalo = shutil.which("clustalo")

    # =========================================================
    # FETCH GENOMES
    # =========================================================
    def fetch_related_genomes(self, txid: str, limit: int = 3) -> str:
        txid = str(txid).strip()
        if not txid.isdigit():
            return ""

        try:
            search_term = f"txid{txid}[Organism:exp] AND (genome OR complete genome OR complete sequence)"
            search = Entrez.esearch(db="nucleotide", term=search_term, retmax=max(1, limit))
            record = Entrez.read(search)

            ids = record.get("IdList", [])
            if not ids:
                return ""

            fetch = Entrez.efetch(
                db="nucleotide",
                id=",".join(ids[:limit]),
                rettype="fasta",
                retmode="text"
            )

            fasta = fetch.read().strip()
            if not fasta or ">" not in fasta:
                return ""

            return fasta

        except Exception as e:
            log.warning("NCBI fetch failed: %s", e)
            return ""

    # =========================================================
    # MSA
    # =========================================================
    def run_msa(self, fasta_text: str) -> str:
        fasta_text = (fasta_text or "").strip()
        if not fasta_text or ">" not in fasta_text:
            return ""

        records = self._parse_fasta_text(fasta_text)
        if len(records) < 2:
            return fasta_text

        with tempfile.TemporaryDirectory() as tmpdir:
            in_fa = os.path.join(tmpdir, "input.fasta")
            out_fa = os.path.join(tmpdir, "aligned.fasta")

            with open(in_fa, "w") as f:
                f.write(fasta_text + "\n")

            # try MAFFT first
            if self.mafft:
                try:
                    proc = subprocess.run([self.mafft, "--auto", in_fa],
                                          capture_output=True, text=True, timeout=180)
                    if proc.returncode == 0 and proc.stdout.strip():
                        return proc.stdout.strip()
                except Exception:
                    pass

            # fallback: MUSCLE
            if self.muscle:
                try:
                    proc = subprocess.run([self.muscle, "-align", in_fa, "-output", out_fa],
                                          capture_output=True, text=True, timeout=180)
                    if proc.returncode == 0 and os.path.exists(out_fa):
                        with open(out_fa) as f:
                            return f.read().strip()
                except Exception:
                    pass

            # fallback: Clustal Omega
            if self.clustalo:
                try:
                    proc = subprocess.run(
                        [self.clustalo, "-i", in_fa, "-o", out_fa, "--outfmt=fasta", "--force"],
                        capture_output=True, text=True, timeout=180
                    )
                    if proc.returncode == 0 and os.path.exists(out_fa):
                        with open(out_fa) as f:
                            return f.read().strip()
                except Exception:
                    pass

        return fasta_text

    # =========================================================
    # CONSERVATION ANALYSIS (UPGRADED)
    # =========================================================
    def calculate_conservation(self, msa_text: str) -> Dict[str, Any]:
        msa_text = (msa_text or "").strip()
        if not msa_text or ">" not in msa_text:
            return {"valid": False, "error": "Invalid MSA"}

        records = self._parse_fasta_text(msa_text)
        if not records:
            return {"valid": False, "error": "No sequences"}

        sequences = [str(r.seq).upper() for r in records]

        # check alignment integrity
        lengths = {len(seq) for seq in sequences}
        if len(lengths) != 1:
            return {
                "valid": False,
                "error": f"Unequal sequence lengths: {sorted(lengths)}"
            }

        aln_len = len(sequences[0])
        if aln_len == 0:
            return {"valid": False, "error": "Empty alignment"}

        # =====================================================
        # COLUMN ANALYSIS
        # =====================================================
        consensus_chars: List[str] = []
        fully_conserved = 0
        informative_cols = 0
        per_col_identity: List[float] = []

        for i in range(aln_len):
            column = [seq[i] for seq in sequences]
            nongap = [c for c in column if c != "-"]

            if not nongap:
                consensus_chars.append("-")
                continue

            informative_cols += 1

            counts = {}
            for c in nongap:
                counts[c] = counts.get(c, 0) + 1

            best_base, best_count = max(counts.items(), key=lambda x: x[1])
            consensus_chars.append(best_base)

            identity = best_count / len(nongap)
            per_col_identity.append(identity)

            if identity == 1.0 and len(nongap) == len(column):
                fully_conserved += 1

        consensus = "".join(consensus_chars)

        mean_identity = (
            sum(per_col_identity) / len(per_col_identity)
            if per_col_identity else 0.0
        )

        fully_conserved_pct = (fully_conserved / aln_len) * 100.0
        informative_pct = (informative_cols / aln_len) * 100.0

        # =====================================================
        # CONSERVED REGIONS (MOTIF LEVEL)
        # =====================================================
        conserved_regions = self._find_conserved_regions(sequences, window=6)

        # =====================================================
        # CONSERVATION FITNESS (KEY ADDITION)
        # =====================================================
        region_score = sum(end - start for start, end in conserved_regions)
        region_norm = region_score / aln_len if aln_len > 0 else 0

        conservation_fitness = (
            0.7 * mean_identity +
            0.3 * region_norm
        )

        # =====================================================
        # OUTPUT
        # =====================================================
        summary = (
            f"MSA sequences: {len(sequences)}\n"
            f"Alignment length: {aln_len}\n"
            f"Mean identity: {mean_identity * 100:.1f}%\n"
            f"Fully conserved: {fully_conserved_pct:.1f}%\n"
            f"Conserved regions: {len(conserved_regions)}\n"
            f"Consensus (120nt): {consensus[:120]}")
        return {
                "valid": True,
                "num_sequences": len(sequences),
                "alignment_length": aln_len,
                "mean_identity": round(mean_identity, 4),
                "fully_conserved_pct": round(fully_conserved_pct, 2),
                "informative_pct": round(informative_pct, 2),

                # ✅ NEW OPTIMISATION SIGNALS
                "conserved_regions": conserved_regions,
                "conservation_fitness": round(conservation_fitness, 4),

                # ✅ diagnostic info
                "consensus": consensus[:120],
                "summary": summary,}

    # =========================================================
    # CONSERVED REGION FINDER
    # =========================================================
    def _find_conserved_regions(self, sequences, window=6):
        regions = []
        length = len(sequences[0])

        for i in range(length - window):
            segment = [seq[i:i+window] for seq in sequences]

            if len(set(segment)) == 1:
                regions.append((i, i + window))

        return regions

    # =========================================================
    # FASTA PARSER
    # =========================================================
    def _parse_fasta_text(self, fasta_text: str) -> List:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "temp.fasta")

            try:
                with open(path, "w") as f:
                    f.write(fasta_text + "\n")

                return list(SeqIO.parse(path, "fasta"))

            except Exception as e:
                log.warning("FASTA parse failed: %s", e)
                return []
