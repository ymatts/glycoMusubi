#!/usr/bin/env python3
"""Comprehensive list of human glycosylation enzyme genes.

Compiled from:
  - GlycoEnzOnto (Neelamegham et al. 2022, Bioinformatics 38:5413)
    https://github.com/neel-lab/GlycoEnzOnto  (403 glycogenes)
  - CAZy database glycosyltransferase families (cazy.org)
  - Essentials of Glycobiology, 4th ed (NCBI Bookshelf)
  - KEGG glycan biosynthesis pathways
  - IUPHAR/BPS Guide to Pharmacology (SLC35 family)

Organised by biosynthetic pathway for querying scRNA-seq data
(e.g. Tabula Sapiens via CZ CELLxGENE Census API).

Usage:
    from scripts.glycosylation_genes import GLYCO_GENES, ALL_GENE_SYMBOLS
    # GLYCO_GENES: dict  gene_symbol -> pathway_category
    # ALL_GENE_SYMBOLS: sorted list of all unique gene symbols
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# N-linked glycosylation pathway
# ---------------------------------------------------------------------------

# Dolichol-linked oligosaccharide (LLO) biosynthesis — ER
_NLINKED_LLO = {
    "DPAGT1": "nlinked_llo",       # GlcNAc-1-P transferase (ALG7 activity)
    "ALG1": "nlinked_llo",         # beta-1,4-mannosyltransferase
    "ALG2": "nlinked_llo",         # alpha-1,3/1,6-mannosyltransferase
    "ALG11": "nlinked_llo",        # alpha-1,2-mannosyltransferase
    "RFT1": "nlinked_llo",         # flippase (Man5GlcNAc2 -> lumenal)
    "ALG3": "nlinked_llo",         # alpha-1,3-mannosyltransferase
    "ALG9": "nlinked_llo",         # alpha-1,2-mannosyltransferase
    "ALG12": "nlinked_llo",        # alpha-1,6-mannosyltransferase
    "ALG5": "nlinked_llo",         # dolichyl-P Glc synthase
    "ALG6": "nlinked_llo",         # alpha-1,3-glucosyltransferase
    "ALG8": "nlinked_llo",         # alpha-1,3-glucosyltransferase
    "ALG10": "nlinked_llo",        # alpha-1,2-glucosyltransferase (ALG10B)
    "ALG10B": "nlinked_llo",       # alpha-1,2-glucosyltransferase paralog
    "ALG13": "nlinked_llo",        # UDP-GlcNAc transferase subunit
    "ALG14": "nlinked_llo",        # UDP-GlcNAc transferase subunit
    "DOLK": "nlinked_llo",         # dolichol kinase
    "DPM1": "nlinked_llo",         # dolichol-phosphate mannosyltransferase subunit 1
    "DPM2": "nlinked_llo",         # dolichol-phosphate mannosyltransferase subunit 2
    "DPM3": "nlinked_llo",         # dolichol-phosphate mannosyltransferase subunit 3
    "MPDU1": "nlinked_llo",        # mannose-P-dolichol utilisation defect 1
    "SRD5A3": "nlinked_llo",       # polyprenol reductase
}

# Oligosaccharyltransferase (OST) complex
_NLINKED_OST = {
    "STT3A": "nlinked_ost",        # catalytic subunit A
    "STT3B": "nlinked_ost",        # catalytic subunit B
    "RPN1": "nlinked_ost",         # ribophorin I
    "RPN2": "nlinked_ost",         # ribophorin II
    "DDOST": "nlinked_ost",        # OST48
    "DAD1": "nlinked_ost",         # defender against cell death 1
    "TUSC3": "nlinked_ost",        # N33 / OST3
    "MAGT1": "nlinked_ost",        # IAP / OST3 paralog
    "OSTC": "nlinked_ost",         # OST complex subunit
    "KRTCAP2": "nlinked_ost",      # OST complex subunit
}

# ER quality control / glucose trimming
_NLINKED_ER_QC = {
    "MOGS": "nlinked_er_processing",      # alpha-glucosidase I
    "GANAB": "nlinked_er_processing",      # alpha-glucosidase II alpha subunit
    "PRKCSH": "nlinked_er_processing",     # alpha-glucosidase II beta subunit
    "UGGT1": "nlinked_er_processing",      # UDP-Glc:glycoprotein glucosyltransferase 1
    "UGGT2": "nlinked_er_processing",      # UDP-Glc:glycoprotein glucosyltransferase 2
    "MAN1B1": "nlinked_er_processing",     # ER alpha-1,2-mannosidase I
    "EDEM1": "nlinked_er_processing",      # ER degradation enhancer mannosidase alpha-like 1
    "EDEM2": "nlinked_er_processing",      # ER degradation enhancer mannosidase alpha-like 2
    "EDEM3": "nlinked_er_processing",      # ER degradation enhancer mannosidase alpha-like 3
}

# Golgi N-glycan processing (mannosidases, GnT, galactosyltransferases, etc.)
_NLINKED_GOLGI = {
    # alpha-1,2-mannosidases (Golgi)
    "MAN1A1": "nlinked_golgi_processing",  # Golgi alpha-mannosidase IA
    "MAN1A2": "nlinked_golgi_processing",  # Golgi alpha-mannosidase IB
    "MAN1C1": "nlinked_golgi_processing",  # Golgi alpha-mannosidase IC
    # alpha-mannosidase II
    "MAN2A1": "nlinked_golgi_processing",  # alpha-mannosidase II
    "MAN2A2": "nlinked_golgi_processing",  # alpha-mannosidase IIx
    # N-acetylglucosaminyltransferases (GnT / MGAT)
    "MGAT1": "nlinked_golgi_processing",   # GnT-I
    "MGAT2": "nlinked_golgi_processing",   # GnT-II
    "MGAT3": "nlinked_golgi_processing",   # GnT-III (bisecting GlcNAc)
    "MGAT4A": "nlinked_golgi_processing",  # GnT-IVa
    "MGAT4B": "nlinked_golgi_processing",  # GnT-IVb
    "MGAT4C": "nlinked_golgi_processing",  # GnT-IVc
    "MGAT5": "nlinked_golgi_processing",   # GnT-V
    "MGAT5B": "nlinked_golgi_processing",  # GnT-Vb / GnT-IX
    # beta-1,4-galactosyltransferases (N-glycan branch elongation)
    "B4GALT1": "nlinked_golgi_processing", # beta4GalT-I
    "B4GALT2": "nlinked_golgi_processing", # beta4GalT-II
    "B4GALT3": "nlinked_golgi_processing", # beta4GalT-III
    "B4GALT4": "nlinked_golgi_processing", # beta4GalT-IV
    # Poly-N-acetyllactosamine extension
    "B3GNT2": "nlinked_golgi_processing",  # i-branching / poly-LacNAc
    "B3GNT8": "nlinked_golgi_processing",  # poly-LacNAc
    # Core fucosylation
    "FUT8": "nlinked_golgi_processing",    # alpha-1,6-fucosyltransferase
}

# Terminal capping of N-glycans (sialylation, fucosylation, etc.)
_NLINKED_CAPPING = {
    # Sialyltransferases acting on N-glycans
    "ST6GAL1": "nlinked_capping",   # alpha-2,6-sialyltransferase (main N-glycan ST)
    "ST6GAL2": "nlinked_capping",   # alpha-2,6-sialyltransferase 2
    "ST3GAL3": "nlinked_capping",   # alpha-2,3-sialyltransferase (N-glycans)
    "ST3GAL4": "nlinked_capping",   # alpha-2,3-sialyltransferase (N-glycans)
    "ST3GAL6": "nlinked_capping",   # alpha-2,3-sialyltransferase
    # Lewis / blood group fucosyltransferases (also act on O-glycans)
    "FUT1": "nlinked_capping",      # alpha-1,2-fucosyltransferase (H antigen)
    "FUT2": "nlinked_capping",      # alpha-1,2-fucosyltransferase (secretor)
    "FUT3": "nlinked_capping",      # alpha-1,3/4-fucosyltransferase (Lewis)
    "FUT4": "nlinked_capping",      # alpha-1,3-fucosyltransferase
    "FUT5": "nlinked_capping",      # alpha-1,3-fucosyltransferase
    "FUT6": "nlinked_capping",      # alpha-1,3-fucosyltransferase
    "FUT7": "nlinked_capping",      # alpha-1,3-fucosyltransferase (sLeX)
    "FUT9": "nlinked_capping",      # alpha-1,3-fucosyltransferase (LeX)
    "FUT10": "nlinked_capping",     # alpha-1,3-fucosyltransferase / POFUT3
    "FUT11": "nlinked_capping",     # alpha-1,3-fucosyltransferase / POFUT4
}

# ---------------------------------------------------------------------------
# O-linked glycosylation pathway (mucin-type O-GalNAc)
# ---------------------------------------------------------------------------

# Initiating enzymes: polypeptide GalNAc-transferases
_OLINKED_GALNT = {
    "GALNT1": "olinked_initiation",
    "GALNT2": "olinked_initiation",
    "GALNT3": "olinked_initiation",
    "GALNT4": "olinked_initiation",
    "GALNT5": "olinked_initiation",
    "GALNT6": "olinked_initiation",
    "GALNT7": "olinked_initiation",
    "GALNT8": "olinked_initiation",
    "GALNT9": "olinked_initiation",
    "GALNT10": "olinked_initiation",
    "GALNT11": "olinked_initiation",
    "GALNT12": "olinked_initiation",
    "GALNT13": "olinked_initiation",
    "GALNT14": "olinked_initiation",
    "GALNT15": "olinked_initiation",
    "GALNT16": "olinked_initiation",
    "GALNT17": "olinked_initiation",
    "GALNT18": "olinked_initiation",
    "GALNT19": "olinked_initiation",
    "GALNTL5": "olinked_initiation",  # polypeptide GalNAc-T like 5 (prev. GALNT20)
}

# Core structures and extension
_OLINKED_CORE = {
    "C1GALT1": "olinked_core",      # core 1 synthase (T-synthase)
    "C1GALT1C1": "olinked_core",    # COSMC chaperone for C1GALT1
    "GCNT1": "olinked_core",        # core 2 beta-1,6-GlcNAc-transferase
    "GCNT2": "olinked_core",        # I-branching enzyme / core 2 GlcNAcT
    "GCNT3": "olinked_core",        # core 2/4 GlcNAc-transferase
    "GCNT4": "olinked_core",        # core 2 GlcNAc-transferase
    "B3GNT3": "olinked_core",       # core 1 extension / poly-LacNAc
    "B3GNT6": "olinked_core",       # core 3 synthase
    # Sialyltransferases on O-glycans
    "ST3GAL1": "olinked_capping",   # alpha-2,3-sialylation of core 1
    "ST3GAL2": "olinked_capping",   # alpha-2,3-sialylation of core 1
    "ST6GALNAC1": "olinked_capping",# alpha-2,6-sialylation of Tn antigen
    "ST6GALNAC2": "olinked_capping",# alpha-2,6-sialylation
    "ST6GALNAC3": "olinked_capping",# alpha-2,6-sialylation
    "ST6GALNAC4": "olinked_capping",# alpha-2,6-sialylation
    "ST6GALNAC5": "olinked_capping",# alpha-2,6-sialylation (brain)
    "ST6GALNAC6": "olinked_capping",# alpha-2,6-sialylation
}

# ---------------------------------------------------------------------------
# O-mannose / dystroglycan glycosylation pathway
# ---------------------------------------------------------------------------

_OMANNOSE = {
    "POMT1": "omannose",           # protein O-mannosyltransferase 1
    "POMT2": "omannose",           # protein O-mannosyltransferase 2
    "POMGNT1": "omannose",         # O-Man beta-1,2-GlcNAc transferase
    "POMGNT2": "omannose",         # O-Man beta-1,4-GlcNAc transferase (GTDC2)
    "POMK": "omannose",            # protein O-mannose kinase
    "B3GALNT2": "omannose",        # beta-1,3-GalNAc transferase (core M3)
    "B4GAT1": "omannose",          # beta-1,4-glucuronyltransferase
    "LARGE1": "omannose",          # matriglycan synthase (LARGE)
    "LARGE2": "omannose",          # matriglycan synthase paralog
    "FKTN": "omannose",            # fukutin (ribitol-P transferase)
    "FKRP": "omannose",            # fukutin-related protein
    "CRPPA": "omannose",            # CDP-L-ribitol pyrophosphorylase A (prev. ISPD)
    "RXYLT1": "omannose",          # ribitol-xylosyltransferase 1 (=TMEM5)
}

# ---------------------------------------------------------------------------
# O-fucose / O-glucose (Notch pathway)
# ---------------------------------------------------------------------------

_OFUCOSE_OGLUCOSE = {
    "POFUT1": "ofucose",           # protein O-fucosyltransferase 1 (EGF repeats)
    "POFUT2": "ofucose",           # protein O-fucosyltransferase 2 (TSR repeats)
    "LFNG": "ofucose",             # Lunatic Fringe (beta-1,3-GlcNAcT)
    "MFNG": "ofucose",             # Manic Fringe
    "RFNG": "ofucose",             # Radical Fringe
    "B3GLCT": "oglucose",          # beta-1,3-glucosyltransferase (O-Glc on EGF)
    "EOGT": "oglucose",            # EGF domain-specific O-GlcNAc transferase
}

# ---------------------------------------------------------------------------
# O-GlcNAc cycling (intracellular, nutrient-sensing)
# ---------------------------------------------------------------------------

_OGLCNAC = {
    "OGT": "oglcnac",              # O-GlcNAc transferase
    "OGA": "oglcnac",              # O-GlcNAcase (prev. MGEA5)
}

# ---------------------------------------------------------------------------
# Glycosaminoglycan (GAG) biosynthesis
# ---------------------------------------------------------------------------

# Linker tetrasaccharide (shared by HS, CS, DS)
_GAG_LINKER = {
    "XYLT1": "gag_linker",        # xylosyltransferase 1
    "XYLT2": "gag_linker",        # xylosyltransferase 2
    "B4GALT7": "gag_linker",      # galactosyltransferase I (linker)
    "B3GALT6": "gag_linker",      # galactosyltransferase II (linker)
    "B3GAT1": "gag_linker",       # glucuronosyltransferase I
    "B3GAT2": "gag_linker",       # glucuronosyltransferase I paralog
    "B3GAT3": "gag_linker",       # glucuronosyltransferase I (linker)
    "FAM20B": "gag_linker",       # xylose kinase (linker phosphorylation)
    "PXYLP1": "gag_linker",       # 2-phosphoxylose phosphatase
}

# Heparan sulfate (HS) polymerisation and modification
_GAG_HS = {
    "EXT1": "gag_hs",             # HS polymerase (GlcA/GlcNAc)
    "EXT2": "gag_hs",             # HS polymerase
    "EXTL1": "gag_hs",            # exostosin-like 1
    "EXTL2": "gag_hs",            # exostosin-like 2
    "EXTL3": "gag_hs",            # exostosin-like 3 (GlcNAc-T initiation)
    "NDST1": "gag_hs",            # N-deacetylase/N-sulfotransferase 1
    "NDST2": "gag_hs",            # N-deacetylase/N-sulfotransferase 2
    "NDST3": "gag_hs",            # N-deacetylase/N-sulfotransferase 3
    "NDST4": "gag_hs",            # N-deacetylase/N-sulfotransferase 4
    "GLCE": "gag_hs",             # glucuronyl C5-epimerase
    "HS2ST1": "gag_hs",           # heparan sulfate 2-O-sulfotransferase
    "HS6ST1": "gag_hs",           # heparan sulfate 6-O-sulfotransferase 1
    "HS6ST2": "gag_hs",           # heparan sulfate 6-O-sulfotransferase 2
    "HS6ST3": "gag_hs",           # heparan sulfate 6-O-sulfotransferase 3
    "HS3ST1": "gag_hs",           # heparan sulfate 3-O-sulfotransferase 1
    "HS3ST2": "gag_hs",           # heparan sulfate 3-O-sulfotransferase 2
    "HS3ST3A1": "gag_hs",         # heparan sulfate 3-O-sulfotransferase 3A1
    "HS3ST3B1": "gag_hs",         # heparan sulfate 3-O-sulfotransferase 3B1
    "HS3ST4": "gag_hs",           # heparan sulfate 3-O-sulfotransferase 4
    "HS3ST5": "gag_hs",           # heparan sulfate 3-O-sulfotransferase 5
    "HS3ST6": "gag_hs",           # heparan sulfate 3-O-sulfotransferase 6
    "SULF1": "gag_hs",            # sulfatase 1 (6-O-desulfation)
    "SULF2": "gag_hs",            # sulfatase 2 (6-O-desulfation)
}

# Chondroitin sulfate (CS) / Dermatan sulfate (DS)
_GAG_CS_DS = {
    "CSGALNACT1": "gag_cs",       # CS GalNAc transferase 1
    "CSGALNACT2": "gag_cs",       # CS GalNAc transferase 2
    "CHSY1": "gag_cs",            # chondroitin sulfate synthase 1
    "CHSY3": "gag_cs",            # chondroitin sulfate synthase 3
    "CHPF": "gag_cs",             # chondroitin polymerising factor
    "CHPF2": "gag_cs",            # chondroitin polymerising factor 2
    "DSE": "gag_cs",              # dermatan sulfate epimerase
    "DSEL": "gag_cs",             # dermatan sulfate epimerase-like
    "CHST3": "gag_cs",            # carbohydrate 6-O-sulfotransferase (CS)
    "CHST7": "gag_cs",            # carbohydrate 6-O-sulfotransferase
    "CHST11": "gag_cs",           # carbohydrate 4-O-sulfotransferase 11
    "CHST12": "gag_cs",           # carbohydrate 4-O-sulfotransferase 12
    "CHST13": "gag_cs",           # carbohydrate 4-O-sulfotransferase 13
    "CHST14": "gag_cs",           # dermatan 4-O-sulfotransferase
    "CHST15": "gag_cs",           # GalNAc4S-6-O-sulfotransferase
    "UST": "gag_cs",              # uronyl 2-sulfotransferase
}

# Keratan sulfate (KS)
_GAG_KS = {
    "CHST1": "gag_ks",            # keratan sulfate Gal-6-O-sulfotransferase
    "CHST2": "gag_ks",            # GlcNAc-6-O-sulfotransferase (KS)
    "CHST4": "gag_ks",            # GlcNAc-6-O-sulfotransferase (HEV)
    "CHST5": "gag_ks",            # GlcNAc-6-O-sulfotransferase (intestinal)
    "CHST6": "gag_ks",            # corneal N-acetylglucosamine-6-O-sulfotransferase
    "B3GNT7": "gag_ks",           # KS chain extension
}

# Hyaluronan
_GAG_HA = {
    "HAS1": "gag_ha",             # hyaluronan synthase 1
    "HAS2": "gag_ha",             # hyaluronan synthase 2
    "HAS3": "gag_ha",             # hyaluronan synthase 3
}

# ---------------------------------------------------------------------------
# Glycosphingolipid (GSL) / glycolipid biosynthesis
# ---------------------------------------------------------------------------

_GLYCOLIPID = {
    "UGCG": "glycolipid",         # glucosylceramide synthase
    "B4GALT5": "glycolipid",      # lactosylceramide synthase
    "B4GALT6": "glycolipid",      # lactosylceramide synthase paralog
    "ST3GAL5": "glycolipid",      # GM3 synthase (alpha-2,3-sialyltransferase)
    "ST8SIA1": "glycolipid",      # GD3 synthase
    "ST8SIA3": "glycolipid",      # alpha-2,8-sialyltransferase III
    "ST8SIA5": "glycolipid",      # GT3 synthase
    "B4GALNT1": "glycolipid",     # GA2/GM2/GD2/GT2 synthase
    "B3GALT4": "glycolipid",      # GA1/GM1 synthase
    "B3GALNT1": "glycolipid",     # Gb4 synthase / globoside
    "A4GALT": "glycolipid",       # Gb3 synthase / Pk antigen
    "A4GNT": "glycolipid",        # alpha-1,4-GlcNAc transferase
    "ST8SIA2": "glycolipid",      # polysialyltransferase (PST, neuronal)
    "ST8SIA4": "glycolipid",      # polysialyltransferase (STX)
    "ST8SIA6": "glycolipid",      # alpha-2,8-sialyltransferase VI
    "B3GNT5": "glycolipid",       # lacto/neolacto GSL extension (Lc3)
    "FUT1": "glycolipid",         # H antigen on glycolipids (also in nlinked_capping)
    "FUT2": "glycolipid",         # secretor FUT (also in nlinked_capping)
}

# ---------------------------------------------------------------------------
# GPI-anchor biosynthesis
# ---------------------------------------------------------------------------

_GPI_ANCHOR = {
    "PIGA": "gpi_anchor",         # GPI-GlcNAc transferase subunit
    "PIGB": "gpi_anchor",         # mannosyltransferase III
    "PIGC": "gpi_anchor",         # GPI-GlcNAc transferase subunit
    "PIGH": "gpi_anchor",         # GPI-GlcNAc transferase subunit
    "PIGP": "gpi_anchor",         # GPI-GlcNAc transferase subunit
    "PIGQ": "gpi_anchor",         # GPI-GlcNAc transferase subunit
    "PIGY": "gpi_anchor",         # GPI-GlcNAc transferase subunit
    "PIGL": "gpi_anchor",         # GlcNAc-PI de-N-acetylase
    "PIGM": "gpi_anchor",         # mannosyltransferase I
    "PIGV": "gpi_anchor",         # mannosyltransferase II
    "PIGN": "gpi_anchor",         # ethanolamine-P transferase 1
    "PIGO": "gpi_anchor",         # ethanolamine-P transferase 3
    "PIGF": "gpi_anchor",         # ethanolamine-P transferase factor
    "PIGG": "gpi_anchor",         # GPI ethanolamine-P transferase 2
    "PIGW": "gpi_anchor",         # acyltransferase
    "PIGX": "gpi_anchor",         # PIGM stabiliser
    "PIGK": "gpi_anchor",         # GPI transamidase subunit
    "PIGS": "gpi_anchor",         # GPI transamidase subunit
    "PIGT": "gpi_anchor",         # GPI transamidase subunit
    "PIGU": "gpi_anchor",         # GPI transamidase subunit
    "GPAA1": "gpi_anchor",        # GPI anchor attachment protein 1
    "PGAP1": "gpi_anchor",        # post-GPI attachment to protein 1 (inositol deacylase)
    "PGAP2": "gpi_anchor",        # post-GPI attachment to protein 2
    "PGAP3": "gpi_anchor",        # post-GPI attachment to protein 3
    "PGAP4": "gpi_anchor",        # post-GPI attachment to protein 4
    "MINAR2": "gpi_anchor",       # post-GPI attachment to protein 5 (prev. PGAP5)
    "PGAP6": "gpi_anchor",        # post-GPI attachment to protein 6
}

# ---------------------------------------------------------------------------
# Sugar nucleotide biosynthesis
# ---------------------------------------------------------------------------

_SUGAR_NUCLEOTIDE = {
    # UDP-Glc / UDP-Gal pathway
    "UGP2": "sugar_nucleotide",    # UDP-glucose pyrophosphorylase
    "GALE": "sugar_nucleotide",    # UDP-galactose 4-epimerase
    "GALT": "sugar_nucleotide",    # galactose-1-phosphate uridylyltransferase
    "GALK1": "sugar_nucleotide",   # galactokinase 1
    "GALK2": "sugar_nucleotide",   # galactokinase 2 (GalNAc kinase)
    # UDP-GlcNAc / UDP-GalNAc pathway
    "GNPNAT1": "sugar_nucleotide", # glucosamine-6-P N-acetyltransferase
    "PGM3": "sugar_nucleotide",    # phosphoglucomutase 3 (GlcNAc pathway)
    "UAP1": "sugar_nucleotide",    # UDP-GlcNAc pyrophosphorylase
    # GDP-Man / GDP-Fuc pathway
    "MPI": "sugar_nucleotide",     # mannose-6-P isomerase
    "PMM2": "sugar_nucleotide",    # phosphomannomutase 2
    "GMPPB": "sugar_nucleotide",   # GDP-mannose pyrophosphorylase B
    "GMPPA": "sugar_nucleotide",   # GDP-mannose pyrophosphorylase A
    "GMDS": "sugar_nucleotide",    # GDP-mannose 4,6-dehydratase (GDP-Fuc)
    "SDR39U1": "sugar_nucleotide", # GDP-L-fucose synthetase (FX; prev. TSTA3)
    "FCSK": "sugar_nucleotide",    # fucokinase (salvage pathway; prev. FUK)
    "FPGT": "sugar_nucleotide",    # fucose-1-P guanylyltransferase (salvage)
    # CMP-Neu5Ac pathway
    "GNE": "sugar_nucleotide",     # UDP-GlcNAc 2-epimerase/ManNAc kinase
    "NANS": "sugar_nucleotide",    # Neu5Ac-9-P synthase
    "NANP": "sugar_nucleotide",    # Neu5Ac-9-P phosphatase
    "CMAS": "sugar_nucleotide",    # CMP-sialic acid synthetase
    # UDP-GlcA / UDP-Xyl pathway
    "UGDH": "sugar_nucleotide",    # UDP-glucose 6-dehydrogenase
    "UXS1": "sugar_nucleotide",    # UDP-glucuronate decarboxylase
}

# ---------------------------------------------------------------------------
# Sugar nucleotide transporters (SLC35 family)
# ---------------------------------------------------------------------------

_SLC35_TRANSPORTERS = {
    # Subfamily A
    "SLC35A1": "slc35_transporter", # CMP-sialic acid transporter
    "SLC35A2": "slc35_transporter", # UDP-galactose transporter
    "SLC35A3": "slc35_transporter", # UDP-GlcNAc transporter
    "SLC35A4": "slc35_transporter", # putative nucleotide sugar transporter
    "SLC35A5": "slc35_transporter", # putative nucleotide sugar transporter
    # Subfamily B
    "SLC35B1": "slc35_transporter", # putative nucleotide sugar transporter
    "SLC35B2": "slc35_transporter", # PAPS transporter (3'-phosphoadenosine 5'-phosphosulfate)
    "SLC35B3": "slc35_transporter", # PAPS transporter
    "SLC35B4": "slc35_transporter", # UDP-Xyl / UDP-GlcNAc transporter
    # Subfamily C
    "SLC35C1": "slc35_transporter", # GDP-fucose transporter
    "SLC35C2": "slc35_transporter", # putative transporter
    # Subfamily D
    "SLC35D1": "slc35_transporter", # UDP-GlcA / UDP-GalNAc transporter
    "SLC35D2": "slc35_transporter", # UDP-GlcNAc transporter
    "SLC35D3": "slc35_transporter", # putative transporter
    # Subfamily E (orphan)
    "SLC35E1": "slc35_transporter",
    "SLC35E2A": "slc35_transporter",
    "SLC35E2B": "slc35_transporter",
    "SLC35E3": "slc35_transporter",
    "SLC35E4": "slc35_transporter",
    # Subfamily F
    "SLC35F1": "slc35_transporter",
    "SLC35F2": "slc35_transporter",
    "SLC35F3": "slc35_transporter",
    "SLC35F4": "slc35_transporter",
    "SLC35F5": "slc35_transporter",
    "SLC35F6": "slc35_transporter",
    # Subfamily G
    "SLC35G1": "slc35_transporter",
    "SLC35G2": "slc35_transporter",
    "SLC35G3": "slc35_transporter",
    "SLC35G4": "slc35_transporter",
    "SLC35G5": "slc35_transporter",
    "SLC35G6": "slc35_transporter",
}

# ---------------------------------------------------------------------------
# Glycan degradation / glycosidases
# ---------------------------------------------------------------------------

_GLYCOSIDASES = {
    # Lysosomal glycosidases
    "HEXA": "glycosidase",         # beta-hexosaminidase alpha (Tay-Sachs)
    "HEXB": "glycosidase",         # beta-hexosaminidase beta (Sandhoff)
    "MAN2B1": "glycosidase",       # lysosomal alpha-mannosidase
    "MAN2B2": "glycosidase",       # epididymis-specific alpha-mannosidase
    "MANBA": "glycosidase",        # lysosomal beta-mannosidase
    "FUCA1": "glycosidase",        # lysosomal alpha-fucosidase
    "FUCA2": "glycosidase",        # plasma alpha-fucosidase
    "GLB1": "glycosidase",         # beta-galactosidase
    "GLA": "glycosidase",          # alpha-galactosidase A (Fabry disease)
    "NAGA": "glycosidase",         # alpha-N-acetylgalactosaminidase
    "NEU1": "glycosidase",         # sialidase 1 (lysosomal)
    "NEU2": "glycosidase",         # sialidase 2 (cytosolic)
    "NEU3": "glycosidase",         # sialidase 3 (plasma membrane)
    "NEU4": "glycosidase",         # sialidase 4
    "AGA": "glycosidase",          # aspartylglucosaminidase
    "ASAH1": "glycosidase",        # acid ceramidase (GSL catabolism)
    "GALC": "glycosidase",         # galactosylceramidase (Krabbe)
    "GBA": "glycosidase",          # glucocerebrosidase (Gaucher)
    "GBA2": "glycosidase",         # non-lysosomal glucosylceramidase
    # ER/Golgi glycosidases (also listed above but here for completeness)
    "NGLY1": "glycosidase",        # N-glycanase 1 (PNGase)
    # Heparanase
    "HPSE": "glycosidase",         # heparanase (HS degradation)
    "HPSE2": "glycosidase",        # heparanase 2 (inactive)
    # Hyaluronidases
    "HYAL1": "glycosidase",        # hyaluronidase 1
    "HYAL2": "glycosidase",        # hyaluronidase 2
    "HYAL3": "glycosidase",        # hyaluronidase 3
    "HYAL4": "glycosidase",        # hyaluronidase 4
    "SPAM1": "glycosidase",        # sperm adhesion molecule (PH-20 hyaluronidase)
    # ENGase / Chitinases
    "ENGASE": "glycosidase",       # endo-beta-N-acetylglucosaminidase
    "CHIA": "glycosidase",         # chitinase acidic
    "CHIT1": "glycosidase",        # chitotriosidase-1
}

# ---------------------------------------------------------------------------
# Additional glycosyltransferases (poly-LacNAc, blood groups, etc.)
# ---------------------------------------------------------------------------

_OTHER_GT = {
    # Beta-1,3-galactosyltransferases
    "B3GALT1": "other_gt",        # beta-1,3-galactosyltransferase 1
    "B3GALT2": "other_gt",        # beta-1,3-galactosyltransferase 2
    "B3GALT5": "other_gt",        # beta-1,3-galactosyltransferase 5 (type 1 chains)
    # Beta-1,3-N-acetylglucosaminyltransferases
    "B3GNT4": "other_gt",         # beta-1,3-GlcNAcT on globo series
    "B3GNT5": "other_gt",         # lacto/neolacto GSL extension (also in glycolipid)
    "B3GNT9": "other_gt",         # beta-1,3-GlcNAcT
    # Beta-1,4-N-acetylgalactosaminyltransferases
    "B4GALNT2": "other_gt",       # Sd(a)/Cad blood group
    "B4GALNT3": "other_gt",       # beta-1,4-GalNAcT 3
    "B4GALNT4": "other_gt",       # beta-1,4-GalNAcT 4
    # ABO blood group
    "ABO": "other_gt",            # ABO histo-blood group (GTA/GTB)
    # Xylosyltransferases (on EGF repeats, Notch)
    "XXYLT1": "other_gt",         # xyloside xylosyltransferase 1
    "GXYLT1": "other_gt",         # glucoside xylosyltransferase 1
    "GXYLT2": "other_gt",         # glucoside xylosyltransferase 2
    # CHST8/9 (GalNAc-4-sulfotransferase, pituitary)
    "CHST8": "other_gt",
    "CHST9": "other_gt",
    "CHST10": "other_gt",
}

# ---------------------------------------------------------------------------
# Golgi organisation / trafficking affecting glycosylation (COG complex)
# ---------------------------------------------------------------------------

_GOLGI_TRAFFICKING = {
    "COG1": "golgi_trafficking",
    "COG2": "golgi_trafficking",
    "COG3": "golgi_trafficking",
    "COG4": "golgi_trafficking",
    "COG5": "golgi_trafficking",
    "COG6": "golgi_trafficking",
    "COG7": "golgi_trafficking",
    "COG8": "golgi_trafficking",
    "TMEM165": "golgi_trafficking",  # Golgi Mn2+ transporter
    "ATP6V0A2": "golgi_trafficking", # V-ATPase subunit (Golgi pH)
}

# ---------------------------------------------------------------------------
# Merge into single dict
# ---------------------------------------------------------------------------

GLYCO_GENES: dict[str, str] = {}
for _d in [
    _NLINKED_LLO,
    _NLINKED_OST,
    _NLINKED_ER_QC,
    _NLINKED_GOLGI,
    _NLINKED_CAPPING,
    _OLINKED_GALNT,
    _OLINKED_CORE,
    _OMANNOSE,
    _OFUCOSE_OGLUCOSE,
    _OGLCNAC,
    _GAG_LINKER,
    _GAG_HS,
    _GAG_CS_DS,
    _GAG_KS,
    _GAG_HA,
    _GLYCOLIPID,
    _GPI_ANCHOR,
    _SUGAR_NUCLEOTIDE,
    _SLC35_TRANSPORTERS,
    _GLYCOSIDASES,
    _OTHER_GT,
    _GOLGI_TRAFFICKING,
]:
    for gene, pathway in _d.items():
        # Some genes appear in multiple pathway dicts (e.g. FUT1 in
        # nlinked_capping and glycolipid).  Keep the first assignment
        # to avoid overwriting a more-specific pathway label.
        if gene not in GLYCO_GENES:
            GLYCO_GENES[gene] = pathway

ALL_GENE_SYMBOLS: list[str] = sorted(GLYCO_GENES.keys())

# ---------------------------------------------------------------------------
# Pathway category descriptions
# ---------------------------------------------------------------------------

PATHWAY_DESCRIPTIONS: dict[str, str] = {
    "nlinked_llo": "N-linked: dolichol-linked oligosaccharide (LLO) biosynthesis in ER",
    "nlinked_ost": "N-linked: oligosaccharyltransferase (OST) complex",
    "nlinked_er_processing": "N-linked: ER quality control / glucosidases",
    "nlinked_golgi_processing": "N-linked: Golgi processing (trimming, branching, elongation)",
    "nlinked_capping": "N-linked: terminal capping (sialylation, fucosylation)",
    "olinked_initiation": "O-GalNAc: initiation (polypeptide GalNAc-transferases, GALNT family)",
    "olinked_core": "O-GalNAc: core structure synthesis and extension",
    "olinked_capping": "O-GalNAc: sialylation capping",
    "omannose": "O-mannose / dystroglycan glycosylation",
    "ofucose": "O-fucose (Notch signalling)",
    "oglucose": "O-glucose / O-GlcNAc on EGF repeats",
    "oglcnac": "O-GlcNAc cycling (intracellular, nutrient-sensing)",
    "gag_linker": "GAG: linker tetrasaccharide biosynthesis",
    "gag_hs": "GAG: heparan sulfate polymerisation and modification",
    "gag_cs": "GAG: chondroitin sulfate / dermatan sulfate biosynthesis",
    "gag_ks": "GAG: keratan sulfate biosynthesis",
    "gag_ha": "GAG: hyaluronan synthesis",
    "glycolipid": "Glycosphingolipid / glycolipid biosynthesis",
    "gpi_anchor": "GPI-anchor biosynthesis",
    "sugar_nucleotide": "Sugar nucleotide donor biosynthesis",
    "slc35_transporter": "Nucleotide sugar Golgi transporters (SLC35 family)",
    "glycosidase": "Glycan degradation / glycosidases",
    "other_gt": "Other glycosyltransferases (blood groups, poly-LacNAc, etc.)",
    "golgi_trafficking": "Golgi organisation / trafficking affecting glycosylation",
}

# ---------------------------------------------------------------------------
# Helper: convert gene symbols to Ensembl IDs using g:Profiler
# ---------------------------------------------------------------------------


def convert_to_ensembl(
    gene_symbols: list[str] | None = None,
    organism: str = "hsapiens",
) -> dict[str, str]:
    """Convert gene symbols to Ensembl Gene IDs via g:Profiler API.

    Parameters
    ----------
    gene_symbols : list of str, optional
        Gene symbols to convert. Defaults to ALL_GENE_SYMBOLS.
    organism : str
        g:Profiler organism code (default: "hsapiens").

    Returns
    -------
    dict
        Mapping of gene_symbol -> ENSG ID.  Genes that cannot be mapped
        are omitted.

    Requires
    --------
    pip install gprofiler-official
    """
    if gene_symbols is None:
        gene_symbols = ALL_GENE_SYMBOLS

    try:
        from gprofiler import GProfiler
    except ImportError:
        raise ImportError(
            "gprofiler-official is required: pip install gprofiler-official"
        )

    gp = GProfiler(return_dataframe=True)
    result = gp.convert(
        organism=organism,
        query=gene_symbols,
        target_namespace="ENSG",
    )
    mapping: dict[str, str] = {}
    for _, row in result.iterrows():
        sym = row["incoming"]
        ensg = row["converted"]
        if ensg and ensg.startswith("ENSG") and sym not in mapping:
            mapping[sym] = ensg
    return mapping


# ---------------------------------------------------------------------------
# CZ CELLxGENE Census helper
# ---------------------------------------------------------------------------


def build_census_var_filter(
    ensembl_ids: list[str] | None = None,
    max_genes_per_query: int = 500,
) -> list[str]:
    """Build Census API var_value_filter strings for the glyco gene set.

    Parameters
    ----------
    ensembl_ids : list of str, optional
        Ensembl IDs to filter on. If None, calls convert_to_ensembl() first.
    max_genes_per_query : int
        Maximum number of genes per filter string (Census query limit).

    Returns
    -------
    list of str
        Filter strings like: "feature_id in ['ENSG...', 'ENSG...']"
    """
    if ensembl_ids is None:
        mapping = convert_to_ensembl()
        ensembl_ids = list(mapping.values())

    filters = []
    for i in range(0, len(ensembl_ids), max_genes_per_query):
        chunk = ensembl_ids[i : i + max_genes_per_query]
        ids_str = ", ".join(f"'{eid}'" for eid in chunk)
        filters.append(f"feature_id in [{ids_str}]")
    return filters


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    counts = Counter(GLYCO_GENES.values())
    print(f"Total unique glycosylation genes: {len(GLYCO_GENES)}")
    print(f"\nGenes per pathway category:")
    for pathway, count in sorted(counts.items(), key=lambda x: -x[1]):
        desc = PATHWAY_DESCRIPTIONS.get(pathway, "")
        print(f"  {pathway:30s} {count:4d}  {desc}")

    print(f"\nAll gene symbols ({len(ALL_GENE_SYMBOLS)}):")
    for i in range(0, len(ALL_GENE_SYMBOLS), 10):
        print("  " + ", ".join(ALL_GENE_SYMBOLS[i : i + 10]))
