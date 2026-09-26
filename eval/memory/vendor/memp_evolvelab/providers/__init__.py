"""Vendored EvolveLab providers (MEMP only).

The upstream providers/__init__.py imports four unrelated providers
(agent_kb, skillweaver, mobilee, expel) that are not vendored here, so this
file replaces it with an empty package marker.  memp_memory_provider.py is
byte-identical to the upstream file; see ../PROVENANCE.txt.
"""
