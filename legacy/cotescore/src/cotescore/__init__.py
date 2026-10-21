"""
cotescore has been replaced by mote_error.

This final release contains no code. It installs mote_error and raises an
ImportError explaining how to migrate.
"""

raise ImportError(
    "cotescore has been replaced by mote_error and is no longer maintained.\n"
    "\n"
    "    pip install mote_error\n"
    "    import mote_error\n"
    "\n"
    "The COTe score is now the MOTe error. With default weights:\n"
    "    MOTe    = 1 - COTe\n"
    "    Missing = 1 - Coverage\n"
    "\n"
    "Renamed functions:\n"
    "    cote_score            -> mote      (returns MOTe, Missing, Overlap, Trespass, Excess)\n"
    "    cote_class            -> mote_class\n"
    "    compute_cote_masks    -> compute_mote_masks\n"
    "    visualize_cote_states -> visualize_mote_states\n"
    "    ClassCOTeResult       -> ClassMOTeResult\n"
    "    ClassCOTeCounts       -> ClassMOTeCounts\n"
    "\n"
    "To keep using the old code, pin the last real release: pip install cotescore==0.3.0\n"
    "More information: https://github.com/THE-3TC/mote_error"
)
