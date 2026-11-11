import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Number of random batches for the heavy equivalence test. Spec says 200; the
# local edit-test loop uses fewer, the Rhombic CPU job sets STAGE0_FULL=1.
N_RANDOM_BATCHES = 200 if os.environ.get("STAGE0_FULL") else 10
