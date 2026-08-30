"""Compatibility entry point for the current five-stage release gate.

The former single-page acceptance implementation became invalid when the
patient journey moved to five explicit stages. Keep this filename for existing
automation, but route every invocation through the manifest-driven gate so it
cannot silently test the retired interface.
"""

from run_release_acceptance import main


if __name__ == "__main__":
    main()
