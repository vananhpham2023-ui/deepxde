from __future__ import annotations

try:  # pragma: no cover
    from .cli import *  # type: ignore[F401,F403]
except ImportError:  # pragma: no cover
    from cli import *  # type: ignore[F401,F403]

if __name__ == "__main__":
    try:
        from .cli import main
    except ImportError:
        from cli import main

    main()
