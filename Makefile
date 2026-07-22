SHELL := /usr/bin/bash
.SHELLFLAGS := --noprofile --norc -eu -o pipefail -c

unexport BASH_ENV
unexport ENV
unexport LD_LIBRARY_PATH
unexport LD_PRELOAD
unexport PYTHONHOME
unexport PYTHONPATH

.PHONY: preflight preflight-unit

preflight:
	@/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C LANG=C TZ=UTC \
		/usr/bin/bash --noprofile --norc scripts/preflight.sh

preflight-unit:
	@PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests/unit -p 'test_preflight_unit.py' -v
