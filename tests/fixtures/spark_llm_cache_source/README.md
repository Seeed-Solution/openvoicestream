# Spark LLM cache patch fixture

These two Python files are a read-only snapshot of the Spark source after the
historical `spark-llm-serialize-syscache-pybind.patch` baseline and before the
new cache-control patch. They were pulled from:

- `/home/harvest/spark-build/upstream/experimental/server/engine.py`
- `/home/harvest/spark-build/upstream/experimental/server/api_server.py`

Hashes at capture time:

- `engine.py`: `76fc24a5998e61e4f9dac82af10e3a17b6bd021a5bb6120a717eeafa5461b300`
- `api_server.py`: `c3446f848a1887d2aa4545e0c8e0c6e1729fb2f90103953382c2cea5e161acce`
- `setup_pybind.py`: `5ca4705caaa7a0cffbfac3c1047810e195bfac9f184a2c4480bb3b1d72f08b01`

The files retain their upstream SPDX copyright and Apache-2.0 license headers.

The fixture includes `setup_pybind.py` so the complete historical serialization
patch (including its setup buildargs hunk) can be reverse-applied and reapplied
before checking the new cache-control patch.
