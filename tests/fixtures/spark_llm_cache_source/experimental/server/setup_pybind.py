#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Setup script for building TensorRT Edge LLM Python bindings (_edgellm_runtime).

Run from the project root::

    TRT_PACKAGE_DIR=/path/to/tensorrt python experimental/server/setup_pybind.py build_ext --inplace
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake-based extension; sources are built by CMake."""

    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build the extension using CMake."""

    def build_extension(self, ext: CMakeExtension) -> None:
        build_dir = Path.cwd() / "build"
        pybind_out_dir = (build_dir / "pybind").resolve()
        pybind_out_dir.mkdir(parents=True, exist_ok=True)
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={pybind_out_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
        ]

        _extra = os.environ.get("EDGELLM_EXTRA_CMAKE_ARGS", "")
        if _extra:
            cmake_args += _extra.split()

        if os.environ.get("AARCH64_BUILD", "").lower() in (
                "1",
                "true",
                "on",
        ):
            cmake_args.append(
                "-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake")
            cmake_args.append("-DEMBEDDED_TARGET=jetson-thor")

        if os.environ.get("ENABLE_CUTE_DSL_FMHA", "").lower() in (
                "1",
                "true",
                "on",
        ):
            cmake_args.append("-DENABLE_CUTE_DSL_FMHA=ON")

        try:
            import pybind11
            cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            raise RuntimeError(
                "pybind11 not found. Install it: pip install pybind11")

        trt_package_dir = os.environ.get("TRT_PACKAGE_DIR")
        if trt_package_dir:
            cmake_args.append(f"-DTRT_PACKAGE_DIR={trt_package_dir}")
        else:
            raise RuntimeError("TRT_PACKAGE_DIR not set. "
                               "Point it at your TensorRT installation.")

        if os.environ.get("CUDA_DIR"):
            cmake_args.append(f"-DCUDA_DIR={os.environ['CUDA_DIR']}")
        if os.environ.get("CUDA_CTK_VERSION"):
            cmake_args.append(
                f"-DCUDA_CTK_VERSION={os.environ['CUDA_CTK_VERSION']}")

        build_args = ["--config", cfg]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            import multiprocessing
            build_args += ["-j", str(multiprocessing.cpu_count())]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        source_dir = Path(__file__).resolve().parent.parent.parent
        print("Running CMake configure...")
        subprocess.run(
            ["cmake", str(source_dir), *cmake_args],
            cwd=build_temp,
            check=True,
        )
        print("Running CMake build...")
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True,
        )

        ext_filename = self.get_ext_filename(self.get_ext_fullname(ext.name))
        built_so = pybind_out_dir / os.path.basename(ext_filename)
        build_lib_dir = (Path.cwd() / self.build_lib).resolve()
        dest_so = build_lib_dir / ext_filename
        dest_so.parent.mkdir(parents=True, exist_ok=True)
        if built_so.exists():
            shutil.copy2(built_so, dest_so)
        else:
            raise RuntimeError(f"CMake did not produce {built_so}; "
                               f"expected .so in {pybind_out_dir}")


def get_version():
    project_root = Path(__file__).resolve().parent.parent.parent
    version_file = project_root / "tensorrt_edgellm" / "version.py"
    if version_file.exists():
        with open(version_file) as f:
            m = re.search(
                r'__version__\s*=\s*["\']([^"\']+)["\']',
                f.read(),
            )
            if m:
                return m.group(1)
    return "0.6.0"


setup(
    name="tensorrt_edgellm",
    version=get_version(),
    author="NVIDIA Corporation",
    description="TensorRT Edge LLM Python Runtime (C++ extension build)",
    ext_modules=[CMakeExtension("_edgellm_runtime")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    install_requires=["numpy"],
    zip_safe=False,
)
