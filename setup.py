# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


"""Custom setup.py to install IRBEM Fortran library before building the Python package."""

import subprocess
import os
import sys
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class CustomBuild(build_py):
    """Custom build command that builds IRBEM before building Python package."""

    def run(self):
        self.build_irbem()
        super().run()

    def build_irbem(self):
        print("=" * 60)
        print("Building IRBEM Fortran library...")
        print("=" * 60)

        try:
            self._init_submodule()
            self._compile_and_install_irbem()
        except subprocess.CalledProcessError as e:
            print(f"✗ Build failed with return code {e.returncode}")
            print(f"Command: {' '.join(e.cmd)}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            sys.exit(1)

    def _init_submodule(self):
        if not os.path.isdir("IRBEM") or not os.listdir("IRBEM"):
            print("Initializing IRBEM submodule...")
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        else:
            print("✓ IRBEM submodule already present")

    def _get_fortran_compiler_darwin(self):
        try:
            fc = subprocess.check_output(["bash", "-c", "compgen -c gfortran | sort -V | tail -n1"], text=True).strip()
            fc = fc if fc else "gfortran"
        except subprocess.CalledProcessError:
            fc = "gfortran"

        return fc

    def _compile_and_install_irbem(self):
        print("Installing IRBEM library...")
        if sys.platform == "darwin":
            fc = self._get_fortran_compiler_darwin()
            base_cmd = ["make", "OS=osx64", f"FC={fc}", f"LD={fc}"]
            subprocess.check_call(base_cmd + ["all"], cwd="IRBEM")
            subprocess.check_call(base_cmd + ["install"], cwd="IRBEM")
        else:
            subprocess.check_call(["make"], cwd="IRBEM")
            subprocess.check_call(["make", "install", "."], cwd="IRBEM")


setup(
    name="el_paso",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    cmdclass={"build_py": CustomBuild},
)
