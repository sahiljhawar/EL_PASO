# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Sahil Jhawar
#
# SPDX-License-Identifier: Apache-2.0


"""Custom setup.py to install IRBEM Fortran library before building the Python package."""

import subprocess
import os
import sys
import shutil
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
            self._apply_patch()
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

    def _compile_and_install_irbem(self):
        print("Installing IRBEM library...")
        subprocess.check_call(["make"], cwd="IRBEM")    
        subprocess.check_call(["make", "install", "."], cwd="IRBEM")
        subprocess.check_call([sys.executable, "setup.py", "install"], cwd="IRBEM/python")

    def _apply_patch(self):
        irbem_py_src = "IRBEM.py"
        irbem_py_dst = os.path.join("IRBEM", "python", "IRBEM", "IRBEM.py")
        print("Applying custom IRBEM.py patch...")
        os.makedirs(os.path.dirname(irbem_py_dst), exist_ok=True)
        shutil.copy(irbem_py_src, irbem_py_dst)


setup(
    name="el_paso",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    cmdclass={"build_py": CustomBuild},
)
