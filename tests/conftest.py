# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache-2.0

import typing

import pytest
from _pytest.config.argparsing import ArgumentError, NotSet, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--renew_solution", action="store", default="false")


@pytest.fixture
def renew_solution(request: pytest.FixtureRequest) -> bool:
    def str2bool(v: str) -> bool:
        return v.lower() in ("yes", "true", "t", "1")

    option = request.config.getoption("--renew_solution")
    if option is NotSet:
        msg = "renew_solution not provided!"
        raise ArgumentError(msg, "renew_solution")

    return str2bool(typing.cast("str", option))
