#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
from collections import OrderedDict

from polygraphy import util
from polygraphy.comparator import RunResults
from polygraphy.json import load_json, save_json
from polygraphy.logger import G_LOGGER
from polygraphy.tools.base import Tool


class ToInput(Tool):
    """
    Combines and converts one or more input/output files generated by
    Polygraphy into a single file usable with --load-inputs.
    """

    def __init__(self):
        super().__init__("to-input")

    def add_parser_args(self, parser):
        parser.add_argument(
            "paths", help="Path(s) to file(s) containing input or output data from Polygraphy", nargs="+"
        )
        parser.add_argument("-o", "--output", help="Path to the file to generate", required=True)

    def run(self, args):
        inputs = []

        def update_inputs(new_inputs, path):
            nonlocal inputs

            if inputs and len(inputs) != len(new_inputs):
                G_LOGGER.warning(
                    f"The provided files have different numbers of iterations.\nNote: Inputs currently contains {len(inputs)} iterations, but the data in {path} contains {len(new_inputs)} iterations. Some iterations will contain incomplete data"
                )

            # Pad to appropriate length
            inputs += [OrderedDict()] * (len(new_inputs) - len(inputs))

            for inp, new_inp in zip(inputs, new_inputs):
                inp.update(new_inp)

        for path in args.paths:
            # Note: It's important we have encode/decode JSON methods registered
            # for the types we care about, e.g. RunResults. Importing the class should generally guarantee this.
            data = load_json(path)
            if isinstance(data, RunResults):
                for _, iters in data.items():
                    update_inputs(iters, path)
            else:
                if not util.is_sequence(data):
                    data = [data]
                update_inputs(data, path)

        save_json(inputs, args.output, description=f"input file containing {len(inputs)} iteration(s)")