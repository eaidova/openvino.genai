# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.dirname(__file__))
    import openvino  # add_dll_directory for openvino lib
