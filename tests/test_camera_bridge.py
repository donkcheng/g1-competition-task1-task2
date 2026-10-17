from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import videohub_camera_server as camera


class G1ChestVideoClientTests(unittest.TestCase):
    def test_uses_g1_chest_service_name(self) -> None:
        with mock.patch.object(camera.Client, "__init__", return_value=None) as init:
            camera.G1ChestVideoClient()
        init.assert_called_once_with("videohub_chest", False)

    def test_registers_g1_image_api(self) -> None:
        client = object.__new__(camera.G1ChestVideoClient)
        with (
            mock.patch.object(client, "_SetApiVerson") as set_version,
            mock.patch.object(client, "_RegistApi") as register,
        ):
            client.Init()
        set_version.assert_called_once_with("1.0.0.1")
        register.assert_called_once_with(1001, 0)


if __name__ == "__main__":
    unittest.main()
