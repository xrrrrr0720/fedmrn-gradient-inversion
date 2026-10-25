import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from audit import map_upload, estimated_gradients, snapshot_inventory, prepare_srn_for_round, infer_label, audit_historical, HERE, FEDMRN, TRACES, DATASET_ROOT


class UploadAuditTests(unittest.TestCase):
    def test_bundle_inputs_are_inside_bundle(self):
        self.assertEqual(FEDMRN, HERE / "fedmrn_code")
        self.assertEqual(TRACES, HERE / "traces")
        self.assertEqual(DATASET_ROOT, HERE / "data")
        self.assertTrue((FEDMRN / "model" / "srn.py").is_file())
        self.assertTrue((TRACES / "1.pt").is_file())
        self.assertTrue((DATASET_ROOT / "FashionMNIST" / "raw" / "train-images-idx3-ubyte").is_file())

    def test_result_checkpoint_path_is_portable(self):
        self.assertEqual(audit_historical()["reference_checkpoint"], "traces/1.pt")

    def test_upload_mapping_preserves_names_and_shapes(self):
        state = OrderedDict([
            ("layer.weight", torch.zeros(2, 3)),
            ("bn.num_batches_tracked", torch.zeros((), dtype=torch.int64)),
        ])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "upload.npz"
            np.savez(path, np.ones((2, 3), dtype=np.float32), np.array(4))
            mapped = map_upload(path, state)
        self.assertEqual(list(mapped), list(state))
        self.assertEqual(tuple(mapped["layer.weight"].shape), (2, 3))
        self.assertEqual(int(mapped["bn.num_batches_tracked"]), 4)

    def test_mismatched_upload_fails_instead_of_silent_padding(self):
        state = OrderedDict([("weight", torch.zeros(2, 3))])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "upload.npz"
            np.savez(path, np.ones((3, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                map_upload(path, state)

    def test_estimated_gradients_skip_buffers_and_preserve_parameter_order(self):
        before = OrderedDict([
            ("weight", torch.tensor([1.0, 2.0])),
            ("bn.running_mean", torch.tensor([0.0])),
        ])
        after = OrderedDict([
            ("weight", torch.tensor([0.9, 1.8])),
            ("bn.running_mean", torch.tensor([0.5])),
        ])
        observed = estimated_gradients(before, after, ["weight"], lr=0.1)
        self.assertTrue(torch.allclose(observed["weight"], torch.tensor([1.0, 2.0])))
        self.assertEqual(len(observed), 1)

    def test_snapshot_inventory_counts_clients_and_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for client, round_id in [(0, 1), (0, 2), (1, 1)]:
                folder = root / f"client_{client}_round_{round_id}"
                folder.mkdir()
                (folder / "local_update.npz").touch()
            result = snapshot_inventory(root)
        self.assertEqual(result["snapshot_count"], 3)
        self.assertEqual(result["clients"], {"0": [1, 2], "1": [1]})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device needed")
    def test_srn_round_preparation_keeps_noise_on_model_device(self):
        import sys
        from audit import FEDMRN
        sys.path.insert(0, str(FEDMRN))
        from model.srn import SRN_Linear
        model = SRN_Linear(2, 2).to("cuda")
        prepare_srn_for_round(model, {"mask_type": "binary", "noise_type": "uniform_0.001"}, "cuda")
        self.assertEqual(model.noise.device.type, "cuda")

    def test_classifier_gradient_identifies_negative_class_row(self):
        classifier = torch.tensor([[0.2, 0.3], [-0.4, -0.2], [0.1, 0.2]])
        self.assertEqual(infer_label(classifier), 1)


if __name__ == "__main__":
    unittest.main()
