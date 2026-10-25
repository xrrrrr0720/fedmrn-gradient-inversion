"""Reproducible, local privacy audit of the FedMRN implementation.

Run ``python audit.py --iterations 60`` from any working directory.
The stored historical uploads are read only. Controlled observations are
generated from one real FashionMNIST training image and exact SRN modules.
"""

import argparse
import json
import math
import random
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
FEDMRN = HERE / "fedmrn_code"
TRACES = HERE / "traces"
DATASET_ROOT = HERE / "data"


def snapshot_inventory(root):
    clients = {}
    pattern = re.compile(r"client_(\d+)_round_(\d+)$")
    for folder in Path(root).iterdir():
        match = pattern.fullmatch(folder.name)
        if match and (folder / "local_update.npz").is_file():
            clients.setdefault(match.group(1), []).append(int(match.group(2)))
    for rounds in clients.values():
        rounds.sort()
    return {
        "snapshot_count": sum(len(v) for v in clients.values()),
        "clients": dict(sorted(clients.items(), key=lambda kv: int(kv[0]))),
    }


def map_upload(path, reference_state):
    """Map positional Flower arrays to state_dict names, rejecting mismatches."""
    with np.load(path, allow_pickle=False) as archive:
        expected = [f"arr_{i}" for i in range(len(reference_state))]
        if archive.files != expected:
            raise ValueError(f"unexpected array keys in {path}: {archive.files}")
        result = OrderedDict()
        for key, array_key in zip(reference_state, expected):
            data = archive[array_key]
            if tuple(data.shape) != tuple(reference_state[key].shape):
                raise ValueError(f"shape mismatch at {key}: {data.shape} != {tuple(reference_state[key].shape)}")
            result[key] = torch.from_numpy(data.copy())
    return result


def estimated_gradients(before, after, parameter_names, lr):
    """One-step SGD approximation from *transmitted* model state."""
    if lr <= 0:
        raise ValueError("learning rate must be positive")
    return OrderedDict((name, (before[name].float() - after[name].float()) / lr)
                       for name in parameter_names)


def state_vector(state, names):
    return torch.cat([state[name].float().reshape(-1) for name in names])


def cosine(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def audit_historical():
    inventory = snapshot_inventory(TRACES)
    reference = torch.load(TRACES / "1.pt", map_location="cpu", weights_only=True)
    first = map_upload(TRACES / "client_0_round_1" / "local_update.npz", reference)
    last = map_upload(TRACES / "client_0_round_100" / "local_update.npz", reference)
    names = [k for k, v in reference.items() if v.is_floating_point()]
    count = sum(v.numel() for v in first.values())
    float_count = sum(first[k].numel() for k in names)
    v1, v100 = state_vector(first, names), state_vector(last, names)
    result = {
        **inventory,
        "reference_checkpoint": "traces/1.pt",
        "reference_state_entries": len(reference),
        "first_upload_entries": len(first),
        "matched_entries": len(first),
        "state_elements_per_upload": count,
        "floating_state_elements_per_upload": float_count,
        "array_bytes_first_upload": sum(v.numel() * v.element_size() for v in first.values()),
        "client_0_round_1_to_100_l2": float(torch.linalg.vector_norm(v100 - v1)),
        "client_0_round_1_to_100_cosine": cosine(v1, v100),
        "all_entries_named": list(first.keys()),
    }
    return result


def load_sample(index):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    dataset = datasets.FashionMNIST(str(DATASET_ROOT), train=True,
                                   download=False, transform=transform)
    image, label = dataset[index]
    return image.unsqueeze(0), int(label)


def setup_fedmrn():
    sys.path.insert(0, str(FEDMRN))
    from model.models import get_model
    from model.srn import srn_get_parameters, srn_set_parameters
    return get_model, srn_get_parameters, srn_set_parameters


def transmitted_state(model):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
                       if "update" not in k and "noise" not in k)


def prepare_srn_for_round(model, config, device):
    """Original set_parameters creates fresh CPU noise; transfer after reset."""
    _, _, srn_set_parameters = setup_fedmrn()
    if hasattr(model, "set_parameters"):
        model.set_parameters(config)
    else:
        srn_set_parameters(model, config)
    model.to(device)


def make_observations(image, label, rounds, lr, seed, device):
    get_model, srn_get_parameters, _ = setup_fedmrn()
    checkpoint = torch.load(TRACES / "1.pt", map_location="cpu", weights_only=True)
    model = get_model("srn_cnn4", "fmnist").to(device)
    model.load_state_dict(checkpoint, strict=False)
    target = torch.tensor([label], device=device)
    records = []
    for round_id in range(1, rounds + 1):
        # Same client and same image in every round; one client means its
        # transmitted state is the next global state in this controlled run.
        config = {"mask_type": "binary", "noise_type": "uniform_0.001"}
        random.seed(seed + round_id)
        torch.manual_seed(seed + round_id)
        prepare_srn_for_round(model, config, device)
        before = transmitted_state(model)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(image.to(device)), target)
        loss.backward()
        raw = OrderedDict()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                raw[name.replace(".update", ".weight")] = parameter.grad.detach().cpu().clone()
        optimizer.step()
        srn_get_parameters(model)
        after = transmitted_state(model)
        estimate = estimated_gradients(before, after, list(raw), lr)
        records.append({"round": round_id, "before": before, "after": after,
                        "raw_gradients": raw, "wire_estimate": estimate,
                        "training_loss": float(loss.item())})
    return records


def gradient_cosines(records):
    result = []
    for record in records:
        names = list(record["raw_gradients"])
        raw = state_vector(record["raw_gradients"], names)
        observed = state_vector(record["wire_estimate"], names)
        result.append({"round": record["round"], "cosine_raw_vs_wire": cosine(raw, observed),
                       "raw_norm": float(torch.linalg.vector_norm(raw)),
                       "wire_norm": float(torch.linalg.vector_norm(observed)),
                       "zero_fraction_wire": float((observed == 0).float().mean())})
    return result


def infer_label(classifier_gradient):
    """iDLG-style row-sign heuristic for a bias-free final linear layer."""
    if classifier_gradient.ndim != 2:
        raise ValueError("expected classifier weight matrix")
    return int(classifier_gradient.sum(dim=1).argmin().item())


def attack(records, image, label, source, iterations, seed, device):
    get_model, _, _ = setup_fedmrn()
    chosen = records[:1] if source in ("raw", "single_wire") else records
    targets = [r["raw_gradients"] if source == "raw" else r["wire_estimate"] for r in chosen]
    models = []
    for record in chosen:
        model = get_model("cnn4", "fmnist").to(device)
        model.load_state_dict(record["before"], strict=True)
        model.train()
        models.append(model)
    torch.manual_seed(seed)
    dummy = torch.randn_like(image, device=device).detach().requires_grad_(True)
    initial = dummy.detach().clone()
    optimizer = torch.optim.Adam([dummy], lr=0.08)
    target_label = torch.tensor([label], device=device)
    history = []
    for step in range(iterations):
        optimizer.zero_grad()
        total = torch.zeros((), device=device)
        for model, target in zip(models, targets):
            prediction = model(dummy)
            loss = F.cross_entropy(prediction, target_label)
            names = [name for name, _ in model.named_parameters()]
            gradients = torch.autograd.grad(loss, tuple(model.parameters()), create_graph=True)
            estimate = torch.cat([g.reshape(-1) for g in gradients])
            observed = state_vector(target, names).to(device)
            total = total + (1 - F.cosine_similarity(estimate[None], observed[None]).mean())
        total = total / len(models)
        tv = (dummy[:, :, :, 1:] - dummy[:, :, :, :-1]).abs().mean()
        tv = tv + (dummy[:, :, 1:, :] - dummy[:, :, :-1, :]).abs().mean()
        objective = total + 1e-4 * tv
        objective.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == iterations:
            history.append({"step": step + 1, "objective": float(objective.detach())})
    return initial.detach().cpu(), dummy.detach().cpu(), history


def metrics(reconstruction, truth):
    pred = (reconstruction * 0.3530 + 0.2860).clamp(0, 1)
    original = (truth * 0.3530 + 0.2860).clamp(0, 1)
    mse = float(F.mse_loss(pred, original))
    return {"mse": mse, "psnr_db": -10 * math.log10(max(mse, 1e-12))}


def save_image(tensor, path):
    from PIL import Image
    pixels = (tensor[0, 0] * 0.3530 + 0.2860).clamp(0, 1).mul(255).byte().numpy()
    Image.fromarray(pixels, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 2:
        parser.error("iterations must be >=1 and rounds >=2")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    historical = audit_historical()
    image, label = load_sample(args.sample_index)
    records = make_observations(image, label, args.rounds, args.lr, args.seed, device)
    torch.save({"sample_index": args.sample_index, "label": label,
                "image": image, "observations": records}, out / "controlled_observations.pt")
    save_image(image, out / "ground_truth.png")
    attacks = {}
    for source in ("raw", "single_wire", "multi_wire"):
        print(f"Running {source} inversion ({args.iterations} iterations)", flush=True)
        initial, reconstruction, history = attack(records, image, label, source,
                                                   args.iterations, args.seed, device)
        save_image(reconstruction, out / f"{source}.png")
        attacks[source] = {"initial": metrics(initial, image),
                           "final": metrics(reconstruction, image), "history": history}
    result = {"command_options": vars(args), "torch": torch.__version__,
              "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "historical": historical, "controlled": {"sample_index": args.sample_index,
              "label": label, "gradient_cosines": gradient_cosines(records),
              "attacks": attacks}}
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"historical_count": historical["snapshot_count"],
                      "attacks": {k: v["final"] for k, v in attacks.items()}}, indent=2))


if __name__ == "__main__":
    main()
