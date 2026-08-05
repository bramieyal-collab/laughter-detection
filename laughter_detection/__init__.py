"""
Laughter detection (Gillick et al.) as an importable library.

Upstream is a set of command-line scripts that find each other through
``sys.path.append('./utils')``. That only works when the interpreter's working directory happens
to be the repository root, and it puts bare names like ``models`` and ``configs`` on the global
import path — where they collide with any application module of the same name. This package
removes both problems: everything lives under ``laughter_detection`` and the modules import each
other relatively.

Typical use::

    from laughter_detection import detect_laughter

    segments = detect_laughter("set.wav")
    # [{'start': 12.4, 'end': 13.1, 'type': 'laughter'}, ...]
"""

from pathlib import Path

__all__ = ["detect_laughter", "DEFAULT_CHECKPOINT_DIR", "DEFAULT_CONFIG"]

#: Ships inside the repository, so an editable install finds it without configuration.
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "in_use" / "resnet_with_augmentation"

DEFAULT_CONFIG = "resnet_with_augmentation"

#: Upstream trained at 8 kHz. Passing anything else silently produces garbage rather than failing,
#: so it is fixed here rather than exposed as a parameter.
SAMPLE_RATE = 8000


def detect_laughter(
    audio_path,
    checkpoint_dir=None,
    threshold=0.5,
    min_length=0.2,
    config_name=DEFAULT_CONFIG,
    device=None,
    batch_size=8,
):
    """
    Find laughter in an audio file.

    :param audio_path: any file librosa can read.
    :param checkpoint_dir: directory holding ``best.pth.tar``. Defaults to the copy in this
        repository.
    :param threshold: probability above which a frame counts as laughter.
    :param min_length: shortest run, in seconds, that survives as an instance.
    :param config_name: key into :data:`configs.CONFIG_MAP`.
    :param device: torch device. Defaults to CUDA when available, otherwise CPU.
    :param batch_size: inference batch size.
    :returns: list of ``{"start": float, "end": float, "type": "laughter"}``, in time order.

    ``num_workers=0`` on the loader is deliberate — the upstream dataset object is not safe to
    fork on Windows, and a worker pool there fails in ways that look like a model error.
    """
    from functools import partial

    import numpy as np
    import torch

    from . import configs, laugh_segmenter
    from .utils import audio_utils, data_loaders, torch_utils

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    checkpoint_file = checkpoint_dir / "best.pth.tar"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            "No model checkpoint at %s. Pass checkpoint_dir= to point at one." % checkpoint_file
        )

    config = configs.CONFIG_MAP[config_name]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config["model"](
        dropout_rate=0.0,
        linear_layer_size=config["linear_layer_size"],
        filter_sizes=config["filter_sizes"],
    )
    model.set_device(device)
    torch_utils.load_checkpoint(str(checkpoint_file), model)
    model.eval()

    dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=str(audio_path), feature_fn=config["feature_fn"], sr=SAMPLE_RATE
    )
    collate_fn = partial(
        audio_utils.pad_sequences_with_labels, expand_channel_dim=config["expand_channel_dim"]
    )
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    probs = []
    for model_inputs, _ in loader:
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        # A single-frame batch squeezes to a 0-d array, which is not iterable.
        probs += [float(preds)] if preds.ndim == 0 else list(preds)

    probs = np.array(probs)
    file_length = audio_utils.get_audio_length(str(audio_path))
    fps = len(probs) / float(file_length)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(
        probs, threshold=threshold, min_length=min_length, fps=fps
    )

    return [{"start": float(i[0]), "end": float(i[1]), "type": "laughter"} for i in instances]
