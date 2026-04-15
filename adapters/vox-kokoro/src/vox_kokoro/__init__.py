from vox_kokoro.torch_adapter import KokoroTorchAdapter

try:
    from vox_kokoro.adapter import KokoroAdapter
except ModuleNotFoundError:
    KokoroAdapter = None

__all__ = ["KokoroAdapter", "KokoroTorchAdapter"]
