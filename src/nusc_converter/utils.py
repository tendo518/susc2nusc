import functools
import hashlib


def expand_scene_ranges(scenes: list[str] | None) -> list[str] | None:
    """
    Expand scene ranges in the format 'scene-XXXXXX_YYYYYY' into a list of individual scenes.
    Example: 'scene-000001_000003' -> ['scene-000001', 'scene-000002', 'scene-000003']
    """
    if scenes is None:
        return None

    expanded_scenes = []
    for item in scenes:
        if "_" in item:
            parts = item.split("_")
            if len(parts) == 2:
                start_str = parts[0]
                end_str = parts[1]
                import re

                match_start = re.search(r"(\d+)$", start_str)

                if match_start:
                    start_num_str = match_start.group(1)
                    prefix = start_str[: -len(start_num_str)]

                    match_end = re.search(r"(\d+)$", end_str)
                    if match_end:
                        end_num_str = match_end.group(1)
                        try:
                            start_num = int(start_num_str)
                            end_num = int(end_num_str)

                            width = len(start_num_str)

                            if start_num <= end_num:
                                for i in range(start_num, end_num + 1):
                                    expanded_scenes.append(f"{prefix}{i:0{width}d}")
                                continue
                        except ValueError:
                            pass

        expanded_scenes.append(item)

    return expanded_scenes


@functools.lru_cache
def generate_token(key: str) -> str:
    """Generate a deterministic token/uuid based on the input key string."""
    m = hashlib.md5()
    m.update(key.encode("utf-8"))
    return m.hexdigest()
