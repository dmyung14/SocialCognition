from interaction_recon.io.discover import DiscoveredMedia


def pair_media(media: list[DiscoveredMedia]) -> tuple[list[dict], list[str]]:
    """Never silently select among duplicate role candidates."""
    groups: dict[str, dict[str, list[str]]] = {}
    unclassified = []

    for item in sorted(media, key=lambda value: (value.id.casefold(), value.id)):
        if item.segment_id is None or item.role is None:
            unclassified.append(item.id)
        if item.segment_id is None:
            continue
        group = groups.setdefault(
            item.segment_id,
            {"guider": [], "builder": [], "unknown": []},
        )
        group[item.role or "unknown"].append(item.id)

    segments = []
    for segment_id in sorted(groups, key=lambda value: (len(value), value)):
        candidates = groups[segment_id]
        guider = candidates["guider"]
        builder = candidates["builder"]
        if len(guider) > 1 or len(builder) > 1:
            status = "ambiguous"
        elif len(guider) == len(builder) == 1:
            status = "paired"
        else:
            status = "unpaired"
        segments.append({
            "segment_id": segment_id,
            "status": status,
            "guider": guider[0] if len(guider) == 1 else None,
            "builder": builder[0] if len(builder) == 1 else None,
            "candidates": candidates,
        })
    return segments, unclassified
