from pathlib import Path
from typing import Any

import h5py
import numpy as np

from preprocess.src.ioLayer import parseSemicolonNames


def classifyFaceZone(name: str, zoneType: int, isBoundaryLike: bool) -> dict[str, Any]:
    n = (name or "").lower()
    if "shadow" in n or n.endswith("-shadow"):
        return {"role": "shadow", "type": "shadow", "isBoundaryLike": bool(isBoundaryLike)}
    if "inlet" in n:
        return {"role": "boundary", "type": "inlet", "isBoundaryLike": bool(isBoundaryLike)}
    if "outlet" in n:
        return {"role": "boundary", "type": "outlet", "isBoundaryLike": bool(isBoundaryLike)}
    if "symmetry" in n:
        return {"role": "boundary", "type": "symmetry", "isBoundaryLike": bool(isBoundaryLike)}
    if "wall" in n:
        return {"role": "boundary", "type": "wall", "isBoundaryLike": bool(isBoundaryLike)}
    if "soild" in n or "solid" in n:
        if isBoundaryLike:
            return {"role": "boundary", "type": "solidBoundary", "isBoundaryLike": bool(isBoundaryLike)}
        if "fluid" in n:
            return {"role": "interface", "type": "fluidSolidInterface", "isBoundaryLike": bool(isBoundaryLike)}
    if not isBoundaryLike or zoneType == 2:
        return {"role": "interior", "type": "interior", "isBoundaryLike": bool(isBoundaryLike)}
    if "fluid" in n and ("soild" in n or "solid" in n):
        return {"role": "interface", "type": "fluidSolidInterface", "isBoundaryLike": bool(isBoundaryLike)}
    return {"role": "boundary", "type": f"zoneType{int(zoneType)}", "isBoundaryLike": bool(isBoundaryLike)}


def extractFaceZoneTopology(casePath: Path) -> dict[str, Any]:
    with h5py.File(casePath, "r") as caseFile:
        z = caseFile["meshes/1/faces/zoneTopology"]
        zoneIds = np.asarray(z["id"][()]).reshape(-1)
        minIds = np.asarray(z["minId"][()]).reshape(-1)
        maxIds = np.asarray(z["maxId"][()]).reshape(-1)
        zoneTypes = np.asarray(z["zoneType"][()]).reshape(-1)
        c0 = np.asarray(z["c0"][()]).reshape(-1)
        c1 = np.asarray(z["c1"][()]).reshape(-1)
        shadow = np.asarray(z["shadowZoneId"][()]).reshape(-1)
        names = parseSemicolonNames(np.asarray(z["name"][()]))
    n = int(zoneIds.size)
    zones: list[dict[str, Any]] = []
    for i in range(n):
        name = names[i] if i < len(names) else f"zone_{int(zoneIds[i])}"
        isBoundaryLike = int(c1[i]) == 0
        cls = classifyFaceZone(name, int(zoneTypes[i]), isBoundaryLike=isBoundaryLike)
        zones.append(
            {
                "id": int(zoneIds[i]),
                "name": name,
                "minFaceId": int(minIds[i]),
                "maxFaceId": int(maxIds[i]),
                "zoneType": int(zoneTypes[i]),
                "c0ThreadId": int(c0[i]),
                "c1ThreadId": int(c1[i]),
                "shadowZoneId": int(shadow[i]),
                "isBoundaryLike": isBoundaryLike,
                "category": cls,
            }
        )
    return {"zones": zones}


def extractBoundaryMap(casePath: Path) -> dict[str, Any]:
    topo = extractFaceZoneTopology(casePath)
    zones = topo["zones"]
    boundaryZones = [z for z in zones if z["isBoundaryLike"]]
    byName = {z["name"]: z["id"] for z in boundaryZones}
    return {"boundaryZones": boundaryZones, "boundaryByName": byName}


def extractShadowPairs(casePath: Path) -> list[dict[str, Any]]:
    topo = extractFaceZoneTopology(casePath)
    zones = topo["zones"]
    byId = {z["id"]: z for z in zones}
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for z in zones:
        sid = int(z["shadowZoneId"])
        if sid == 0 or sid not in byId:
            continue
        a = int(z["id"])
        b = int(sid)
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "zoneAId": key[0],
                "zoneBId": key[1],
                "zoneAName": byId[key[0]]["name"],
                "zoneBName": byId[key[1]]["name"],
                "shadowZoneName": byId[key[0]]["name"] if "shadow" in byId[key[0]]["name"].lower() else byId[key[1]]["name"],
            }
        )
    return pairs


def extractInterfacePairs(casePath: Path) -> list[dict[str, Any]]:
    topo = extractFaceZoneTopology(casePath)
    zones = topo["zones"]
    byId = {z["id"]: z for z in zones}
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for z in zones:
        if z["isBoundaryLike"]:
            continue
        a = int(z["c0ThreadId"])
        b = int(z["c1ThreadId"])
        if a <= 0 or b <= 0:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "threadAId": key[0],
                "threadBId": key[1],
                "viaZoneId": int(z["id"]),
                "viaZoneName": z["name"],
                "interfaceType": "fluidSolidInterface"
                if ("soild" in str(z["name"]).lower() or "solid" in str(z["name"]).lower())
                and ("fluid" in str(z["name"]).lower())
                else ("internal" if "internal" in str(z["name"]).lower() else "interface"),
            }
        )
    return pairs


def extractTopologyMeta(casePath: Path) -> dict[str, Any]:
    topo = extractFaceZoneTopology(casePath)
    boundary = extractBoundaryMap(casePath)
    shadowPairs = extractShadowPairs(casePath)
    interfacePairs = extractInterfacePairs(casePath)
    return {
        "faceZones": topo["zones"],
        "boundaryZones": boundary["boundaryZones"],
        "shadowPairs": shadowPairs,
        "interfacePairs": interfacePairs,
        "counts": {
            "faceZones": len(topo["zones"]),
            "boundaryZones": len(boundary["boundaryZones"]),
            "shadowPairs": len(shadowPairs),
            "interfacePairs": len(interfacePairs),
        },
    }
