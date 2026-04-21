import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import torch

PREPROCESS_OUT = Path(__file__).resolve().parents[1] / "preprocess" / "out"
SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sampleData"
CAS_PATH = SAMPLE_DATA / "1.cas.h5"
SCHEMA_PATH = PREPROCESS_OUT / "schema.json"
TOPOLOGY_PATH = PREPROCESS_OUT / "topologyMeta.json"
MATRIX_PATH = PREPROCESS_OUT / "snapshotMatrix.npy"

results: list[dict] = []


def report(name: str, ok: bool, detail: str = "", elapsed: float = 0.0):
    tag = "PASS" if ok else "FAIL"
    results.append({"name": name, "ok": ok, "detail": detail, "elapsed": elapsed})
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else "") + (f"  [{elapsed:.2f}s]" if elapsed > 0 else ""))


def step1_loadData():
    print("\n=== Step 1: Load preprocessed data ===")
    t0 = time.time()
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        report("schema.json", True, f"nCells={schema['nCells']}, fields={schema['fieldOrder']}", time.time() - t0)
    except Exception as e:
        report("schema.json", False, str(e))
        return None, None, None
    t0 = time.time()
    try:
        topo = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
        report("topologyMeta.json", True, f"faceZones={topo['counts']['faceZones']}", time.time() - t0)
    except Exception as e:
        report("topologyMeta.json", False, str(e))
        return schema, None, None
    t0 = time.time()
    try:
        matrix = np.load(MATRIX_PATH)
        report("snapshotMatrix.npy", True, f"shape={matrix.shape}, dtype={matrix.dtype}", time.time() - t0)
    except Exception as e:
        report("snapshotMatrix.npy", False, str(e))
        return schema, topo, None
    return schema, topo, matrix


def step2_graphConstruction(topo):
    print("\n=== Step 2: Graph construction (cas.h5) ===")
    from training.shared.graphBuilder import buildCellAdjacencyFast, buildCellAdjacencyTyped, buildEdgeAttr
    t0 = time.time()
    try:
        graph = buildCellAdjacencyFast(CAS_PATH)
        report("buildCellAdjacencyFast", True, f"nCells={graph['nCells']}, nEdges={graph['nEdges']}", time.time() - t0)
    except Exception as e:
        report("buildCellAdjacencyFast", False, str(e))
        return None
    t0 = time.time()
    try:
        graphTyped = buildCellAdjacencyTyped(CAS_PATH, topo)
        nInt = int((graphTyped["edgeType"] == 0).sum())
        nIface = int((graphTyped["edgeType"] == 1).sum())
        nBound = int((graphTyped["edgeType"] == 2).sum())
        report("buildCellAdjacencyTyped", True, f"internal={nInt}, interface={nIface}, boundary={nBound}", time.time() - t0)
    except Exception as e:
        report("buildCellAdjacencyTyped", False, str(e), time.time() - t0)
        return graph
    t0 = time.time()
    try:
        edgeAttr = buildEdgeAttr(graphTyped["edgeType"])
        report("buildEdgeAttr", True, f"shape={edgeAttr.shape}", time.time() - t0)
    except Exception as e:
        report("buildEdgeAttr", False, str(e))
    return graphTyped


def step3_surfaceExtraction(schema, topo, matrix):
    print("\n=== Step 3: Surface extraction ===")
    from training.shared.surfaceExtractor import cellIdsForZones, filterZones, loadFaceOwnerCells, sliceFieldFromMatrix, zoneMeanFromMatrix
    t0 = time.time()
    try:
        faceOwner = loadFaceOwnerCells(CAS_PATH)
        report("loadFaceOwnerCells", True, f"nFaces={faceOwner.size}", time.time() - t0)
    except Exception as e:
        report("loadFaceOwnerCells", False, str(e))
        return
    t0 = time.time()
    inletZones = filterZones(topo, role="boundary", zoneType="inlet")
    report("filterZones(inlet)", True, f"found {len(inletZones)} zones: {[z['name'] for z in inletZones]}", time.time() - t0)
    nCells = schema["nCells"]
    t0 = time.time()
    try:
        inletCellIds = cellIdsForZones(faceOwner, inletZones, maxCellId=nCells)
        report("cellIdsForZones(inlet)", True, f"nCells={inletCellIds.size}", time.time() - t0)
    except Exception as e:
        report("cellIdsForZones(inlet)", False, str(e))
    wallZones = filterZones(topo, role="interface", zoneType="fluidSolidInterface")
    t0 = time.time()
    try:
        wallCellIds = cellIdsForZones(faceOwner, wallZones, maxCellId=nCells)
        report("cellIdsForZones(fluidSolidInterface)", True, f"nCells={wallCellIds.size}", time.time() - t0)
    except Exception as e:
        report("cellIdsForZones(fluidSolidInterface)", False, str(e))
    if matrix is not None:
        t0 = time.time()
        try:
            bcMean = zoneMeanFromMatrix(matrix, schema, inletCellIds, schema["fieldOrder"][:2])
            report("zoneMeanFromMatrix(inlet)", True, f"shape={bcMean.shape}", time.time() - t0)
        except Exception as e:
            report("zoneMeanFromMatrix(inlet)", False, str(e))
        t0 = time.time()
        try:
            surfData = sliceFieldFromMatrix(matrix, schema, "pressure", wallCellIds)
            report("sliceFieldFromMatrix(pressure, interface)", True, f"shape={surfData.shape}", time.time() - t0)
        except Exception as e:
            report("sliceFieldFromMatrix(pressure, interface)", False, str(e))


def step4_modelInstantiation(schema, graph):
    print("\n=== Step 4: Model instantiation ===")
    nCells = schema["nCells"]
    nFields = schema["nFields"]
    from training.endToEnd.model import bcToSurfaceModel
    t0 = time.time()
    try:
        m = bcToSurfaceModel(nInputFeatures=2, nOutputFeatures=100, hiddenSize=32, nLayers=1)
        dummy = torch.randn(1, 5, 2)
        out = m(dummy)
        report("bcToSurfaceModel forward", True, f"out={out.shape}", time.time() - t0)
    except Exception as e:
        report("bcToSurfaceModel forward", False, str(e))
    from training.podRom.model import latentDynamicsModel
    t0 = time.time()
    try:
        m = latentDynamicsModel(nLatent=10, nBcFeatures=2, hiddenSize=32, nLayers=1)
        dummy = torch.randn(1, 5, 12)
        out = m(dummy)
        report("latentDynamicsModel forward", True, f"out={out.shape}", time.time() - t0)
    except Exception as e:
        report("latentDynamicsModel forward", False, str(e))
    from training.pinn.model import pinnModel
    t0 = time.time()
    try:
        m = pinnModel(nSpatialDim=3, nBcParams=2, nOutputFields=nFields, hiddenSize=32, nBlocks=2)
        coords = torch.randn(10, 3, requires_grad=True)
        t_in = torch.randn(10, 1, requires_grad=True)
        bc = torch.randn(10, 2)
        out = m(coords, t_in, bc)
        report("pinnModel forward", True, f"out={out.shape}", time.time() - t0)
    except Exception as e:
        report("pinnModel forward", False, str(e))
    from training.gnn.model import temporalMeshGnn
    t0 = time.time()
    try:
        nSmallNodes = 20
        m = temporalMeshGnn(
            nNodeFeatures=nFields, nOutputFeatures=nFields,
            hiddenSize=16, nConvLayers=2, convType="gat", nHeads=2, nGruLayers=1, edgeDim=3,
        )
        dummyEdge = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                                   [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0]], dtype=torch.long)
        dummyX = torch.randn(1, 4, nSmallNodes, nFields)
        dummyEa = torch.randn(20, 3)
        out = m(dummyX, dummyEdge, dummyEa)
        report("temporalMeshGnn(GAT) forward", True, f"out={out.shape}", time.time() - t0)
    except Exception as e:
        report("temporalMeshGnn(GAT) forward", False, traceback.format_exc())
    t0 = time.time()
    try:
        m = temporalMeshGnn(
            nNodeFeatures=nFields, nOutputFeatures=nFields,
            hiddenSize=16, nConvLayers=2, convType="gcn", nGruLayers=1,
        )
        out = m(torch.randn(1, 4, nSmallNodes, nFields), dummyEdge)
        report("temporalMeshGnn(GCN) forward", True, f"out={out.shape}", time.time() - t0)
    except Exception as e:
        report("temporalMeshGnn(GCN) forward", False, traceback.format_exc())


def step5_gnnRealGraphForward(schema, graph):
    print("\n=== Step 5: GNN forward with REAL graph topology ===")
    from training.shared.graphBuilder import buildEdgeAttr
    from training.gnn.model import temporalMeshGnn
    nCells = graph["nCells"]
    nEdges = graph["nEdges"]
    nFields = schema["nFields"]
    edgeIndex = torch.from_numpy(graph["edgeIndex"])
    edgeAttrNp = buildEdgeAttr(graph["edgeType"])
    edgeAttr = torch.from_numpy(edgeAttrNp)
    print(f"  Real graph: nCells={nCells}, nEdges={nEdges}, edgeIndex={edgeIndex.shape}, edgeAttr={edgeAttr.shape}")
    uniqueCells = torch.unique(edgeIndex.reshape(-1))
    nSubNodes = min(500, uniqueCells.size(0))
    selectedCells = uniqueCells[:nSubNodes]
    maxId = int(selectedCells.max()) + 1
    inSet = torch.zeros(maxId, dtype=torch.bool)
    inSet[selectedCells] = True
    srcInSet = inSet[edgeIndex[0].clamp(max=maxId - 1)] & (edgeIndex[0] < maxId)
    dstInSet = inSet[edgeIndex[1].clamp(max=maxId - 1)] & (edgeIndex[1] < maxId)
    mask = srcInSet & dstInSet
    subEdgeIndex = edgeIndex[:, mask]
    subEdgeAttr = edgeAttr[mask]
    remapTable = torch.full((maxId,), -1, dtype=torch.long)
    remapTable[selectedCells] = torch.arange(nSubNodes)
    subEdgeIndex = remapTable[subEdgeIndex]
    print(f"  Sub-graph ({nSubNodes} cells from real graph): edges={subEdgeIndex.shape[1]}")
    t0 = time.time()
    try:
        m = temporalMeshGnn(
            nNodeFeatures=nFields, nOutputFeatures=nFields,
            hiddenSize=16, nConvLayers=2, convType="gat", nHeads=2, nGruLayers=1, edgeDim=3,
        )
        dummyX = torch.randn(1, 3, nSubNodes, nFields)
        out = m(dummyX, subEdgeIndex, subEdgeAttr)
        report("temporalMeshGnn(GAT) on real sub-graph", True, f"out={out.shape}", time.time() - t0)
    except Exception as e:
        report("temporalMeshGnn(GAT) on real sub-graph", False, traceback.format_exc())


def step6_podSmoke(matrix):
    print("\n=== Step 6: POD basis (single snapshot) ===")
    from training.shared.podBasis import podBasis
    t0 = time.time()
    try:
        pod = podBasis(nModes=1)
        pod.fit(matrix)
        coeffs = pod.encode(matrix)
        report("podBasis fit+encode", True, f"coeffs={coeffs.shape}, energy={pod.truncatedEnergy():.6f}", time.time() - t0)
    except Exception as e:
        report("podBasis fit+encode", False, str(e))


def step7_pygDataCreation(topo):
    print("\n=== Step 7: PyG Data creation ===")
    from training.shared.graphBuilder import buildPygData
    t0 = time.time()
    try:
        data = buildPygData(CAS_PATH, topologyMeta=topo)
        report("buildPygData", True, f"num_nodes={data.num_nodes}, edge_index={data.edge_index.shape}, edge_attr={data.edge_attr.shape}", time.time() - t0)
    except Exception as e:
        report("buildPygData", False, traceback.format_exc())


def printSummary():
    print("\n" + "=" * 60)
    nPass = sum(1 for r in results if r["ok"])
    nFail = sum(1 for r in results if not r["ok"])
    print(f"SUMMARY: {nPass} passed, {nFail} failed, {len(results)} total")
    if nFail > 0:
        print("\nFailed items:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['name']}: {r['detail'][:200]}")
    print("=" * 60)
    nSnapshots = 1
    print(f"\n[WARNING] BLOCKER: snapshotMatrix only has {nSnapshots} snapshot.")
    print("  Training requires multiple time steps (seqLen+1 minimum).")
    print("  Need to run buildDataset with multiple dat.h5 files.")
    return nFail == 0


if __name__ == "__main__":
    schema, topo, matrix = step1_loadData()
    if schema is None:
        print("ABORT: cannot load schema")
        sys.exit(1)
    graph = step2_graphConstruction(topo)
    if topo is not None and matrix is not None:
        step3_surfaceExtraction(schema, topo, matrix)
    if schema is not None:
        step4_modelInstantiation(schema, graph)
    if graph is not None:
        step5_gnnRealGraphForward(schema, graph)
    if matrix is not None:
        step6_podSmoke(matrix)
    if topo is not None:
        step7_pygDataCreation(topo)
    allOk = printSummary()
    sys.exit(0 if allOk else 1)
