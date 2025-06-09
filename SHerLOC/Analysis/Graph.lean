import SHerLOC.AST1

namespace StableHLO.Analysis

-- -- GraphViz AST definition

structure AttributeList where
  attributes : List (String × String)
  deriving Repr, Inhabited, Nonempty

structure Vertex where
  id : String
  attributes : AttributeList
  deriving Repr, Inhabited, Nonempty

structure Edge where
  source : String
  dest : String
  attributes : AttributeList
  deriving Repr, Inhabited, Nonempty

structure Graph where
  name : String
  vertices : List Vertex
  edges : List Edge
  deriving Repr, Inhabited, Nonempty

instance : ToString AttributeList where
  toString al :=
    match al with
    | { attributes := [] } => ""
    | { attributes } =>
      let attrStrs := attributes.map (fun (k, v) => s!"{k}=\"{v}\"")
      s!"[{String.intercalate " " attrStrs}];"

instance : ToString Vertex where
  toString v :=
    s!"{v.id} {v.attributes}"

instance : ToString Edge where
  toString e :=
    s!"{e.source} -> {e.dest} {e.attributes}"

instance : ToString Graph where
  toString g :=
    let vertexStrs := g.vertices.map toString
    let edgeStrs := g.edges.map toString
    s!"digraph {g.name} \{\n" ++
      String.intercalate "\n" vertexStrs ++ "\n" ++
      String.intercalate "\n" edgeStrs ++ "\n" ++
      "}"

-- -- Convert the StableHLO AST to a GraphViz representation

def stripOpCodePrefix (op : String) : String :=
  let head := "StableHLO.Parsing.OpCode."
  if op.startsWith head then
    op.drop head.length
  else
    op

def makeArgNode (id : String) : Vertex :=
  Vertex.mk
    s!"node_{id}"
    (AttributeList.mk [
      ("label", s!"Input\\n{id}"),
      ("shape", "diamond"),
      ("style", "filled"),
      ("fillcolor", "lightgreen"),
      ("color", "green")
    ])

def makeOpNode (op : String) (output : String) : Vertex :=
  let op := stripOpCodePrefix op
  Vertex.mk
    s!"node_{output}"
    (AttributeList.mk [
      ("label", s!"{op}\\n{output}"),
    ])
def makeDotNode (output : String) : Vertex :=
  Vertex.mk
    s!"node_{output}"
    (AttributeList.mk [
      ("label", s!"dot_general\\n{output}"),
      ("style", "filled"),
      ("fillcolor", "lightpink"),
      ("color", "red")
    ])

def makeEdge (source : String) (dest : String) : Edge :=
  Edge.mk
    s!"node_{source}"
    s!"node_{dest}"
    (AttributeList.mk [])

def moduleToGraph (m : Parsing.Module) : Graph := Id.run do
  let mut vertices := []
  let mut edges := []
  for {funcBody := .mk funcInputs body, .. } in m.modFuncs do
    for {id, ..} in funcInputs do
      vertices := makeArgNode id :: vertices
    for instr in body do
      match instr with
      | .stablehlo .dotGeneral inputs _ _ outputs _ =>
        vertices := makeDotNode outputs[0]! :: vertices
        edges := inputs.map (fun input => makeEdge input outputs[0]!) ++ edges
      | .stablehlo op inputs _ _ outputs _ =>
        let op := stripOpCodePrefix s!"{repr op}"
        vertices := makeOpNode op outputs[0]! :: vertices
        edges := inputs.map (fun input => makeEdge input outputs[0]!) ++ edges
      | .tanh _ _ =>
        panic! "tanh operation not yet supported in graph representation"
      | .other name _ _ _ outputs _ =>
        vertices := makeOpNode name outputs[0]! :: vertices
      | .return inputs _ =>
        vertices := makeOpNode "return" "return" :: vertices
        edges := inputs.map (fun input => makeEdge input "return") ++ edges
      | .call func_name inputs outputs _ =>
        vertices := makeOpNode (s!"call {func_name}") outputs[0]! :: vertices
        edges := inputs.map (fun input => makeEdge input outputs[0]!) ++ edges
  match m.modId with
    | .none =>
      panic! "Module has no name"
    | .some name =>
      Graph.mk name vertices edges

def modulesToGraph (p : List Parsing.Module) : List Graph :=
  p.map moduleToGraph


end StableHLO.Analysis
