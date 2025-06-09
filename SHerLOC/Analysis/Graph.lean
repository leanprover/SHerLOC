import SHerLOC.AST1

namespace StableHLO.Analysis

def stripOpCodePrefix (op : String) : String :=
  let head := "StableHLO.Parsing.OpCode."
  if op.startsWith head then
    op.drop head.length
  else
    op

def makeArgNode (id : String) : String :=
  s!"node_{id} [label=\"Input\\n{id}\" shape=diamond style=\"filled\" fillcolor=\"lightgreen\" color=\"green\"];"
def makeOpNode (op : String) (output : String) : String :=
  let op := stripOpCodePrefix op
  s!"node_{output} [label=\"{op}\\n{output}\"];"
def makeDotNode (output : String) : String :=
  s!"node_{output} [label=\"dot_general\\n{output}\" color=\"red\" style=\"filled\" fillcolor=\"lightpink\"];"

def graph (p : List Parsing.Module) : String :=
  let nodes : List String := p.bind (fun {modFuncs, .. } => modFuncs.bind (fun {funcBody := .mk funcInputs body, .. } =>
    funcInputs.map (fun {id, ..} => makeArgNode id) ++
    body.map (fun instr =>
      match instr with
      | .stablehlo .dotGeneral _ _ _ outputs _ =>
        makeDotNode outputs[0]!
      | .stablehlo op _ _ _ outputs _ =>
        let op := stripOpCodePrefix s!"{repr op}"
        makeOpNode op outputs[0]!
      | .tanh _ _ =>
        panic! "tanh operation not yet supported in graph representation"
      | .other name _ _ _ outputs _ =>
        makeOpNode name outputs[0]!
      | .return _ _ => ""
      | .call func_name _ outputs _ =>
        makeOpNode (s!"call {func_name}") outputs[0]!
      )))
  let edges : List String := p.bind (fun {modFuncs, .. } => modFuncs.bind (fun {funcBody := .mk _ body, .. } => body.bind (fun i =>
    match i with
    | .stablehlo _ inputs _ _ outputs _ =>
      inputs.map fun input => s!"node_{input} -> node_{outputs[0]!}"
    | .tanh _ _ =>
      panic! "tanh operation not yet supported in graph representation"
    | .other _ inputs _ _ outputs _ =>
      inputs.map fun input => s!"node_{input} -> node_{outputs[0]!}"
    | .return inputs _ =>
      inputs.map fun input => s!"node_{input} -> return"
    | .call _ inputs outputs _ =>
      inputs.map (fun input => s!"node_{input} -> node_{outputs[0]!}")
    )))
  s!"digraph G \{\n" ++
    String.intercalate "\n" nodes ++ "\n" ++
    String.intercalate "\n" edges ++ "\n" ++
    "}"

end StableHLO.Analysis
