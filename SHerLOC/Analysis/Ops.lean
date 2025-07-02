import SHerLOC.AST1

namespace StableHLO.Analysis

def uniqueOps (p : List Parsing.Module) : List String :=
  let ops := p.flatMap (fun {modFuncs, .. } => modFuncs.flatMap (fun {funcBody := .mk _ body, .. } => body.map (fun i =>
  match i with
  | .stablehlo o _ _ _ _ _ => s!"{repr o}"
  | .tanh _ _ => "tanh"
  | .other name _ _ _ _ _ => name
  | .return _ _ => "return"
  | .call _ _ _ _ => "call"
  )))
  ops.eraseDups

end StableHLO.Analysis
