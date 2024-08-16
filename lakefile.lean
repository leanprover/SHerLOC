import Lake
open Lake DSL

package "SHerLOC" where
  -- add package configuration options here

lean_lib «SHerLOC» where
  -- add library configuration options here

@[default_target]
lean_exe "sherloc" where
  root := `Main
