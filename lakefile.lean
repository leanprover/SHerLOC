import Lake
open Lake DSL

package "SHerLOC" where
  -- add package configuration options here

lean_lib «SHerLOC» where
  -- add library configuration options here

@[default_target]
lean_exe "sherloc" where
  root := `Main

require Cli from git
  "https://github.com/leanprover/lean4-cli.git" @ "v4.19.0"

@[test_driver]
lean_exe "test" where
  root := `Test
