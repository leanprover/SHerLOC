"builtin.module"() <{sym_name = "distribution_ops"}> ({
  "func.func"() <{function_type = () -> tensor<ui32>, sym_name = "replica_id"}> ({
    %1 = "stablehlo.replica_id"() : () -> tensor<ui32>
    "func.return"(%1) : (tensor<ui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0:2 = "interpreter.run_parallel"() <{programs = [[@replica_id], [@replica_id]]}> : () -> (tensor<ui32>, tensor<ui32>)
    "check.expect_eq_const"(%0#0) <{value = dense<0> : tensor<ui32>}> : (tensor<ui32>) -> ()
    "check.expect_eq_const"(%0#1) <{value = dense<1> : tensor<ui32>}> : (tensor<ui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

