"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x3xi1>) -> tensor<?x3xi1>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x3xi1>):
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.reduce"(%arg1, %0) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<i1>, %arg3: tensor<i1>):
      %2 = "stablehlo.or"(%arg2, %arg3) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%2) : (tensor<i1>) -> ()
    }) : (tensor<?x2x3xi1>, tensor<i1>) -> tensor<?x3xi1>
    "func.return"(%1) : (tensor<?x3xi1>) -> ()
  }) : () -> ()
}) : () -> ()

