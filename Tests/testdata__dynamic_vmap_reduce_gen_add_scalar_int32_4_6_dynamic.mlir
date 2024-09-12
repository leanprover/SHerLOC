"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x6xi32>) -> tensor<?x6xi32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4x6xi32>):
    %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.reduce"(%arg1, %0) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %2 = "stablehlo.add"(%arg2, %arg3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%2) : (tensor<i32>) -> ()
    }) : (tensor<?x4x6xi32>, tensor<i32>) -> tensor<?x6xi32>
    "func.return"(%1) : (tensor<?x6xi32>) -> ()
  }) : () -> ()
}) : () -> ()

