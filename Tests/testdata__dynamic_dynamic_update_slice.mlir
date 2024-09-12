"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.dynamic_update_slice"(%arg1, %arg1, %0, %1) : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i64>, tensor<i64>) -> tensor<?x4xf32>
    "func.return"(%2) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

