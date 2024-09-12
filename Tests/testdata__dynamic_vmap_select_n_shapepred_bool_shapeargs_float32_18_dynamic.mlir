"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?xi1>, tensor<?x18xf32>, tensor<?x18xf32>) -> tensor<?x18xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xi1>, %arg2: tensor<?x18xf32>, %arg3: tensor<?x18xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<18> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.concatenate"(%1, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %3) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi1>, tensor<2xi32>) -> tensor<?x18xi1>
    %5 = "stablehlo.select"(%4, %arg3, %arg2) : (tensor<?x18xi1>, tensor<?x18xf32>, tensor<?x18xf32>) -> tensor<?x18xf32>
    "func.return"(%5) : (tensor<?x18xf32>) -> ()
  }) : () -> ()
}) : () -> ()

