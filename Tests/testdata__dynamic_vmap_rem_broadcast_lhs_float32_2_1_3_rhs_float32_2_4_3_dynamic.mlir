"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x1x3xf32>, tensor<?x2x4x3xf32>) -> tensor<?x2x4x3xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x1x3xf32>, %arg2: tensor<?x2x4x3xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%1, %2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %5) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<?x2x1x3xf32>, tensor<4xi32>) -> tensor<?x2x4x3xf32>
    %7 = "stablehlo.remainder"(%6, %arg2) : (tensor<?x2x4x3xf32>, tensor<?x2x4x3xf32>) -> tensor<?x2x4x3xf32>
    "func.return"(%7) : (tensor<?x2x4x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()

